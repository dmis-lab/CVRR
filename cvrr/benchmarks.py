"""Shared benchmark loading and scoring for the CVRR release."""

from __future__ import annotations

import collections
import csv
import io
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError


BLINK_TASKS = (
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
)

DATASETS = {
    "vstar": "craigwu/vstar_bench",
    "blink": "BLINK-Benchmark/BLINK",
    "mmvp": "MMVP/MMVP",
    "mme_lite": "mm-eval/MME-RealWorld-Lite",
}


def dataset_root(name: str, *, local_files_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    if name not in DATASETS:
        raise ValueError(f"unknown benchmark {name!r}")
    return Path(
        snapshot_download(
            DATASETS[name],
            repo_type="dataset",
            local_files_only=local_files_only,
        )
    )


def normalize_letter(value: object) -> str:
    match = re.search(
        r"(?:^|[^A-Za-z])\(?([A-Ea-e])\)?(?:[^A-Za-z]|$)", str(value)
    )
    return match.group(1).upper() if match else ""


def extract_final_letter(value: object) -> str:
    if not isinstance(value, str):
        return ""
    patterns = (
        r"<answer>\s*\(?\s*([A-E])\s*\)?(?:\s*</answer>)?",
        r"\\boxed\s*\{\s*\(?\s*([A-E])\s*\)?\s*\}",
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\(?\s*([A-E])",
        r"(?:option|choice)\s*\(?\s*([A-E])",
    )
    for pattern in patterns:
        hits = re.findall(pattern, value, flags=re.IGNORECASE)
        if hits:
            return hits[-1].upper()
    stripped = re.sub(r"(?:\s*<\|[^>]+\|>)+\s*$", "", value).rstrip()
    match = re.search(r"(?:^|\s|\()([A-E])(?:\)?[.!]?\s*)$", stripped, re.I)
    return match.group(1).upper() if match else ""


def stitch_images(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("benchmark item has no image")
    images = [image.convert("RGB") for image in images]
    if len(images) == 1:
        return images[0]
    gap = 10
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    left = 0
    for image in images:
        canvas.paste(image, (left, (height - image.height) // 2))
        left += image.width + gap
    return canvas


def _decode_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    raise TypeError(f"unsupported image value: {type(value)!r}")


def _selected(index: int, shard: int, num_shards: int, limit: int) -> bool:
    return (not limit or index < limit) and index % num_shards == shard


def _iter_vstar(
    root: Path, *, shard: int, num_shards: int, limit: int
) -> Iterator[dict]:
    with (root / "test_questions.jsonl").open() as handle:
        for index, line in enumerate(handle):
            if limit and index >= limit:
                return
            if not _selected(index, shard, num_shards, limit):
                continue
            row = json.loads(line)
            category = str(row["category"])
            yield {
                "global_index": index,
                "qid": str(row.get("question_id", index)),
                "benchmark": "vstar",
                "task": category,
                "prompt": str(row["text"]),
                "answer": normalize_letter(row["label"]),
                "letters": list("ABCD"),
                "image": Image.open(root / row["image"]).convert("RGB"),
            }


def _iter_blink(
    root: Path, *, shard: int, num_shards: int, limit: int
) -> Iterator[dict]:
    import pyarrow.parquet as pq

    index = 0
    for task in BLINK_TASKS:
        parquet = pq.ParquetFile(next(root.glob(f"{task}/val-*.parquet")))
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                current = index
                index += 1
                if limit and current >= limit:
                    return
                if not _selected(current, shard, num_shards, limit):
                    continue
                images = [
                    _decode_image(row[key])
                    for key in ("image_1", "image_2", "image_3", "image_4")
                    if row.get(key) is not None
                ]
                yield {
                    "global_index": current,
                    "qid": str(row.get("idx", f"{task}_{current}")),
                    "benchmark": "blink",
                    "task": task,
                    "prompt": str(row["prompt"]).strip()
                    + "\nAnswer with only the letter of the correct option.",
                    "answer": normalize_letter(row["answer"]),
                    "letters": [
                        chr(ord("A") + offset)
                        for offset in range(len(row["choices"]))
                    ],
                    "image": stitch_images(images),
                }


def _iter_mmvp(
    root: Path, *, shard: int, num_shards: int, limit: int
) -> Iterator[dict]:
    with (root / "Questions.csv").open() as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit and index >= limit:
                return
            if not _selected(index, shard, num_shards, limit):
                continue
            qid = int(row["Index"])
            letters = [
                value.upper()
                for value in re.findall(r"\(([a-e])\)", row["Options"], re.I)
            ]
            yield {
                "global_index": index,
                "qid": str(qid),
                "benchmark": "mmvp",
                "task": "MMVP",
                "prompt": (
                    f"{row['Question'].strip()}\n{row['Options'].strip()}\n"
                    "Answer with only the letter of the correct option."
                ),
                "answer": normalize_letter(row["Correct Answer"]),
                "letters": letters or ["A", "B"],
                "image": Image.open(root / "MMVP Images" / f"{qid}.jpg").convert(
                    "RGB"
                ),
            }


def _mme_prompt(messages: object) -> str:
    messages = json.loads(messages) if isinstance(messages, str) else messages
    message = messages[-1]
    choices = [str(choice).strip() for choice in message["choices"]]
    parts = [str(message["question"]).strip()]
    if message.get("hint"):
        parts.append(str(message["hint"]).strip())
    parts.extend(
        [
            "The choices are listed below:",
            *choices,
            "Select the best answer based on the image. Respond with only the "
            "letter (A, B, C, D, or E) of the correct option.",
            "The best answer is:",
        ]
    )
    return "\n".join(parts)


def _iter_mme_lite(
    root: Path, *, shard: int, num_shards: int, limit: int
) -> Iterator[dict]:
    import pyarrow.parquet as pq

    index = 0
    for path in sorted((root / "data").glob("test-*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32):
            rows = batch.to_pylist()
            for row in rows:
                current = index
                index += 1
                if limit and current >= limit:
                    return
                if not _selected(current, shard, num_shards, limit):
                    continue
                try:
                    images = [_decode_image(value) for value in row["media"]]
                except (UnidentifiedImageError, OSError) as error:
                    pair_key = re.sub(r"_q\d+$", "", str(row["id"]))
                    images = None
                    for sibling in rows:
                        if sibling is row or re.sub(
                            r"_q\d+$", "", str(sibling["id"])
                        ) != pair_key:
                            continue
                        try:
                            candidate = [
                                _decode_image(value) for value in sibling["media"]
                            ]
                        except (UnidentifiedImageError, OSError):
                            continue
                        if len(candidate) == len(row["media"]):
                            images = candidate
                            break
                    if images is None:
                        raise error
                yield {
                    "global_index": current,
                    "qid": str(row["id"]),
                    "benchmark": "mme_lite",
                    "task": str(row["task"]),
                    "prompt": _mme_prompt(row["messages"]),
                    "answer": normalize_letter(row["answer"]),
                    "letters": list("ABCDE"),
                    "image": stitch_images(images),
                }


def iter_benchmark(
    name: str,
    *,
    shard: int = 0,
    num_shards: int = 1,
    limit: int = 0,
    local_files_only: bool = False,
) -> Iterator[dict]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"invalid shard {shard}/{num_shards}")
    root = dataset_root(name, local_files_only=local_files_only)
    loaders = {
        "vstar": _iter_vstar,
        "blink": _iter_blink,
        "mmvp": _iter_mmvp,
        "mme_lite": _iter_mme_lite,
    }
    yield from loaders[name](root, shard=shard, num_shards=num_shards, limit=limit)


def _score(values: Sequence[bool]) -> dict:
    return {
        "accuracy": 100.0 * sum(values) / len(values) if values else 0.0,
        "correct": int(sum(values)),
        "count": len(values),
    }


def summarize(records: Iterable[dict]) -> dict:
    records = list(records)
    by_task: dict[str, list[bool]] = collections.defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(bool(record["correct"]))
    summary = {
        "overall": _score([bool(record["correct"]) for record in records]),
        "tasks": {key: _score(value) for key, value in sorted(by_task.items())},
    }
    pairs: dict[int, dict[int, bool]] = collections.defaultdict(dict)
    for record in records:
        if record["benchmark"] != "mmvp":
            continue
        qid = int(record["qid"])
        if qid < 1:
            raise ValueError("MMVP question IDs must be positive")
        pair = pairs[(qid - 1) // 2]
        if qid in pair:
            raise ValueError(f"duplicate MMVP question ID: {qid}")
        pair[qid] = bool(record["correct"])
    incomplete_pairs = sum(len(pair) != 2 for pair in pairs.values())
    if incomplete_pairs:
        # Item shards may split pairs; score only after their records are merged.
        summary["incomplete_pairs"] = incomplete_pairs
    elif pairs:
        summary["pair"] = _score(
            [all(pair.values()) for pair in pairs.values()]
        )
    return summary


__all__ = [
    "BLINK_TASKS",
    "extract_final_letter",
    "iter_benchmark",
    "normalize_letter",
    "summarize",
]
