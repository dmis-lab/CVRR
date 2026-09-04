"""Dataset and prompt helpers for held-out CVRR analyses."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, replace
from typing import Any


LETTERS = ("A", "B", "C", "D")
LOCALIZATION_ANSWER_HINT = "\nAnswer the question using a single word or phrase."


@dataclass
class PreparedExample:
    item_id: str
    text: str
    label: str
    image: Any
    multimodal_text: str
    text_only_text: str
    multimodal: dict[str, Any]
    question_ids: Any
    question_attention_mask: Any

    @property
    def question_tokens(self) -> int:
        return int(self.question_attention_mask.sum().item())


def move_prepared(example: PreparedExample, device) -> PreparedExample:
    """Move only tensor fields while retaining immutable prompt/image metadata."""

    return replace(
        example,
        multimodal={key: value.to(device) for key, value in example.multimodal.items()},
        question_ids=example.question_ids.to(device),
        question_attention_mask=example.question_attention_mask.to(device),
    )


def resolve_vstar_root(explicit: str | None = None) -> pathlib.Path:
    if explicit:
        root = pathlib.Path(explicit).expanduser().resolve()
        if not (root / "test_questions.jsonl").is_file():
            raise FileNotFoundError(root / "test_questions.jsonl")
        return root
    from huggingface_hub import snapshot_download

    return pathlib.Path(
        snapshot_download(
            "craigwu/vstar_bench",
            repo_type="dataset",
            local_files_only=True,
        )
    )


def load_vstar_rows(root: pathlib.Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (root / "test_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        row.setdefault("question_id", str(index))
        if row.get("label") not in LETTERS:
            raise ValueError(f"V* row {index} has invalid label {row.get('label')!r}")
    return rows


def configure_visual_cap(processor, model_config, max_visual_tokens: int) -> None:
    if max_visual_tokens <= 0:
        return
    vision = model_config.vision_config
    stride = int(vision.patch_size) * int(vision.spatial_merge_size)
    max_pixels = stride * stride * int(max_visual_tokens)
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        image_processor.size = {
            "shortest_edge": min(int(size.get("shortest_edge", max_pixels)), max_pixels),
            "longest_edge": max_pixels,
        }
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


def text_only_prompt(processor, text: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def multimodal_prompt(processor, text: str) -> str:
    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_example(
    processor,
    *,
    item_id: str,
    text: str,
    label: str,
    image,
    device,
) -> PreparedExample:
    mm_text = multimodal_prompt(processor, text)
    q_text = text_only_prompt(processor, text)
    multimodal = processor(
        text=[mm_text], images=[image], return_tensors="pt"
    )
    question = processor.tokenizer(
        q_text, return_tensors="pt", add_special_tokens=False
    )
    return PreparedExample(
        item_id=str(item_id),
        text=text,
        label=label,
        image=image,
        multimodal_text=mm_text,
        text_only_text=q_text,
        multimodal={key: value.to(device) for key, value in multimodal.items()},
        question_ids=question["input_ids"].to(device),
        question_attention_mask=question["attention_mask"].to(device),
    )


def vstar_example(processor, root: pathlib.Path, row: dict[str, Any], device):
    from PIL import Image

    with Image.open(root / row["image"]) as opened:
        image = opened.convert("RGB")
    return prepare_example(
        processor,
        item_id=str(row["question_id"]),
        text=row["text"],
        label=row["label"],
        image=image,
        device=device,
    )


def blank_like(image, value: int = 127):
    from PIL import Image

    return Image.new("RGB", image.size, color=(value, value, value))


def first_continuation_token(tokenizer, prompt: str, answer: str) -> int:
    """First answer token under the exact serialized chat prompt."""

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer.strip(), add_special_tokens=False).input_ids
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"answer {answer!r} contributed no token")
    return int(full_ids[len(prompt_ids)])


def option_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        pieces = tokenizer.encode(" " + letter, add_special_tokens=False)
        if not pieces:
            raise ValueError(f"could not tokenize option {letter}")
        ids.append(int(pieces[0]))
    return ids


def question_token_lengths(processor, rows: list[dict[str, Any]]) -> list[int]:
    tokenizer = processor.tokenizer
    return [
        len(
            tokenizer(
                text_only_prompt(processor, row["text"]),
                add_special_tokens=False,
            ).input_ids
        )
        for row in rows
    ]


def exact_length_donors(
    rows: list[dict[str, Any]], lengths: list[int]
) -> dict[int, int]:
    """Deterministic same-task, different-label donors with identical shape."""

    donors: dict[int, int] = {}
    for index, row in enumerate(rows):
        candidates = [
            other
            for other, other_row in enumerate(rows)
            if other != index
            and lengths[other] == lengths[index]
            and other_row["label"] != row["label"]
            and other_row.get("category") == row.get("category")
        ]
        if candidates:
            donors[index] = min(
                candidates,
                key=lambda other: (
                    abs(other - index),
                    str(rows[other].get("question_id", other)),
                ),
            )
    return donors


def load_localization_pair_records(path: str | pathlib.Path) -> list[dict[str, Any]]:
    payload = json.loads(pathlib.Path(path).read_text())
    records = payload.get("pairs")
    if not isinstance(records, list):
        raise ValueError("localization pair file must contain a 'pairs' list")
    return records


class PairImageStore:
    """Read localization-pair images from an extracted Visual-CoT root."""

    def __init__(self, *, image_root: str | None = None, index_path: str | None = None):
        if not image_root:
            suffix = f" (legacy index {index_path!r} was supplied)" if index_path else ""
            raise ValueError(
                "the public analysis release requires --image-root containing "
                f"gqa/<image>{suffix}"
            )
        self.image_root = pathlib.Path(image_root).expanduser().resolve()

    def image(self, name: str):
        key = f"gqa/{name}"
        from PIL import Image

        path = self.image_root / key
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            return opened.convert("RGB")
