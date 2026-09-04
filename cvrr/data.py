"""Visual-CoT loading and preprocessing for CVRR training."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _pad(sequences: list[torch.Tensor], value: int) -> torch.Tensor:
    width = max(sequence.shape[0] for sequence in sequences)
    output = torch.full(
        (len(sequences), width), value, dtype=sequences[0].dtype
    )
    for index, sequence in enumerate(sequences):
        output[index, : sequence.shape[0]] = sequence
    return output


def _open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"unsupported image value: {type(value)!r}")


def _chat_prompt(processor, question: str, hint: str, *, image: bool) -> str:
    content = [{"type": "text", "text": question.strip() + hint}]
    if image:
        content.insert(0, {"type": "image"})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


class VisualCoTCollator:
    """Materialize model inputs from the fixed Visual-CoT Arrow records.

    Expected fields are ``image_bytes`` (or ``image``), ``fixed_question``,
    ``fixed_answer``, and optionally ``fixed_hint``.  MC formatting is assumed
    to have been fixed before the dataset split and stored in these fields.
    """

    def __init__(self, processor, max_visual_tokens: int = 8192) -> None:
        self.processor = processor
        self.max_visual_tokens = int(max_visual_tokens)

    def _vision_kwargs(self) -> dict[str, Any]:
        if self.max_visual_tokens <= 0:
            return {}
        image_processor = self.processor.image_processor
        stride = int(image_processor.patch_size) * int(image_processor.merge_size)
        max_pixels = stride * stride * self.max_visual_tokens
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict) and "longest_edge" in size:
            return {
                "size": {
                    "shortest_edge": min(
                        int(size.get("shortest_edge", max_pixels)), max_pixels
                    ),
                    "longest_edge": max_pixels,
                }
            }
        return {"max_pixels": max_pixels}

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        examples = []
        for record in records:
            image_value = record.get("image_bytes", record.get("image"))
            image = _open_image(image_value)
            question = str(record.get("fixed_question", record.get("question", "")))
            answer = str(record.get("fixed_answer", record.get("answer", "")))
            hint = str(record.get("fixed_hint", record.get("hint", "")))
            if not question or not answer:
                raise ValueError("every record needs a non-empty question and answer")

            multimodal = self.processor(
                text=[_chat_prompt(self.processor, question, hint, image=True)],
                images=[image],
                return_tensors="pt",
                **self._vision_kwargs(),
            )
            text_prompt = _chat_prompt(self.processor, question, hint, image=False)
            text_only = tokenizer(
                text_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            answer_ids = torch.tensor(
                tokenizer(" " + answer, add_special_tokens=False).input_ids
                + [tokenizer.eos_token_id],
                dtype=torch.long,
            )
            examples.append(
                {
                    "input_ids": multimodal["input_ids"][0],
                    "attention_mask": multimodal["attention_mask"][0],
                    "pixel_values": multimodal["pixel_values"],
                    "image_grid_thw": multimodal["image_grid_thw"],
                    "question_ids": text_only["input_ids"][0],
                    "question_attention_mask": text_only["attention_mask"][0],
                    "answer_ids": answer_ids,
                }
            )

        answers = [example["answer_ids"] for example in examples]
        return {
            "input_ids": _pad(
                [example["input_ids"] for example in examples],
                tokenizer.pad_token_id,
            ),
            "attention_mask": _pad(
                [example["attention_mask"] for example in examples], 0
            ),
            "pixel_values": torch.cat(
                [example["pixel_values"] for example in examples], dim=0
            ),
            "image_grid_thw": torch.cat(
                [example["image_grid_thw"] for example in examples], dim=0
            ),
            "question_ids": _pad(
                [example["question_ids"] for example in examples],
                tokenizer.pad_token_id,
            ),
            "question_attention_mask": _pad(
                [example["question_attention_mask"] for example in examples], 0
            ),
            "answer_ids": _pad(answers, tokenizer.pad_token_id),
            "labels": _pad(answers, -100),
        }


def load_dataset_split(path: str | Path, split: str):
    """Load a saved HF Dataset or a sharded Arrow manifest split."""
    from datasets import concatenate_datasets, load_from_disk

    path = Path(path).expanduser().resolve()
    manifest_path = path if path.is_file() else path / "manifest.json"
    if not manifest_path.is_file():
        dataset_path = path / split if (path / split).is_dir() else path
        return load_from_disk(str(dataset_path))

    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("splits", {}).get(split)
    if not entries:
        raise ValueError(f"manifest has no {split!r} split: {manifest_path}")
    shards = []
    for entry in entries:
        recorded = Path(entry["path"])
        relocated = manifest_path.parent / split / recorded.name
        shard_path = recorded if recorded.exists() else relocated
        if not shard_path.exists():
            raise FileNotFoundError(
                f"dataset shard is absent at {recorded} and {relocated}"
            )
        shards.append(load_from_disk(str(shard_path)))
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


__all__ = ["VisualCoTCollator", "load_dataset_split"]
