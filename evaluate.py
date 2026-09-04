#!/usr/bin/env python3
"""Evaluate CVRR on V*, MMVP, BLINK, and MME-RealWorld-Lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

from cvrr import CVRRConfig, CVRRForConditionalGeneration
from cvrr.benchmarks import extract_final_letter, iter_benchmark, summarize


def _load_recipe(path: str) -> dict:
    with open(path) as handle:
        recipe = yaml.safe_load(handle)
    if not isinstance(recipe, dict):
        raise ValueError("evaluation config must be a mapping")
    return recipe


def _set_visual_cap(processor, max_visual_tokens: int) -> None:
    if max_visual_tokens <= 0:
        return
    image_processor = processor.image_processor
    stride = int(image_processor.patch_size) * int(image_processor.merge_size)
    max_pixels = stride * stride * max_visual_tokens
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        image_processor.size = {
            "shortest_edge": min(
                int(size.get("shortest_edge", max_pixels)), max_pixels
            ),
            "longest_edge": max_pixels,
        }
    elif hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


def _encode(processor, image, prompt: str, device: torch.device, dtype: torch.dtype):
    multimodal_prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    multimodal = processor(
        text=[multimodal_prompt], images=[image], return_tensors="pt"
    )
    text_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    question = processor.tokenizer(
        text_prompt, return_tensors="pt", add_special_tokens=False
    )
    multimodal = {key: value.to(device) for key, value in multimodal.items()}
    question = {key: value.to(device) for key, value in question.items()}
    multimodal["pixel_values"] = multimodal["pixel_values"].to(dtype)
    return multimodal, question


def _letter_token_ids(tokenizer) -> dict[str, int]:
    """Return the exact restricted option tokens used by the V* analysis."""
    result = {}
    for letter in "ABCDE":
        pieces = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not pieces:
            raise ValueError(f"could not tokenize option {letter}")
        result[letter] = int(pieces[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    recipe = _load_recipe(args.config)
    checkpoint = args.checkpoint or recipe["checkpoint"]
    output_dir = Path(args.output_dir or recipe["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(recipe.get("device", "cuda:0"))
    protocol = recipe.get("prediction_mode", "greedy")
    if protocol not in {"greedy", "choice_logits"}:
        raise ValueError("prediction_mode must be greedy or choice_logits")

    config = CVRRConfig.from_pretrained(checkpoint)
    if "beta" in recipe:
        config.beta = float(recipe["beta"])
        config.validate_cvrr()
    model, loading = CVRRForConditionalGeneration.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise RuntimeError(
            "checkpoint did not load exactly: "
            f"missing={loading['missing_keys'][:8]}, "
            f"unexpected={loading['unexpected_keys'][:8]}"
        )
    model = model.to(device).eval()
    processor_source = recipe.get(
        "processor", config.base_model_name_or_path
    )
    processor = AutoProcessor.from_pretrained(processor_source, use_fast=True)
    _set_visual_cap(processor, int(recipe.get("max_visual_tokens", 8192)))
    letter_token_ids = _letter_token_ids(processor.tokenizer)
    max_new_tokens = int(recipe.get("max_new_tokens", 32))
    local_files_only = bool(recipe.get("local_files_only", False))

    all_summaries = {}
    for benchmark in recipe.get(
        "benchmarks", ["vstar", "mmvp", "blink", "mme_lite"]
    ):
        rows = []
        iterator = iter_benchmark(
            benchmark,
            shard=args.shard,
            num_shards=args.num_shards,
            limit=args.limit,
            local_files_only=local_files_only,
        )
        for item in tqdm(iterator, desc=benchmark):
            multimodal, question = _encode(
                processor, item["image"], item["prompt"], device, model.dtype
            )
            if protocol == "choice_logits":
                with torch.inference_mode():
                    logits = model.next_token_logits(
                        input_ids=multimodal["input_ids"],
                        attention_mask=multimodal["attention_mask"],
                        pixel_values=multimodal["pixel_values"],
                        image_grid_thw=multimodal["image_grid_thw"],
                        question_ids=question["input_ids"],
                        question_attention_mask=question["attention_mask"],
                    )[0, 0]
                scores = {
                    letter: float(logits[letter_token_ids[letter]])
                    for letter in item["letters"]
                }
                prediction = max(scores, key=scores.get)
                raw_output = None
            else:
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=multimodal["input_ids"],
                        attention_mask=multimodal["attention_mask"],
                        pixel_values=multimodal["pixel_values"],
                        image_grid_thw=multimodal["image_grid_thw"],
                        question_ids=question["input_ids"],
                        question_attention_mask=question["attention_mask"],
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                raw_output = processor.tokenizer.decode(
                    generated[0], skip_special_tokens=True
                ).strip()
                prediction = extract_final_letter(raw_output)
                scores = None
            row = {
                key: item[key]
                for key in ("global_index", "qid", "benchmark", "task", "answer")
            }
            row.update(
                prediction=prediction,
                correct=prediction == item["answer"],
                output=raw_output,
                scores=scores,
            )
            rows.append(row)

        prediction_path = output_dir / f"{benchmark}.predictions.jsonl"
        with prediction_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = summarize(rows)
        summary["protocol"] = {
            "prediction_mode": protocol,
            "max_new_tokens": max_new_tokens if protocol == "greedy" else None,
            "max_visual_tokens": int(recipe.get("max_visual_tokens", 8192)),
            "beta": config.beta,
            "shard": args.shard,
            "num_shards": args.num_shards,
        }
        (output_dir / f"{benchmark}.summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        all_summaries[benchmark] = summary

    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
