#!/usr/bin/env python3
"""Convert a research CVRR checkpoint into a clean Hugging Face directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from cvrr import CVRRConfig


def _weight_keys(checkpoint: Path) -> set[str]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    keys: set[str] = set()
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _validate_weights(keys: set[str], recurrent_layer: int) -> None:
    if not keys or any(not key.startswith("backbone.") for key in keys):
        raise ValueError("checkpoint is not a full CVRR wrapper checkpoint")
    expected_lora = {
        f"backbone.model.language_model.layers.{recurrent_layer}.{block}.{projection}."
        f"lora_{side}.default.weight"
        for block, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
        for side in ("A", "B")
    }
    actual_lora = {key for key in keys if ".lora_" in key}
    if actual_lora != expected_lora:
        raise ValueError(
            "checkpoint LoRA tensors do not match released CVRR: "
            f"missing={sorted(expected_lora - actual_lora)}, "
            f"unexpected={sorted(actual_lora - expected_lora)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--processor",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--ell-star", type=int, default=20)
    parser.add_argument("--num-recurrent-steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.01)
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw = json.loads((source / "config.json").read_text())
    base_fields = set(Qwen2_5_VLConfig().to_dict())
    base_config = {
        key: value
        for key, value in raw.items()
        if key in base_fields
        and key not in {"model_type", "architectures", "auto_map", "_name_or_path"}
    }
    config = CVRRConfig(
        **base_config,
        ell_star=args.ell_star,
        num_recurrent_steps=args.num_recurrent_steps,
        beta=args.beta,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        base_model_name_or_path=args.processor,
        per_example_answer_loss=bool(raw.get("per_example_answer_loss", True)),
    )
    keys = _weight_keys(source)
    _validate_weights(keys, config.recurrent_layer)

    for path in source.glob("*.safetensors"):
        shutil.copy2(path, output / path.name)
    index = source / "model.safetensors.index.json"
    if index.is_file():
        shutil.copy2(index, output / index.name)
    config.save_pretrained(output)
    AutoProcessor.from_pretrained(args.processor, use_fast=True).save_pretrained(output)

    package = Path(__file__).resolve().parent / "cvrr"
    shutil.copy2(package / "configuration_cvrr.py", output / "configuration_cvrr.py")
    shutil.copy2(package / "modeling_cvrr.py", output / "modeling_cvrr.py")
    print(f"wrote Hugging Face checkpoint to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
