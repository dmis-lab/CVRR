#!/usr/bin/env python3
"""Train CVRR with answer-token cross entropy."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

from cvrr import CVRRConfig, CVRRForConditionalGeneration
from cvrr.data import VisualCoTCollator, load_dataset_split


def _read_config(path: str) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    for section in ("model", "data", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"configuration requires a {section!r} mapping")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args()
    recipe = _read_config(args.config)
    model_cfg = dict(recipe["model"])
    data_cfg = dict(recipe["data"])
    train_cfg = dict(recipe["training"])
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    if "output_dir" not in train_cfg:
        raise ValueError("training.output_dir is required")

    seed = int(train_cfg.get("seed", 0))
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("tf32", True))

    base_model = model_cfg.pop(
        "base_model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    init_checkpoint = model_cfg.pop("init_checkpoint", "")
    processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
    config = CVRRConfig.from_pretrained(
        base_model,
        base_model_name_or_path=base_model,
        **model_cfg,
    )
    if init_checkpoint:
        model, loading = CVRRForConditionalGeneration.from_pretrained(
            init_checkpoint,
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
    else:
        model = CVRRForConditionalGeneration.from_backbone(
            base_model,
            config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    counts = model.freeze_for_training()
    print(
        f"trainable={counts['trainable']:,} / total={counts['total']:,} "
        f"({100.0 * counts['trainable'] / counts['total']:.4f}%)",
        flush=True,
    )

    dataset_path = data_cfg["path"]
    train_dataset = load_dataset_split(
        dataset_path, data_cfg.get("train_split", "train")
    )
    eval_split = data_cfg.get("validation_split", "")
    eval_dataset = (
        load_dataset_split(dataset_path, eval_split) if eval_split else None
    )
    collator = VisualCoTCollator(
        processor,
        max_visual_tokens=int(data_cfg.get("max_visual_tokens", 8192)),
    )

    training_args = TrainingArguments(
        **train_cfg,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

    metadata = {
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "trainable_parameters": counts["trainable"],
        "total_parameters": counts["total"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
