"""Axis 5: ablate persistent visual access while preserving recurrent depth."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LETTERS,
    load_vstar_rows,
    option_token_ids,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import classification_result, scalar_summary
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace, state_geometry


CONDITIONS = ("persistent_every_step", "first_read_only", "no_visual")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _summarize(records, *, steps: int, bootstrap: int, seed: int):
    summary = {}
    for condition in CONDITIONS:
        by_step = {}
        for step in range(1, steps + 1):
            rows = [record["conditions"][condition][step - 1] for record in records]
            by_step[f"T{step}"] = {
                "n": len(rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "accuracy_stats": scalar_summary(
                    (float(bool(row["correct"])) for row in rows),
                    n_boot=bootstrap,
                    seed=seed,
                ),
                "gold_margin": scalar_summary(
                    (row["gold_margin"] for row in rows),
                    n_boot=bootstrap,
                    seed=seed,
                ),
            }
        changes = [
            float(record["conditions"][condition][-1]["gold_margin"])
            - float(record["conditions"][condition][0]["gold_margin"])
            for record in records
        ]
        summary[condition] = {
            "steps": by_step,
            "final_minus_initial_margin": scalar_summary(
                changes, n_boot=bootstrap, seed=seed + 1
            ),
        }

    reference = "persistent_every_step"
    for condition in ("first_read_only", "no_visual"):
        correctness_delta = [
            float(record["conditions"][condition][-1]["correct"])
            - float(record["conditions"][reference][-1]["correct"])
            for record in records
        ]
        margin_delta = [
            float(record["conditions"][condition][-1]["gold_margin"])
            - float(record["conditions"][reference][-1]["gold_margin"])
            for record in records
        ]
        summary[condition]["vs_persistent_final"] = {
            "accuracy_delta": scalar_summary(
                correctness_delta, n_boot=bootstrap, seed=seed
            ),
            "gold_margin_delta": scalar_summary(
                margin_delta, n_boot=bootstrap, seed=seed + 1
            ),
        }

    category_values = defaultdict(lambda: defaultdict(list))
    for record in records:
        category = str(record.get("category", "unknown"))
        for condition in CONDITIONS:
            category_values[category][condition].append(
                record["conditions"][condition][-1]
            )
    per_category = {}
    for category, condition_rows in sorted(category_values.items()):
        per_category[category] = {}
        for condition, rows in condition_rows.items():
            per_category[category][condition] = {
                "n": len(rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "accuracy_stats": scalar_summary(
                    (float(bool(row["correct"])) for row in rows),
                    n_boot=bootstrap,
                    seed=seed,
                ),
                "gold_margin": scalar_summary(
                    (row["gold_margin"] for row in rows),
                    n_boot=bootstrap,
                    seed=seed,
                ),
            }
    return {"overall": summary, "per_category": per_category}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1:
        raise SystemExit("--steps must be >=1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    records_path = out.with_suffix(out.suffix + ".records.jsonl")
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"{records_path} exists; pass --resume")
    records = []
    if args.resume and records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line.strip()
        ]
    completed = {str(record["question_id"]) for record in records}

    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        adapter_scale=args.adapter_scale,
        offline=not args.allow_download,
    )
    root = resolve_vstar_root(args.vstar_root)
    rows = load_vstar_rows(root)
    if args.limit:
        rows = rows[: args.limit]
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
        and str(row["question_id"]) not in completed
    ]
    token_ids = option_token_ids(runtime.processor.tokenizer)
    started = time.monotonic()
    for position, (index, row) in enumerate(indexed, start=1):
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with runtime.torch.inference_mode():
            trace = runtime.extract(example)
            trajectories = {
                "persistent_every_step": runtime.rollout(
                    trace, steps=args.steps, visual_access="every_step"
                ),
                "first_read_only": runtime.rollout(
                    trace, steps=args.steps, visual_access="first_only"
                ),
                "no_visual": runtime.rollout(
                    trace, steps=args.steps, visual_access="no_visual"
                ),
            }
            results = {
                condition: [
                    classification_result(
                        runtime.token_scores(
                            trace, state, token_ids, path="strict"
                        ),
                        LETTERS,
                        row["label"],
                    )
                    for state in states
                ]
                for condition, states in trajectories.items()
            }
            geometry = {
                condition: [
                    state_geometry(previous, current)
                    for previous, current in zip(states[:-1], states[1:])
                ]
                for condition, states in trajectories.items()
            }
        record = {
            "global_index": index,
            "question_id": str(row["question_id"]),
            "category": row.get("category"),
            "label": row["label"],
            "visual_tokens": trace.visual_tokens,
            "question_tokens": trace.question_tokens,
            "conditions": results,
            "geometry": geometry,
        }
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        records.append(record)
        release_trace(trace)
        if position % 10 == 0 or position == len(indexed):
            elapsed = time.monotonic() - started
            print(
                f"persistent {position}/{len(indexed)} "
                f"eta={(len(indexed)-position)*elapsed/position/60:.1f} min",
                flush=True,
            )

    payload = {
        "kind": "cvrr_persistent_visual_ablation",
        "protocol": {
            "dataset": "V*",
            "steps": args.steps,
            "persistent_every_step": "R1 plus normal q-query to visual-KV at T>=2",
            "first_read_only": (
                "R1 is native multimodal; q-query to visual-KV edges are blocked "
                "at every later transition, while q self-attention and MLP remain"
            ),
            "no_visual": (
                "starts from text-only B and repeats the shared native cell in "
                "the text-only scaffold; no visual rows are present"
            ),
        },
        "runtime": runtime.metadata(),
        "n": len(records),
        "summary": _summarize(
            records,
            steps=args.steps,
            bootstrap=args.bootstrap,
            seed=args.seed,
        ),
        "records": records,
        "meta": run_metadata(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(out)
    print(f"wrote {out}", flush=True)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
