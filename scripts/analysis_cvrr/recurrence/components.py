"""Matched inference-time component ablations for canonical CVRR."""

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
from scripts.analysis_cvrr.core.metrics import (
    exact_mcnemar_p,
    holm_adjust,
    scalar_summary,
)
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace, state_geometry


CONDITIONS = (
    "full_cvrr",
    "without_recurrence",
    "without_native_mm_initialization",
    "without_learned_transition",
    "without_entire_visual_path",
)


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


def _condition_summary(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for condition in CONDITIONS:
        rows = [record["conditions"][condition] for record in records]
        correctness = [bool(row["correct"]) for row in rows]
        margins = [float(row["gold_margin"]) for row in rows]
        summary: dict[str, Any] = {
            "n": len(rows),
            "accuracy": sum(correctness) / len(correctness) if rows else None,
            "accuracy_stats": scalar_summary(
                (float(value) for value in correctness),
                n_boot=bootstrap,
                seed=seed,
            ),
            "gold_margin": scalar_summary(
                margins, n_boot=bootstrap, seed=seed + 1
            ),
        }
        if condition != "full_cvrr" and rows:
            full = [record["conditions"]["full_cvrr"] for record in records]
            accuracy_delta = [
                float(row["correct"]) - float(reference["correct"])
                for reference, row in zip(full, rows)
            ]
            margin_delta = [
                float(row["gold_margin"]) - float(reference["gold_margin"])
                for reference, row in zip(full, rows)
            ]
            p_value = exact_mcnemar_p(
                [bool(row["correct"]) for row in full], correctness
            )
            raw_p[condition] = p_value
            summary["vs_full"] = {
                "accuracy_delta": scalar_summary(
                    accuracy_delta, n_boot=bootstrap, seed=seed
                ),
                "gold_margin_delta": scalar_summary(
                    margin_delta, n_boot=bootstrap, seed=seed + 1
                ),
                "prediction_flip_fraction": sum(
                    reference["pred"] != row["pred"]
                    for reference, row in zip(full, rows)
                )
                / len(rows),
                "regress_fraction": sum(
                    bool(reference["correct"]) and not bool(row["correct"])
                    for reference, row in zip(full, rows)
                )
                / len(rows),
                "improve_fraction": sum(
                    not bool(reference["correct"]) and bool(row["correct"])
                    for reference, row in zip(full, rows)
                )
                / len(rows),
                "mcnemar_exact_p": p_value,
            }
        output[condition] = summary
    for condition, adjusted in holm_adjust(raw_p).items():
        output[condition]["vs_full"]["mcnemar_holm_p"] = adjusted
    return output


def summarize(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        categories[str(record.get("category", "unknown"))].append(record)
    return {
        "overall": _condition_summary(records, bootstrap=bootstrap, seed=seed),
        "per_category": {
            category: _condition_summary(rows, bootstrap=bootstrap, seed=seed)
            for category, rows in sorted(categories.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 2:
        raise SystemExit("--steps must be >=2 for the component ablation")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")

    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    records_path = out.with_suffix(out.suffix + ".records.jsonl")
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"{records_path} exists; pass --resume")
    records: list[dict[str, Any]] = []
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
            full = runtime.rollout(trace, steps=args.steps)
            no_native = runtime.rollout(
                trace,
                steps=args.steps,
                initial_state=trace.base_anchor,
                evidence=trace.evidence,
            )
            no_learned = runtime.rollout(
                trace,
                steps=args.steps,
                recurrent_adapters=False,
            )
            no_visual = runtime.rollout(
                trace,
                steps=args.steps,
                visual_access="no_visual",
            )
            states = {
                "full_cvrr": full[-1],
                "without_recurrence": trace.r1,
                "without_native_mm_initialization": no_native[-1],
                "without_learned_transition": no_learned[-1],
                "without_entire_visual_path": no_visual[-1],
            }
            results = {
                condition: runtime.token_scores(
                    trace, state, token_ids, path="strict"
                )
                for condition, state in states.items()
            }
            classified = {}
            for condition, scores in results.items():
                pred_index = max(range(len(scores)), key=scores.__getitem__)
                gold_index = LETTERS.index(row["label"])
                classified[condition] = {
                    "scores": [float(score) for score in scores],
                    "pred": LETTERS[pred_index],
                    "correct": pred_index == gold_index,
                    "gold_margin": float(scores[gold_index])
                    - max(
                        float(score)
                        for candidate, score in enumerate(scores)
                        if candidate != gold_index
                    ),
                }
            geometry = {
                "full_cvrr": [
                    state_geometry(previous, current)
                    for previous, current in zip(full[:-1], full[1:])
                ],
                "without_native_mm_initialization": [
                    state_geometry(previous, current)
                    for previous, current in zip(no_native[:-1], no_native[1:])
                ],
                "without_learned_transition": [
                    state_geometry(previous, current)
                    for previous, current in zip(no_learned[:-1], no_learned[1:])
                ],
                "without_entire_visual_path": [
                    state_geometry(previous, current)
                    for previous, current in zip(no_visual[:-1], no_visual[1:])
                ],
            }

        record = {
            "global_index": index,
            "question_id": str(row["question_id"]),
            "category": row.get("category"),
            "label": row["label"],
            "visual_tokens": trace.visual_tokens,
            "question_tokens": trace.question_tokens,
            "conditions": classified,
            "geometry": geometry,
        }
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        records.append(record)
        release_trace(trace)
        if position % 5 == 0 or position == len(indexed):
            elapsed = time.monotonic() - started
            eta = (len(indexed) - position) * elapsed / position / 60
            print(
                f"component {position}/{len(indexed)} eta={eta:.1f} min",
                flush=True,
            )

    payload = {
        "kind": "cvrr_component_ablation",
        "protocol": {
            "dataset": "V*",
            "steps": args.steps,
            "conditions": {
                "full_cvrr": "native multimodal h1, persistent v, learned transition",
                "without_recurrence": "native multimodal h1 decoded at T=1",
                "without_native_mm_initialization": (
                    "text-only B initialization, persistent clean v, learned transition"
                ),
                "without_learned_transition": (
                    "native multimodal h1 and persistent v, frozen native cell only"
                ),
                "without_entire_visual_path": (
                    "text-only B initialization and text-only learned recurrence"
                ),
            },
            "ablation_scope": "matched inference-time structural intervention",
        },
        "runtime": runtime.metadata(),
        "n": len(records),
        "summary": summarize(
            records, bootstrap=args.bootstrap, seed=args.seed
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
