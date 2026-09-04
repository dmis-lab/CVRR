"""Axis 2: recurrent depth and crossed initial-state/visual-evidence dynamics."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LETTERS,
    LOCALIZATION_ANSWER_HINT,
    PairImageStore,
    first_continuation_token,
    load_localization_pair_records,
    load_vstar_rows,
    option_token_ids,
    prepare_example,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import classification_result, scalar_summary
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace, state_geometry


FACTORIAL_CONDITIONS = (
    "clean_h1_clean_v",
    "wrong_h1_clean_v",
    "clean_h1_wrong_v",
    "wrong_h1_wrong_v",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--factorial-max-steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--vstar-limit", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-factorial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.steps = sorted(set(args.steps))
    if not args.steps or args.steps[0] < 1:
        raise SystemExit("--steps must contain positive integers")
    if args.factorial_max_steps < 1:
        raise SystemExit("--factorial-max-steps must be >=1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    if args.skip_depth and args.skip_factorial:
        raise SystemExit("both recurrence analyses were disabled")


def _jsonl_path(out: pathlib.Path, name: str) -> pathlib.Path:
    return out.with_suffix(out.suffix + f".{name}.jsonl")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _summarize_depth(records, steps, *, n_boot: int, seed: int):
    summary = {}
    for step in steps:
        key = f"T{step}"
        rows = [record["steps"][key] for record in records if key in record["steps"]]
        correctness = [bool(row["correct"]) for row in rows]
        margins = [float(row["gold_margin"]) for row in rows]
        summary[key] = {
            "n": len(rows),
            "accuracy": sum(correctness) / len(correctness) if rows else None,
            "accuracy_stats": scalar_summary(
                (float(value) for value in correctness),
                n_boot=n_boot,
                seed=seed,
            ),
            "gold_margin": scalar_summary(margins, n_boot=n_boot, seed=seed),
        }
        if step != 1:
            paired = [
                record
                for record in records
                if key in record["steps"] and "T1" in record["steps"]
            ]
            summary[key]["vs_T1"] = {
                "accuracy_delta": scalar_summary(
                    (
                        float(record["steps"][key]["correct"])
                        - float(record["steps"]["T1"]["correct"])
                        for record in paired
                    ),
                    n_boot=n_boot,
                    seed=seed,
                ),
                "gold_margin_delta": scalar_summary(
                    (
                        float(record["steps"][key]["gold_margin"])
                        - float(record["steps"]["T1"]["gold_margin"])
                        for record in paired
                    ),
                    n_boot=n_boot,
                    seed=seed + 1,
                ),
            }
    return summary


def _summarize_factorial(records, *, max_steps: int, n_boot: int, seed: int):
    conditions = {}
    for condition in FACTORIAL_CONDITIONS:
        step_summary = {}
        for step in range(1, max_steps + 1):
            rows = [
                record["conditions"][condition][step - 1]
                for record in records
                if condition in record.get("conditions", {})
                and len(record["conditions"][condition]) >= step
            ]
            step_summary[f"T{step}"] = {
                "n": len(rows),
                "target_choice_fraction": (
                    sum(row["pred"] == "target" for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "target_choice_stats": scalar_summary(
                    (float(row["pred"] == "target") for row in rows),
                    n_boot=n_boot,
                    seed=seed,
                ),
                "target_margin": scalar_summary(
                    (row["gold_margin"] for row in rows),
                    n_boot=n_boot,
                    seed=seed,
                ),
            }
        conditions[condition] = step_summary

    def change(condition: str):
        return [
            float(record["conditions"][condition][-1]["gold_margin"])
            - float(record["conditions"][condition][0]["gold_margin"])
            for record in records
            if condition in record.get("conditions", {})
        ]

    correction_eligible = [
        record
        for record in records
        if record["conditions"]["wrong_h1_clean_v"][0]["pred"] == "donor"
    ]
    contamination_eligible = [
        record
        for record in records
        if record["conditions"]["clean_h1_wrong_v"][0]["pred"] == "target"
    ]
    mechanism = {
        "clean_evidence_correction": {
            "final_minus_initial_target_margin": scalar_summary(
                change("wrong_h1_clean_v"), n_boot=n_boot, seed=seed
            ),
            "donor_to_target_flip_fraction": (
                sum(
                    record["conditions"]["wrong_h1_clean_v"][-1]["pred"]
                    == "target"
                    for record in correction_eligible
                )
                / len(correction_eligible)
                if correction_eligible
                else None
            ),
            "eligible_n": len(correction_eligible),
        },
        "wrong_evidence_contamination": {
            "final_minus_initial_target_margin": scalar_summary(
                change("clean_h1_wrong_v"), n_boot=n_boot, seed=seed + 1
            ),
            "target_to_donor_flip_fraction": (
                sum(
                    record["conditions"]["clean_h1_wrong_v"][-1]["pred"]
                    == "donor"
                    for record in contamination_eligible
                )
                / len(contamination_eligible)
                if contamination_eligible
                else None
            ),
            "eligible_n": len(contamination_eligible),
        },
    }
    return {"conditions": conditions, "mechanism": mechanism}


def _run_depth(runtime, args, root, path, existing):
    token_ids = option_token_ids(runtime.processor.tokenizer)
    rows = load_vstar_rows(root)
    if args.vstar_limit:
        rows = rows[: args.vstar_limit]
    completed = {str(record["question_id"]) for record in existing}
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
        and str(row["question_id"]) not in completed
    ]
    max_steps = max(args.steps)
    started = time.monotonic()
    for position, (index, row) in enumerate(indexed, start=1):
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with runtime.torch.inference_mode():
            trace = runtime.extract(example)
            states = runtime.rollout(trace, steps=max_steps)
            step_results = {}
            geometry = {}
            for step in args.steps:
                scores = runtime.token_scores(
                    trace, states[step - 1], token_ids, path="strict"
                )
                step_results[f"T{step}"] = classification_result(
                    scores, LETTERS, row["label"]
                )
            for step, (previous, current) in enumerate(
                zip(states[:-1], states[1:]), start=2
            ):
                geometry[f"T{step}"] = state_geometry(previous, current)
        record = {
            "global_index": index,
            "question_id": str(row["question_id"]),
            "category": row.get("category"),
            "label": row["label"],
            "visual_tokens": trace.visual_tokens,
            "question_tokens": trace.question_tokens,
            "steps": step_results,
            "geometry": geometry,
        }
        _append(path, record)
        existing.append(record)
        release_trace(trace)
        if position % 10 == 0 or position == len(indexed):
            elapsed = time.monotonic() - started
            print(
                f"depth {position}/{len(indexed)} "
                f"eta={(len(indexed)-position)*elapsed/position/60:.1f} min",
                flush=True,
            )
    return existing


def _pair_result(scores: list[float]) -> dict[str, Any]:
    return {
        "scores": scores,
        "pred": "target" if scores[0] >= scores[1] else "donor",
        "correct": scores[0] >= scores[1],
        "gold_margin": float(scores[0] - scores[1]),
    }


def _run_factorial(runtime, args, path, existing):
    pair_records = load_localization_pair_records(args.pairs)
    if args.max_pairs:
        pair_records = pair_records[: args.max_pairs]
    completed = {
        (int(record["pair_index"]), record["direction"]) for record in existing
    }
    store = PairImageStore(image_root=args.image_root, index_path=args.index)
    tokenizer = runtime.processor.tokenizer
    for pair_index in range(args.shard_index, len(pair_records), args.num_shards):
        pair = pair_records[pair_index]
        question = pair["question"].strip() + LOCALIZATION_ANSWER_HINT
        sides = {}
        for side in ("x", "x_prime"):
            sides[side] = prepare_example(
                runtime.processor,
                item_id=f"pair{pair_index}:{side}",
                text=question,
                label=pair[side]["answer"],
                image=store.image(pair[side]["image"]),
                device=runtime.device,
            )
        target_token = first_continuation_token(
            tokenizer, sides["x"].multimodal_text, pair["x"]["answer"]
        )
        donor_token = first_continuation_token(
            tokenizer,
            sides["x_prime"].multimodal_text,
            pair["x_prime"]["answer"],
        )
        if target_token == donor_token:
            print(f"skip pair {pair_index}: first-token collision", flush=True)
            continue
        with runtime.torch.inference_mode():
            traces = {side: runtime.extract(example) for side, example in sides.items()}
            if traces["x"].r1.shape != traces["x_prime"].r1.shape:
                print(f"skip pair {pair_index}: question shape mismatch", flush=True)
                for trace in traces.values():
                    release_trace(trace)
                continue
            if traces["x"].evidence.shape != traces["x_prime"].evidence.shape:
                print(f"skip pair {pair_index}: visual shape mismatch", flush=True)
                for trace in traces.values():
                    release_trace(trace)
                continue
            if not runtime.torch.equal(
                sides["x"].question_ids, sides["x_prime"].question_ids
            ):
                raise RuntimeError(f"pair {pair_index} does not share a tokenized question")

            for target_side, donor_side, direction_tokens in (
                ("x", "x_prime", [target_token, donor_token]),
                ("x_prime", "x", [donor_token, target_token]),
            ):
                direction = f"{target_side}<-{donor_side}"
                if (pair_index, direction) in completed:
                    continue
                target = traces[target_side]
                donor = traces[donor_side]
                specifications = {
                    "clean_h1_clean_v": (target.r1, target.evidence),
                    "wrong_h1_clean_v": (donor.r1, target.evidence),
                    "clean_h1_wrong_v": (target.r1, donor.evidence),
                    "wrong_h1_wrong_v": (donor.r1, donor.evidence),
                }
                conditions = {}
                geometry = {}
                for name, (initial, evidence) in specifications.items():
                    states = runtime.rollout(
                        target,
                        steps=args.factorial_max_steps,
                        initial_state=initial,
                        evidence=evidence,
                    )
                    conditions[name] = [
                        _pair_result(
                            runtime.token_scores(
                                target, state, direction_tokens, path="strict"
                            )
                        )
                        for state in states
                    ]
                    geometry[name] = [
                        state_geometry(previous, current)
                        for previous, current in zip(states[:-1], states[1:])
                    ]
                record = {
                    "pair_index": pair_index,
                    "direction": direction,
                    "question": pair["question"],
                    "target_answer": pair[target_side]["answer"],
                    "donor_answer": pair[donor_side]["answer"],
                    "target_token_id": direction_tokens[0],
                    "donor_token_id": direction_tokens[1],
                    "visual_tokens": target.visual_tokens,
                    "conditions": conditions,
                    "geometry": geometry,
                }
                _append(path, record)
                existing.append(record)
        for trace in traces.values():
            release_trace(trace)
        completed_count = sum(
            int(record["pair_index"]) <= pair_index for record in existing
        )
        if (pair_index + 1) % 10 == 0 or pair_index + 1 == len(pair_records):
            print(
                f"factorial pair {pair_index + 1}/{len(pair_records)} "
                f"direction_records={completed_count}",
                flush=True,
            )
    return existing


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    depth_path = _jsonl_path(out, "depth")
    factorial_path = _jsonl_path(out, "factorial")
    for path in (depth_path, factorial_path):
        if path.exists() and not args.resume:
            raise FileExistsError(f"{path} exists; pass --resume or change --out")
    depth_records = _load_jsonl(depth_path) if args.resume else []
    factorial_records = _load_jsonl(factorial_path) if args.resume else []

    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        adapter_scale=args.adapter_scale,
        offline=not args.allow_download,
    )
    if not args.skip_depth:
        depth_records = _run_depth(
            runtime,
            args,
            resolve_vstar_root(args.vstar_root),
            depth_path,
            depth_records,
        )
    if not args.skip_factorial:
        factorial_records = _run_factorial(
            runtime, args, factorial_path, factorial_records
        )

    payload = {
        "kind": "cvrr_recurrence_mechanism",
        "protocol": {
            "depth_dataset": "V*",
            "factorial_dataset": "same-question/different-image GQA pairs",
            "depths": args.steps,
            "factorial_max_steps": args.factorial_max_steps,
            "crossed_scaffold": (
                "target geometry is fixed; only post-cell visual-row values or "
                "the initial question state are replaced"
            ),
        },
        "runtime": runtime.metadata(),
        "depth": {
            "n": len(depth_records),
            "summary": _summarize_depth(
                depth_records,
                args.steps,
                n_boot=args.bootstrap,
                seed=args.seed,
            ),
            "records": depth_records,
        },
        "factorial": {
            "n": len(factorial_records),
            "summary": _summarize_factorial(
                factorial_records,
                max_steps=args.factorial_max_steps,
                n_boot=args.bootstrap,
                seed=args.seed,
            ),
            "records": factorial_records,
        },
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
