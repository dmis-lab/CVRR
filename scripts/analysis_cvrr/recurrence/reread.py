"""Crossed-state test of whether recurrent steps actually re-read visual evidence.

For same-question/different-image contrastive pairs, this combines the two
controls that must be crossed to identify re-reading:

* donor initial question state with target visual evidence (correction), and
* target initial question state with donor visual evidence (contamination),

each with text-query-to-visual-key/value edges either available or blocked at
every transition after R1.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LOCALIZATION_ANSWER_HINT,
    PairImageStore,
    first_continuation_token,
    load_localization_pair_records,
    prepare_example,
)
from scripts.analysis_cvrr.core.metrics import scalar_summary
from scripts.analysis_cvrr.recurrence.depth import _pair_result
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


STATE_CONDITIONS = (
    "clean_h1_clean_v",
    "wrong_h1_clean_v",
    "clean_h1_wrong_v",
)
READ_MODES = {
    "read_on": "every_step",
    "read_off": "first_only",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _append(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _trajectory_change(record: dict[str, Any], condition: str) -> float:
    states = record["conditions"][condition]
    return float(states[-1]["gold_margin"]) - float(states[0]["gold_margin"])


def _summarize(
    records: list[dict[str, Any]], *, steps: int, bootstrap: int, seed: int
) -> dict[str, Any]:
    conditions = {}
    names = [
        f"{state}_{mode}"
        for state in STATE_CONDITIONS
        for mode in READ_MODES
    ]
    for condition_index, condition in enumerate(names):
        step_summary = {}
        for step in range(1, steps + 1):
            rows = [record["conditions"][condition][step - 1] for record in records]
            choices = [float(row["pred"] == "target") for row in rows]
            step_summary[f"T{step}"] = {
                "n": len(rows),
                "target_choice_fraction": sum(choices) / len(choices),
                "target_choice_stats": scalar_summary(
                    choices,
                    n_boot=bootstrap,
                    seed=seed + condition_index,
                ),
                "target_margin": scalar_summary(
                    (row["gold_margin"] for row in rows),
                    n_boot=bootstrap,
                    seed=seed + condition_index + 1,
                ),
            }
        conditions[condition] = step_summary

    correction = []
    contamination = []
    correction_choice = []
    contamination_choice = []
    for record in records:
        correction_on = "wrong_h1_clean_v_read_on"
        correction_off = "wrong_h1_clean_v_read_off"
        contamination_on = "clean_h1_wrong_v_read_on"
        contamination_off = "clean_h1_wrong_v_read_off"
        correction.append(
            _trajectory_change(record, correction_on)
            - _trajectory_change(record, correction_off)
        )
        # Positive means that reading wrong evidence causes additional loss of
        # target margin beyond the no-read trajectory.
        contamination.append(
            _trajectory_change(record, contamination_off)
            - _trajectory_change(record, contamination_on)
        )
        correction_choice.append(
            float(record["conditions"][correction_on][-1]["pred"] == "target")
            - float(record["conditions"][correction_off][-1]["pred"] == "target")
        )
        contamination_choice.append(
            float(record["conditions"][contamination_off][-1]["pred"] == "target")
            - float(record["conditions"][contamination_on][-1]["pred"] == "target")
        )

    return {
        "conditions": conditions,
        "identified_reread_effect": {
            "clean_v_additional_correction": {
                "target_margin_gain": scalar_summary(
                    correction, n_boot=bootstrap, seed=seed + 100
                ),
                "target_choice_gain": scalar_summary(
                    correction_choice, n_boot=bootstrap, seed=seed + 101
                ),
            },
            "wrong_v_additional_contamination": {
                "target_margin_loss": scalar_summary(
                    contamination, n_boot=bootstrap, seed=seed + 102
                ),
                "target_choice_loss": scalar_summary(
                    contamination_choice, n_boot=bootstrap, seed=seed + 103
                ),
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 2:
        raise SystemExit("--steps must be >=2")
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
    completed = {
        (int(record["pair_index"]), str(record["direction"]))
        for record in records
    }

    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        offline=not args.allow_download,
    )
    pairs = load_localization_pair_records(args.pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    store = PairImageStore(image_root=args.image_root, index_path=args.index)
    tokenizer = runtime.processor.tokenizer
    torch = runtime.torch

    for pair_index in range(args.shard_index, len(pairs), args.num_shards):
        pair = pairs[pair_index]
        question = pair["question"].strip() + LOCALIZATION_ANSWER_HINT
        examples = {
            side: prepare_example(
                runtime.processor,
                item_id=f"pair{pair_index}:{side}",
                text=question,
                label=pair[side]["answer"],
                image=store.image(pair[side]["image"]),
                device=runtime.device,
            )
            for side in ("x", "x_prime")
        }
        tokens = {
            side: first_continuation_token(
                tokenizer,
                examples[side].multimodal_text,
                pair[side]["answer"],
            )
            for side in ("x", "x_prime")
        }
        if tokens["x"] == tokens["x_prime"]:
            continue
        with torch.inference_mode():
            traces = {
                side: runtime.extract(example) for side, example in examples.items()
            }
            if traces["x"].r1.shape != traces["x_prime"].r1.shape:
                for trace in traces.values():
                    release_trace(trace)
                continue
            if traces["x"].evidence.shape != traces["x_prime"].evidence.shape:
                for trace in traces.values():
                    release_trace(trace)
                continue
            if not torch.equal(examples["x"].question_ids, examples["x_prime"].question_ids):
                raise RuntimeError(f"pair {pair_index} tokenized questions differ")

            for target_side, donor_side in (("x", "x_prime"), ("x_prime", "x")):
                direction = f"{target_side}<-{donor_side}"
                if (pair_index, direction) in completed:
                    continue
                target = traces[target_side]
                donor = traces[donor_side]
                specifications = {
                    "clean_h1_clean_v": (target.r1, target.evidence),
                    "wrong_h1_clean_v": (donor.r1, target.evidence),
                    "clean_h1_wrong_v": (target.r1, donor.evidence),
                }
                direction_tokens = [tokens[target_side], tokens[donor_side]]
                conditions = {}
                for state_name, (initial, evidence) in specifications.items():
                    for mode_name, visual_access in READ_MODES.items():
                        name = f"{state_name}_{mode_name}"
                        states = runtime.rollout(
                            target,
                            steps=args.steps,
                            initial_state=initial,
                            evidence=evidence,
                            visual_access=visual_access,
                        )
                        conditions[name] = [
                            _pair_result(
                                runtime.token_scores(
                                    target, state, direction_tokens, path="strict"
                                )
                            )
                            for state in states
                        ]
                record = {
                    "pair_index": pair_index,
                    "direction": direction,
                    "question": pair["question"],
                    "target_answer": pair[target_side]["answer"],
                    "donor_answer": pair[donor_side]["answer"],
                    "visual_tokens": target.visual_tokens,
                    "conditions": conditions,
                }
                _append(records_path, record)
                records.append(record)
                completed.add((pair_index, direction))
        for trace in traces.values():
            release_trace(trace)
        print(f"reread pair {pair_index + 1}/{len(pairs)}", flush=True)

    payload = {
        "kind": "cvrr_reread_correction",
        "protocol": {
            "dataset": "same-question/different-image GQA pairs",
            "steps": args.steps,
            "read_on": "normal Q-to-visual KV edges at every transition",
            "read_off": "all Q-to-visual KV edges blocked after supplied R1",
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
