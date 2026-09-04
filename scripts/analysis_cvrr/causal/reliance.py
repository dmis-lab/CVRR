"""Axis 1: semantic reliance under strict and restored-bypass paths."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LETTERS,
    blank_like,
    exact_length_donors,
    load_vstar_rows,
    option_token_ids,
    prepare_example,
    question_token_lengths,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import (
    classification_result,
    exact_mcnemar_p,
    scalar_summary,
    summarize_condition_records,
)
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, norm_matched_noise, release_trace


CONDITIONS = ("clean", "matched_swap", "blank", "norm_matched_noise")
PATHS = (
    "strict",
    "visual_rows_restored",
    "dual_path_bypass",
    "full_mm_oracle",
    # Backward-compatible alias for the original visual-row proxy.
    "bypass_restored",
)
DEFAULT_PATHS = (
    "strict",
    "visual_rows_restored",
    "dual_path_bypass",
    "full_mm_oracle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=0, help="0 uses checkpoint T")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blank-value", type=int, default=127)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument(
        "--paths", nargs="+", choices=PATHS, default=list(DEFAULT_PATHS)
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    if not 0 <= args.blank_value <= 255:
        raise SystemExit("--blank-value must lie in [0,255]")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")


def _record_path(out: pathlib.Path) -> pathlib.Path:
    return out.with_suffix(out.suffix + ".records.jsonl")


def _load_existing(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _residual_bank(
    runtime: CVRRRuntime,
    root: pathlib.Path,
    rows: list[dict[str, Any]],
    indices: set[int],
    *,
    steps: int,
) -> dict[int, Any]:
    torch = runtime.torch
    bank = {}
    for completed, index in enumerate(sorted(indices), start=1):
        example = vstar_example(runtime.processor, root, rows[index], runtime.device)
        with torch.inference_mode():
            trace = runtime.extract(example)
            state = runtime.rollout(trace, steps=steps)[-1]
            bank[index] = (state - trace.base_anchor).detach().cpu()
        release_trace(trace)
        del trace, state, example
        if completed % 20 == 0 or completed == len(indices):
            print(f"donor residuals {completed}/{len(indices)}", flush=True)
    return bank


def _bypass_sensitivity(summary: dict[str, Any]) -> dict[str, Any]:
    strict = summary.get("strict", {})
    restored = summary.get("bypass_restored", {})
    output = {}
    for condition in CONDITIONS:
        if condition == "clean" or condition not in strict or condition not in restored:
            continue
        strict_delta = strict[condition].get("vs_clean", {}).get("accuracy_delta", {})
        bypass_delta = restored[condition].get("vs_clean", {}).get("accuracy_delta", {})
        if strict_delta.get("mean") is None or bypass_delta.get("mean") is None:
            continue
        strict_drop = -float(strict_delta["mean"])
        restored_drop = -float(bypass_delta["mean"])
        output[condition] = {
            "strict_accuracy_drop": strict_drop,
            "restored_bypass_accuracy_drop": restored_drop,
            "sensitivity_reduction": strict_drop - restored_drop,
        }
    return output


def _path_sensitivity(summary: dict[str, Any]) -> dict[str, Any]:
    """Accuracy-drop attenuation for every non-strict answer path."""

    strict = summary.get("strict", {})
    output = {}
    for path, path_summary in summary.items():
        if path == "strict" or not isinstance(path_summary, dict):
            continue
        conditions = {}
        for condition in CONDITIONS:
            if condition == "clean" or condition not in path_summary:
                continue
            strict_delta = strict.get(condition, {}).get("vs_clean", {}).get(
                "accuracy_delta", {}
            )
            path_delta = path_summary[condition].get("vs_clean", {}).get(
                "accuracy_delta", {}
            )
            if strict_delta.get("mean") is None or path_delta.get("mean") is None:
                continue
            strict_drop = -float(strict_delta["mean"])
            restored_drop = -float(path_delta["mean"])
            conditions[condition] = {
                "strict_accuracy_drop": strict_drop,
                "path_accuracy_drop": restored_drop,
                "sensitivity_reduction": strict_drop - restored_drop,
            }
        output[path] = conditions
    return output


def _path_interactions(
    records: list[dict[str, Any]], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Paired difference-in-differences for bypass attenuation.

    The accuracy interaction is

    ``(strict_clean - strict_changed) - (path_clean - path_changed)``.

    Positive values therefore mean that restoring the path makes an
    intervention less damaging. Complete examples, rather than answer tokens,
    are the bootstrap unit.
    """

    paths = sorted(
        {
            path
            for record in records
            for path in record.get("results", {})
            if path != "strict"
        }
    )
    output: dict[str, Any] = {}
    for path_index, path in enumerate(paths):
        clean_records = [
            record
            for record in records
            if "clean" in record.get("results", {}).get("strict", {})
            and "clean" in record.get("results", {}).get(path, {})
        ]
        strict_clean = [
            bool(record["results"]["strict"]["clean"]["correct"])
            for record in clean_records
        ]
        path_clean = [
            bool(record["results"][path]["clean"]["correct"])
            for record in clean_records
        ]
        path_output: dict[str, Any] = {
            "clean_vs_strict": {
                "accuracy_delta": scalar_summary(
                    (
                        float(after) - float(before)
                        for before, after in zip(strict_clean, path_clean)
                    ),
                    n_boot=n_boot,
                    seed=seed + 100 * path_index,
                ),
                "mcnemar_exact_p": exact_mcnemar_p(strict_clean, path_clean),
            },
            "intervention_attenuation": {},
        }
        conditions = sorted(
            {
                condition
                for record in records
                for condition in record.get("results", {}).get(path, {})
                if condition != "clean"
            }
        )
        for condition_index, condition in enumerate(conditions):
            paired = [
                record
                for record in records
                if all(
                    name in record.get("results", {}).get(answer_path, {})
                    for answer_path in ("strict", path)
                    for name in ("clean", condition)
                )
            ]
            accuracy_interaction = []
            margin_interaction = []
            for record in paired:
                strict = record["results"]["strict"]
                restored = record["results"][path]
                strict_accuracy_drop = float(strict["clean"]["correct"]) - float(
                    strict[condition]["correct"]
                )
                path_accuracy_drop = float(restored["clean"]["correct"]) - float(
                    restored[condition]["correct"]
                )
                accuracy_interaction.append(
                    strict_accuracy_drop - path_accuracy_drop
                )
                strict_margin_drop = float(strict["clean"]["gold_margin"]) - float(
                    strict[condition]["gold_margin"]
                )
                path_margin_drop = float(restored["clean"]["gold_margin"]) - float(
                    restored[condition]["gold_margin"]
                )
                margin_interaction.append(strict_margin_drop - path_margin_drop)
            interaction_seed = seed + 100 * path_index + 2 * condition_index + 1
            path_output["intervention_attenuation"][condition] = {
                "n": len(paired),
                "accuracy_drop_reduction": scalar_summary(
                    accuracy_interaction,
                    n_boot=n_boot,
                    seed=interaction_seed,
                ),
                "gold_margin_drop_reduction": scalar_summary(
                    margin_interaction,
                    n_boot=n_boot,
                    seed=interaction_seed + 1,
                ),
            }
        output[path] = path_output
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    records_path = _record_path(out)
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"{records_path} exists; pass --resume or choose a new --out"
        )
    existing = _load_existing(records_path) if args.resume else []
    completed_ids = {str(record["question_id"]) for record in existing}

    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        adapter_scale=args.adapter_scale,
        offline=not args.allow_download,
    )
    steps = args.steps or int(runtime.config.num_recurrent_steps)
    root = resolve_vstar_root(args.vstar_root)
    rows = load_vstar_rows(root)
    if args.limit:
        rows = rows[: args.limit]
    lengths = question_token_lengths(runtime.processor, rows)
    donors = exact_length_donors(rows, lengths)
    target_indices = list(range(args.shard_index, len(rows), args.num_shards))
    target_indices = [
        index
        for index in target_indices
        if str(rows[index]["question_id"]) not in completed_ids
    ]
    needed_donors = {
        donors[index]
        for index in target_indices
        if "matched_swap" in args.conditions and index in donors
    }
    bank = _residual_bank(runtime, root, rows, needed_donors, steps=steps)
    token_ids = option_token_ids(runtime.processor.tokenizer)

    started = time.monotonic()
    for position, index in enumerate(target_indices, start=1):
        row = rows[index]
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with runtime.torch.inference_mode():
            trace = runtime.extract(example)
            clean_state = runtime.rollout(trace, steps=steps)[-1]
            clean_residual = clean_state - trace.base_anchor
            residuals = {"clean": clean_residual}
            donor_index = donors.get(index)
            if (
                "matched_swap" in args.conditions
                and donor_index is not None
                and donor_index in bank
            ):
                donor_residual = bank[donor_index].to(
                    runtime.device, dtype=trace.base_anchor.dtype
                )
                if donor_residual.shape != clean_residual.shape:
                    raise RuntimeError(
                        "equal-token donor produced a different residual shape: "
                        f"{tuple(donor_residual.shape)} vs {tuple(clean_residual.shape)}"
                    )
                residuals["matched_swap"] = donor_residual
            if "norm_matched_noise" in args.conditions:
                residuals["norm_matched_noise"] = norm_matched_noise(
                    clean_residual, seed=args.seed * 1_000_003 + index
                )
            blank_trace = None
            if "blank" in args.conditions:
                blank_example = prepare_example(
                    runtime.processor,
                    item_id=f"{row['question_id']}:blank",
                    text=row["text"],
                    label=row["label"],
                    image=blank_like(example.image, args.blank_value),
                    device=runtime.device,
                )
                blank_trace = runtime.extract(blank_example)
                blank_state = runtime.rollout(blank_trace, steps=steps)[-1]
                blank_residual = blank_state - blank_trace.base_anchor
                if blank_residual.shape != clean_residual.shape:
                    raise RuntimeError("blank and clean question states are misaligned")
                residuals["blank"] = blank_residual

            results = {}
            for path in args.paths:
                results[path] = {}
                # This is deliberately candidate-independent: it represents a
                # complete, untouched multimodal bypass and therefore serves
                # as the zero-sensitivity sanity bound.
                if path == "full_mm_oracle":
                    scores = runtime.token_scores(
                        trace,
                        clean_residual,
                        token_ids,
                        path=path,
                        representation="residual",
                    )
                    oracle_result = classification_result(
                        scores, LETTERS, row["label"]
                    )
                    for condition in args.conditions:
                        if condition in residuals:
                            results[path][condition] = dict(oracle_result)
                    continue
                for condition in args.conditions:
                    if condition not in residuals:
                        continue
                    scores = runtime.token_scores(
                        trace,
                        residuals[condition],
                        token_ids,
                        path=path,
                        representation="residual",
                    )
                    results[path][condition] = classification_result(
                        scores, LETTERS, row["label"]
                    )

        record = {
            "global_index": index,
            "question_id": str(row["question_id"]),
            "category": row.get("category"),
            "label": row["label"],
            "question_tokens": trace.question_tokens,
            "visual_tokens": trace.visual_tokens,
            "donor_question_id": (
                str(rows[donor_index]["question_id"])
                if donor_index is not None and donor_index in bank
                else None
            ),
            "donor_label": (
                rows[donor_index]["label"]
                if donor_index is not None and donor_index in bank
                else None
            ),
            "matched_swap_eligible": donor_index is not None,
            "results": results,
        }
        _write_record(records_path, record)
        existing.append(record)
        if blank_trace is not None:
            release_trace(blank_trace)
        release_trace(trace)
        del trace, example, residuals
        if position % 10 == 0 or position == len(target_indices):
            elapsed = time.monotonic() - started
            rate = elapsed / position
            print(
                f"targets {position}/{len(target_indices)} "
                f"eta={(len(target_indices) - position) * rate / 60:.1f} min",
                flush=True,
            )

    summary = summarize_condition_records(
        existing,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    payload = {
        "kind": "cvrr_causal_reliance",
        "protocol": {
            "dataset": "V*",
            "n_total": len(rows),
            "steps": steps,
            "conditions": args.conditions,
            "paths": args.paths,
            "matched_swap": (
                "same-category/different-answer donor with exactly equal "
                "text-only token count"
            ),
            "blank": "same-size mid-gray image, residual transplanted onto target B",
            "noise": "independent Gaussian direction with per-row L2 norm preserved",
            "strict": "only recurrent question state reaches L(cell+1):end",
            "visual_rows_restored": (
                "target post-cell visual rows are restored to L(cell+1):end while "
                "the recurrent candidate remains intervened"
            ),
            "dual_path_bypass": (
                "the untouched native multimodal prefix is retained and the "
                "intervened recurrent state is appended as a causal query stream"
            ),
            "full_mm_oracle": (
                "untouched native multimodal continuation; candidate-independent "
                "complete-bypass sanity bound"
            ),
        },
        "runtime": runtime.metadata(),
        "n": len(existing),
        "summary": summary,
        "bypass_restoration": _bypass_sensitivity(summary),
        "path_sensitivity": _path_sensitivity(summary),
        "path_interactions": _path_interactions(
            existing, n_boot=args.bootstrap, seed=args.seed
        ),
        "records": existing,
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
