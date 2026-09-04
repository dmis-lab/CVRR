"""Same-question image-conditioned residual swap under strict CVRR decoding.

The intervention keeps the target text-only anchor ``B(q)`` and transplants
only the operational residual ``h_T(I', q) - B(q)`` from an exact
same-question, different-image partner.  This subtraction is an operational
control; it is not assumed to perfectly disentangle visual and textual
information.

The unrelated whole-state swap is retained as the deliberately confounded
reference used to motivate this control.  It uses a different-question,
different-answer donor with exactly the same serialized question-state
length.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

from scripts.analysis_cvrr.core.data import (
    load_vstar_rows,
    prepare_example,
    resolve_vstar_root,
    text_only_prompt,
)
from scripts.analysis_cvrr.core.metrics import (
    exact_mcnemar_p,
    holm_adjust,
    scalar_summary,
    summarize_condition_records,
)
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


CONDITIONS = (
    "clean",
    "whole_state_swap",
    "same_question_residual_swap",
    "text_only_anchor",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", choices=("vstar", "mmvp"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--mmvp-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=0, help="0 uses checkpoint T")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
    )
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _latest_snapshot(repo_dir: str) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    model_id = repo_dir.removeprefix("datasets--").replace("--", "/")
    return pathlib.Path(
        snapshot_download(model_id, repo_type="dataset", local_files_only=True)
    )


def _normalize_letter(value: object) -> str:
    import re

    match = re.search(r"\(?([A-Ea-e])\)?", str(value))
    return match.group(1).upper() if match else ""


def load_rows(
    benchmark: str,
    *,
    vstar_root: str | None = None,
    mmvp_root: str | None = None,
) -> list[dict[str, Any]]:
    """Load metadata only; images are opened lazily during evaluation."""

    if benchmark == "vstar":
        root = resolve_vstar_root(vstar_root)
        rows = []
        for index, row in enumerate(load_vstar_rows(root)):
            rows.append(
                {
                    "global_index": index,
                    "question_id": str(row["question_id"]),
                    "text": str(row["text"]),
                    "pair_key": str(row["text"]),
                    "label": str(row["label"]),
                    "category": str(row.get("category", "unknown")),
                    "choice_letters": ["A", "B", "C", "D"],
                    "image_path": str(root / row["image"]),
                }
            )
        return rows

    root = (
        pathlib.Path(mmvp_root).expanduser().resolve()
        if mmvp_root
        else _latest_snapshot("datasets--MMVP--MMVP")
    )
    rows = []
    with (root / "Questions.csv").open() as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            question = row["Question"].strip()
            options = row["Options"].strip()
            text = (
                f"{question}\n{options}\n"
                "Answer with only the letter of the correct option."
            )
            rows.append(
                {
                    "global_index": index,
                    "question_id": str(row["Index"]),
                    "text": text,
                    "pair_key": f"{question}\n{options}",
                    "label": _normalize_letter(row["Correct Answer"]),
                    "category": "MMVP",
                    "choice_letters": ["A", "B"],
                    "image_path": str(root / "MMVP Images" / f"{row['Index']}.jpg"),
                }
            )
    return rows


def exact_same_question_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_key"])].append(index)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                if rows[left]["label"] != rows[right]["label"]:
                    pairs.append((left, right))
    return sorted(pairs)


def _question_lengths(processor, rows: list[dict[str, Any]]) -> list[int]:
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


def unrelated_whole_state_donors(
    rows: list[dict[str, Any]], lengths: list[int]
) -> dict[int, int]:
    """Different-question/different-answer donors with state-shape alignment."""

    donors: dict[int, int] = {}
    for index, row in enumerate(rows):
        candidates = [
            other
            for other, other_row in enumerate(rows)
            if other != index
            and lengths[other] == lengths[index]
            and other_row["label"] != row["label"]
            and other_row["pair_key"] != row["pair_key"]
            and other_row["category"] == row["category"]
            and other_row["choice_letters"] == row["choice_letters"]
        ]
        if candidates:
            donors[index] = min(
                candidates,
                key=lambda other: (
                    abs(other - index),
                    str(rows[other]["question_id"]),
                ),
            )
    return donors


def _prepared(runtime: CVRRRuntime, row: dict[str, Any]):
    from PIL import Image

    with Image.open(row["image_path"]) as opened:
        image = opened.convert("RGB")
    return prepare_example(
        runtime.processor,
        item_id=row["question_id"],
        text=row["text"],
        label=row["label"],
        image=image,
        device=runtime.device,
    )


def _candidate_token_ids(runtime: CVRRRuntime, trace, letters: list[str]) -> list[int]:
    # Match the established CVRR causal suite exactly.  This is deliberately
    # the restricted ``" A"``-style candidate-token readout rather than the
    # first token under generation serialization.
    token_ids = []
    for letter in letters:
        pieces = runtime.processor.tokenizer.encode(
            " " + letter, add_special_tokens=False
        )
        if not pieces:
            raise RuntimeError(f"no restricted candidate token for {letter!r}")
        token_ids.append(int(pieces[0]))
    return token_ids


def _classification(scores: list[float], labels: list[str], gold: str) -> dict[str, Any]:
    from scripts.analysis_cvrr.core.metrics import classification_result

    return classification_result(scores, labels, gold)


def _evaluate_candidate(runtime, trace, candidate, row, *, representation: str):
    token_ids = _candidate_token_ids(runtime, trace, row["choice_letters"])
    scores = runtime.token_scores(
        trace,
        candidate,
        token_ids,
        path="strict",
        representation=representation,
    )
    return _classification(scores, row["choice_letters"], row["label"])


def _record_path(out: pathlib.Path) -> pathlib.Path:
    return out.with_suffix(out.suffix + ".records.jsonl")


def _write_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pair_cluster_summary(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    """Bootstrap complete image pairs so the two directions are not independent."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pair_id"])].append(record)
    output: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        complete = [
            pair_records
            for pair_records in grouped.values()
            if all(condition in row["results"]["strict"] for row in pair_records)
        ]
        pair_accuracy = [
            sum(float(row["results"]["strict"][condition]["correct"]) for row in pair)
            / len(pair)
            for pair in complete
        ]
        item_rows = [row for pair in complete for row in pair]
        summary: dict[str, Any] = {
            "n_pairs": len(complete),
            "n_directed_items": len(item_rows),
            "accuracy": scalar_summary(
                pair_accuracy,
                n_boot=bootstrap,
                seed=seed + 20 * condition_index,
            ),
        }
        if condition != "clean":
            pair_delta = [
                sum(
                    float(row["results"]["strict"][condition]["correct"])
                    - float(row["results"]["strict"]["clean"]["correct"])
                    for row in pair
                )
                / len(pair)
                for pair in complete
            ]
            pair_margin_delta = [
                sum(
                    float(row["results"]["strict"][condition]["gold_margin"])
                    - float(row["results"]["strict"]["clean"]["gold_margin"])
                    for row in pair
                )
                / len(pair)
                for pair in complete
            ]
            clean_correct = [
                bool(row["results"]["strict"]["clean"]["correct"])
                for row in item_rows
            ]
            changed_correct = [
                bool(row["results"]["strict"][condition]["correct"])
                for row in item_rows
            ]
            p_value = exact_mcnemar_p(clean_correct, changed_correct)
            raw_p[condition] = p_value
            summary["vs_clean"] = {
                "accuracy_delta": scalar_summary(
                    pair_delta,
                    n_boot=bootstrap,
                    seed=seed + 20 * condition_index + 1,
                ),
                "gold_margin_delta": scalar_summary(
                    pair_margin_delta,
                    n_boot=bootstrap,
                    seed=seed + 20 * condition_index + 2,
                ),
                "prediction_flip_fraction": sum(
                    row["results"]["strict"][condition]["pred"]
                    != row["results"]["strict"]["clean"]["pred"]
                    for row in item_rows
                )
                / len(item_rows),
                "mcnemar_exact_p_item_level": p_value,
            }
        output[condition] = summary
    for condition, adjusted in holm_adjust(raw_p).items():
        output[condition]["vs_clean"]["mcnemar_holm_p_item_level"] = adjusted
    return output


def summarize_records(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    if not records:
        return {"item_level": {}, "pair_clustered": {}, "per_category": {}}
    item = summarize_condition_records(records, n_boot=bootstrap, seed=seed)
    categories = sorted({str(record.get("category", "unknown")) for record in records})
    per_category = {
        category: summarize_condition_records(
            [record for record in records if str(record.get("category")) == category],
            n_boot=bootstrap,
            seed=seed,
        )
        for category in categories
    }
    return {
        "item_level": item,
        "pair_clustered": _pair_cluster_summary(
            records, bootstrap=bootstrap, seed=seed
        ),
        "per_category": per_category,
    }


def _write_csv(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "pair_id",
        "target_question_id",
        "partner_question_id",
        "target_label",
        "partner_label",
        "condition",
        "prediction",
        "correct",
        "gold_margin",
        "scores",
        "whole_state_donor_question_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for condition, result in record["results"]["strict"].items():
                writer.writerow(
                    {
                        "benchmark": record["benchmark"],
                        "pair_id": record["pair_id"],
                        "target_question_id": record["question_id"],
                        "partner_question_id": record["partner_question_id"],
                        "target_label": record["label"],
                        "partner_label": record["partner_label"],
                        "condition": condition,
                        "prediction": result["pred"],
                        "correct": int(bool(result["correct"])),
                        "gold_margin": result["gold_margin"],
                        "scores": json.dumps(result["scores"]),
                        "whole_state_donor_question_id": record.get(
                            "whole_state_donor_question_id"
                        ),
                    }
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    records_path = _record_path(out)
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"{records_path} exists; pass --resume")
    records = _load_records(records_path) if args.resume else []
    completed_pairs = {str(record["pair_id"]) for record in records}

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
    rows = load_rows(
        args.benchmark,
        vstar_root=args.vstar_root,
        mmvp_root=args.mmvp_root,
    )
    all_pairs = exact_same_question_pairs(rows)
    if args.limit_pairs:
        all_pairs = all_pairs[: args.limit_pairs]
    indexed_pairs = [
        (pair_index, pair)
        for pair_index, pair in enumerate(all_pairs)
        if pair_index % args.num_shards == args.shard_index
        and f"{args.benchmark}:{pair_index}" not in completed_pairs
    ]
    lengths = _question_lengths(runtime.processor, rows)
    whole_donors = unrelated_whole_state_donors(rows, lengths)
    started = time.monotonic()

    for position, (pair_index, (left_index, right_index)) in enumerate(
        indexed_pairs, start=1
    ):
        pair_id = f"{args.benchmark}:{pair_index}"
        pair_indices = (left_index, right_index)
        pair_rows = [rows[index] for index in pair_indices]
        examples = [_prepared(runtime, row) for row in pair_rows]
        traces = []
        states = []
        with runtime.torch.inference_mode():
            for example in examples:
                trace = runtime.extract(example)
                traces.append(trace)
                states.append(runtime.rollout(trace, steps=steps)[-1])

            left_trace, right_trace = traces
            anchor_shape_equal = left_trace.base_anchor.shape == right_trace.base_anchor.shape
            ids_equal = runtime.torch.equal(
                left_trace.example.question_ids,
                right_trace.example.question_ids,
            )
            masks_equal = runtime.torch.equal(
                left_trace.example.question_attention_mask,
                right_trace.example.question_attention_mask,
            )
            anchor_equal = runtime.torch.equal(
                left_trace.base_anchor, right_trace.base_anchor
            )
            anchor_max_abs = float(
                (left_trace.base_anchor.float() - right_trace.base_anchor.float())
                .abs()
                .max()
                .item()
            )
            if not all((anchor_shape_equal, ids_equal, masks_equal, anchor_equal)):
                raise RuntimeError(
                    f"same-question anchor mismatch for {pair_id}: "
                    f"shape={anchor_shape_equal}, ids={ids_equal}, masks={masks_equal}, "
                    f"anchor={anchor_equal}, max_abs={anchor_max_abs}"
                )

            pair_records = []
            for direction, (target_position, partner_position) in enumerate(
                ((0, 1), (1, 0))
            ):
                target_index = pair_indices[target_position]
                partner_index = pair_indices[partner_position]
                target = rows[target_index]
                partner = rows[partner_index]
                target_trace = traces[target_position]
                target_state = states[target_position]
                partner_trace = traces[partner_position]
                partner_state = states[partner_position]

                whole_donor_index = whole_donors.get(target_index)
                donor_row = rows[whole_donor_index] if whole_donor_index is not None else None
                donor_trace = donor_state = donor_example = None
                if donor_row is not None:
                    donor_example = _prepared(runtime, donor_row)
                    donor_trace = runtime.extract(donor_example)
                    donor_state = runtime.rollout(donor_trace, steps=steps)[-1]
                    if donor_state.shape != target_state.shape:
                        raise RuntimeError(
                            f"whole-state donor shape mismatch for {target['question_id']}"
                        )

                clean_residual = target_state - target_trace.base_anchor
                partner_residual = partner_state - partner_trace.base_anchor
                if partner_residual.shape != clean_residual.shape:
                    raise RuntimeError("same-question residual shape mismatch")
                reconstructed_partner = target_trace.base_anchor + partner_residual
                reconstruction_max_abs = float(
                    (reconstructed_partner.float() - partner_state.float())
                    .abs()
                    .max()
                    .item()
                )

                results = {
                    "clean": _evaluate_candidate(
                        runtime,
                        target_trace,
                        clean_residual,
                        target,
                        representation="residual",
                    ),
                    "same_question_residual_swap": _evaluate_candidate(
                        runtime,
                        target_trace,
                        partner_residual,
                        target,
                        representation="residual",
                    ),
                    "text_only_anchor": _evaluate_candidate(
                        runtime,
                        target_trace,
                        target_trace.base_anchor,
                        target,
                        representation="decoder",
                    ),
                }
                if donor_state is not None:
                    results["whole_state_swap"] = _evaluate_candidate(
                        runtime,
                        target_trace,
                        donor_state,
                        target,
                        representation="decoder",
                    )
                record = {
                    "benchmark": args.benchmark,
                    "pair_id": pair_id,
                    "pair_index": pair_index,
                    "direction": direction,
                    "global_index": target_index,
                    "question_id": str(target["question_id"]),
                    "partner_global_index": partner_index,
                    "partner_question_id": str(partner["question_id"]),
                    "category": target["category"],
                    "label": target["label"],
                    "partner_label": partner["label"],
                    "whole_state_donor_global_index": whole_donor_index,
                    "whole_state_donor_question_id": (
                        str(donor_row["question_id"]) if donor_row is not None else None
                    ),
                    "whole_state_donor_label": (
                        donor_row["label"] if donor_row is not None else None
                    ),
                    "whole_state_swap_eligible": donor_row is not None,
                    "question_tokens": target_trace.question_tokens,
                    "visual_tokens": target_trace.visual_tokens,
                    "sanity": {
                        "exact_same_question": target["pair_key"] == partner["pair_key"],
                        "different_gold_answers": target["label"] != partner["label"],
                        "question_ids_equal": ids_equal,
                        "question_masks_equal": masks_equal,
                        "base_anchor_bit_identical": anchor_equal,
                        "base_anchor_max_abs_difference": anchor_max_abs,
                        "state_shapes_aligned": target_state.shape == partner_state.shape,
                        "residual_reconstruction_max_abs_difference": reconstruction_max_abs,
                        "strict_decoder_input_rows": "question_only",
                        "visual_rows_exposed_to_answer_decoder": False,
                        "multimodal_kv_cache_exposed_to_answer_decoder": False,
                    },
                    "results": {"strict": results},
                }
                pair_records.append(record)
                if donor_trace is not None:
                    release_trace(donor_trace)
                del donor_trace, donor_state, donor_example

        for record in pair_records:
            _write_record(records_path, record)
            records.append(record)
        for trace in traces:
            release_trace(trace)
        del traces, states, examples

        if position % 5 == 0 or position == len(indexed_pairs):
            elapsed = time.monotonic() - started
            eta = (len(indexed_pairs) - position) * elapsed / position / 60
            print(
                f"residual-swap {args.benchmark} pairs "
                f"{position}/{len(indexed_pairs)} eta={eta:.1f} min",
                flush=True,
            )

    unique_pairs = {record["pair_id"] for record in records}
    payload = {
        "kind": "cvrr_same_question_residual_swap_shard",
        "protocol": {
            "benchmark": args.benchmark,
            "readout": "restricted first-option-token",
            "steps": steps,
            "same_question_pairing": "exact serialized question and options only",
            "whole_state_swap": (
                "entire h_T from a different-question/different-answer donor "
                "with identical serialized question-state length"
            ),
            "same_question_residual_swap": (
                "target B(q) + [partner h_T(I',q) - partner B(q)]"
            ),
            "residual_caveat": (
                "operational image-conditioned residual; subtraction is not "
                "claimed to perfectly disentangle visual and textual information"
            ),
            "strict_decoder": (
                "question-shaped candidate only; no visual rows or original "
                "multimodal KV cache; upper decoding uses use_cache=False"
            ),
            "bootstrap_unit": "exact same-question image pair",
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "runtime": runtime.metadata(),
        "n_available_exact_pairs": len(all_pairs),
        "n_pairs": len(unique_pairs),
        "n_directed_items": len(records),
        "n_whole_state_swap_eligible_items": sum(
            bool(record.get("whole_state_swap_eligible")) for record in records
        ),
        "n_pairs_with_different_gold_answers": len(unique_pairs),
        "summary": summarize_records(
            records, bootstrap=args.bootstrap, seed=args.seed
        ),
        "records": records,
        "meta": run_metadata(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(out)
    _write_csv(out.with_suffix(".csv"), records)
    print(f"wrote {out}", flush=True)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
