"""Functional step-wise analysis of the learned recurrent LoRA transition."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

from scripts.analysis_cvrr.core.data import (
    load_vstar_rows,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import scalar_summary
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _find_lora_modules(runtime: CVRRRuntime) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, module in runtime.model.named_modules():
        label = name.rsplit(".", 1)[-1]
        if label not in MODULES or not hasattr(module, "lora_A"):
            continue
        if label in found:
            raise RuntimeError(f"multiple recurrent LoRA modules found for {label}")
        found[label] = module
    missing = [label for label in MODULES if label not in found]
    if missing:
        raise RuntimeError(f"missing recurrent LoRA modules: {missing}")
    return found


def _weight_summary(modules: dict[str, Any]) -> dict[str, Any]:
    import torch

    output = {}
    for label, module in modules.items():
        adapters = [
            adapter
            for adapter in module.active_adapters
            if adapter in module.lora_A
        ]
        if len(adapters) != 1:
            raise RuntimeError(f"expected one active adapter for {label}: {adapters}")
        adapter = adapters[0]
        a = module.lora_A[adapter].weight.detach().float()
        b = module.lora_B[adapter].weight.detach().float()
        scale = float(module.scaling[adapter])
        gram_b = b.T @ b
        gram_a = a @ a.T
        delta_fro = float((gram_b * gram_a.T).sum().clamp_min(0).sqrt() * abs(scale))
        base_fro = float(module.get_base_layer().weight.detach().float().norm())
        qb, rb = torch.linalg.qr(b, mode="reduced")
        qa, ra = torch.linalg.qr(a.T, mode="reduced")
        del qb, qa
        singular = torch.linalg.svdvals(rb @ ra.T).mul(abs(scale)).cpu()
        energy = singular.square()
        total = float(energy.sum())
        output[label] = {
            "rank": int(a.shape[0]),
            "scaling": scale,
            "delta_frobenius": delta_fro,
            "base_frobenius": base_fro,
            "relative_weight_frobenius": delta_fro / max(base_fro, 1e-12),
            "top1_energy_fraction": float(energy[0]) / total if total else 0.0,
            "top4_energy_fraction": float(energy[:4].sum()) / total if total else 0.0,
            "effective_rank_99": int(
                torch.searchsorted(energy.cumsum(0), 0.99 * energy.sum()).item() + 1
            ) if total else 0,
            "singular_values": [float(value) for value in singular],
        }
    return output


def _activation_hook(module, label: str, trace, bucket: list[dict[str, float]]):
    import torch

    def hook(_module, args, result):
        x = args[0]
        base = module.get_base_layer()(x)
        delta = result.float() - base.float()
        if label in {"k_proj", "v_proj"}:
            if delta.shape[:2] != trace.visual_rows.shape:
                raise RuntimeError(f"{label} did not receive the full scaffold")
            selected_delta = delta[trace.visual_rows]
            selected_base = base.float()[trace.visual_rows]
        else:
            # The released recurrent kernel evaluates q/o/MLP projections only
            # on question output rows, so no full-scaffold mask is required.
            selected_delta = delta.reshape(-1, delta.shape[-1])
            selected_base = base.float().reshape(-1, base.shape[-1])
        delta_rms = float(selected_delta.square().mean().sqrt())
        base_rms = float(selected_base.square().mean().sqrt())
        flat_delta = selected_delta.reshape(-1)
        flat_base = selected_base.reshape(-1)
        cosine = float(
            torch.nn.functional.cosine_similarity(flat_delta, flat_base, dim=0)
        )
        bucket.append(
            {
                "delta_rms": delta_rms,
                "base_rms": base_rms,
                "relative_activation": delta_rms / max(base_rms, 1e-12),
                "delta_base_cosine": cosine,
                "selected_rows": int(selected_delta.shape[0]),
            }
        )

    return hook


def summarize(records: list[dict[str, Any]], *, bootstrap: int, seed: int):
    output = {}
    for step in (2, 3, 4):
        step_key = f"T{step}"
        output[step_key] = {}
        for label in MODULES:
            rows = [record["steps"][step_key][label] for record in records]
            output[step_key][label] = {
                metric: scalar_summary(
                    (row[metric] for row in rows),
                    n_boot=bootstrap,
                    seed=seed + step,
                )
                for metric in (
                    "relative_activation",
                    "delta_rms",
                    "base_rms",
                    "delta_base_cosine",
                )
            }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record.get("category", "unknown"))].append(record)
    category_output = {}
    for category, category_records in sorted(by_category.items()):
        category_output[category] = {}
        for step in (2, 3, 4):
            step_key = f"T{step}"
            category_output[category][step_key] = {
                label: scalar_summary(
                    (
                        record["steps"][step_key][label]["relative_activation"]
                        for record in category_records
                    ),
                    n_boot=bootstrap,
                    seed=seed + step,
                )
                for label in MODULES
            }
    return {"overall": output, "per_category": category_output}



def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps != 4:
        raise SystemExit("this analysis expects canonical T=4")
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
        offline=not args.allow_download,
    )
    modules = _find_lora_modules(runtime)
    weights = _weight_summary(modules)
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
    started = time.monotonic()
    for position, (index, row) in enumerate(indexed, start=1):
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with runtime.torch.inference_mode():
            trace = runtime.extract(example)
            captures: dict[str, list[dict[str, float]]] = {
                label: [] for label in MODULES
            }
            handles = [
                module.register_forward_hook(
                    _activation_hook(module, label, trace, captures[label])
                )
                for label, module in modules.items()
            ]
            try:
                runtime.rollout(trace, steps=4)
            finally:
                for handle in handles:
                    handle.remove()
            if any(len(captures[label]) != 3 for label in MODULES):
                counts = {label: len(values) for label, values in captures.items()}
                raise RuntimeError(f"unexpected recurrent projection calls: {counts}")
        record = {
            "global_index": index,
            "question_id": str(row["question_id"]),
            "category": row.get("category"),
            "steps": {
                f"T{step}": {
                    label: captures[label][step - 2] for label in MODULES
                }
                for step in (2, 3, 4)
            },
        }
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        records.append(record)
        release_trace(trace)
        if position % 10 == 0 or position == len(indexed):
            elapsed = time.monotonic() - started
            eta = (len(indexed) - position) * elapsed / position / 60
            print(f"transition {position}/{len(indexed)} eta={eta:.1f} min", flush=True)

    payload = {
        "kind": "cvrr_transition_activation",
        "protocol": {
            "dataset": "V*",
            "steps": 4,
            "metric": "RMS(LoRA output) / RMS(frozen projection output)",
            "supports": "k/v on visual rows; q/o/MLP projections on question rows",
        },
        "runtime": runtime.metadata(),
        "weights": weights,
        "n": len(records),
        "summary": summarize(records, bootstrap=args.bootstrap, seed=args.seed),
        "records": records,
        "meta": run_metadata(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(out)
    print(f"wrote {out}", flush=True)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
