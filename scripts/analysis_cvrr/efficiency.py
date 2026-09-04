"""Axis 6: first-token accuracy, latency, memory, FLOPs, and parameter counts by T."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LETTERS,
    load_vstar_rows,
    move_prepared,
    option_token_ids,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import classification_result, scalar_summary
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _parameter_counts(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    recurrence_lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
        and f"layers.{int(model.config.ell_star) + 1}." in name
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "recurrent_layer_lora": recurrence_lora,
    }


def _decoder_layer_macs(text_config, sequence_length: int) -> int:
    """Dense causal decoder MAC estimate for attention and SwiGLU linears."""

    length = int(sequence_length)
    hidden = int(text_config.hidden_size)
    heads = int(text_config.num_attention_heads)
    kv_heads = int(text_config.num_key_value_heads)
    head_dim = int(getattr(text_config, "head_dim", hidden // heads))
    intermediate = int(text_config.intermediate_size)
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    projection_macs = length * hidden * (
        query_width + 2 * kv_width + query_width
    )
    visible_pairs = length * (length + 1) // 2
    attention_macs = 2 * visible_pairs * query_width
    swiglu_macs = 3 * length * hidden * intermediate
    return int(projection_macs + attention_macs + swiglu_macs)


def _decoder_path_compute(runtime, visual_counts, question_counts, steps):
    config = runtime.model.text_model.config
    n_layers = len(runtime.model.text_model.layers)
    # Multimodal calls: lower layers through ell*, one native R1 read, then
    # T-1 repeats. Text-only calls collectively traverse the full stack once.
    multimodal_layer_calls = runtime.cell_index + int(steps)
    text_layer_calls = (
        runtime.cell_index + 1 + (n_layers - runtime.upper_start)
    )
    per_example = []
    for visual, question in zip(visual_counts, question_counts):
        multimodal_length = int(visual) + int(question)
        macs = multimodal_layer_calls * _decoder_layer_macs(
            config, multimodal_length
        )
        macs += text_layer_calls * _decoder_layer_macs(config, int(question))
        per_example.append(macs)
    mean_macs = statistics.mean(per_example)
    return {
        "macs_mean": mean_macs,
        "flops_mean": 2 * mean_macs,
        "multimodal_layer_calls": multimodal_layer_calls,
        "text_only_layer_calls": text_layer_calls,
        "scope": "decoder attention/MLP matrix operations",
        "note": (
            "analytic dense-causal estimate; excludes vision encoder, norms, "
            "rotary embeddings, activations, and kernel implementation effects"
        ),
    }


def _one(runtime, example, row, token_ids, steps):
    with runtime.torch.inference_mode():
        trace = runtime.extract(example)
        final_state = runtime.rollout(trace, steps=steps)[-1]
        scores = runtime.token_scores(trace, final_state, token_ids, path="strict")
        result = classification_result(scores, LETTERS, row["label"])
        visual_tokens = trace.visual_tokens
        question_tokens = trace.question_tokens
    release_trace(trace)
    return result, visual_tokens, question_tokens


def _profile_flops(runtime, example, row, token_ids, steps) -> dict[str, Any]:
    torch = runtime.torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if runtime.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True) as prof:
        _one(runtime, example, row, token_ids, steps)
    events = prof.key_averages()
    counted = [int(event.flops) for event in events if getattr(event, "flops", 0)]
    return {
        "partial_flops": sum(counted),
        "operators_with_flops": len(counted),
        "note": (
            "torch.profiler counts supported matmul/conv operators only; fused "
            "attention and custom kernels can be absent"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.steps = sorted(set(args.steps))
    if not args.steps or args.steps[0] < 1:
        raise SystemExit("--steps must contain positive integers")
    if args.samples < 1 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("invalid sample/warmup/repeat count")
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        adapter_scale=args.adapter_scale,
        offline=not args.allow_download,
    )
    torch = runtime.torch
    root = resolve_vstar_root(args.vstar_root)
    rows = load_vstar_rows(root)[: args.samples]
    # Preprocess once on CPU.  Move one sample at a time outside the timed
    # region so the metric isolates model execution without retaining an
    # entire long-image benchmark shard in accelerator memory.
    prepared = [vstar_example(runtime.processor, root, row, "cpu") for row in rows]
    token_ids = option_token_ids(runtime.processor.tokenizer)

    for index in range(args.warmup):
        example = move_prepared(prepared[index % len(prepared)], runtime.device)
        _one(
            runtime,
            example,
            rows[index % len(rows)],
            token_ids,
            max(args.steps),
        )
        del example
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)

    base_allocated = (
        int(torch.cuda.memory_allocated(runtime.device))
        if runtime.device.type == "cuda"
        else 0
    )
    results = {}
    for steps in args.steps:
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(runtime.device)
        latencies = []
        outcomes = []
        visual_counts = []
        question_counts = []
        for _ in range(args.repeats):
            for row, example in zip(rows, prepared):
                device_example = move_prepared(example, runtime.device)
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                started = time.perf_counter()
                outcome, visual_tokens, question_tokens = _one(
                    runtime, device_example, row, token_ids, steps
                )
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                latencies.append(time.perf_counter() - started)
                outcomes.append(outcome)
                visual_counts.append(visual_tokens)
                question_counts.append(question_tokens)
                del device_example
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.device.type == "cuda"
            else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(runtime.device))
            if runtime.device.type == "cuda"
            else None
        )
        step_result = {
            "steps": steps,
            "n_examples": len(outcomes),
            "accuracy": sum(outcome["correct"] for outcome in outcomes) / len(outcomes),
            "gold_margin": scalar_summary(
                (outcome["gold_margin"] for outcome in outcomes),
                n_boot=args.bootstrap,
                seed=args.seed,
            ),
            "latency_seconds": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "minimum": min(latencies),
                "maximum": max(latencies),
                "all": latencies,
            },
            "peak_memory_bytes": {
                "allocated": peak_allocated,
                "reserved": peak_reserved,
                "increment_over_loaded_model": (
                    max(0, peak_allocated - base_allocated)
                    if peak_allocated is not None
                    else None
                ),
            },
            "visual_tokens": {
                "mean": statistics.mean(visual_counts),
                "min": min(visual_counts),
                "max": max(visual_counts),
            },
            "question_tokens": {
                "mean": statistics.mean(question_counts),
                "min": min(question_counts),
                "max": max(question_counts),
            },
            "estimated_decoder_compute": _decoder_path_compute(
                runtime, visual_counts, question_counts, steps
            ),
        }
        if args.profile_flops:
            profile_example = move_prepared(prepared[0], runtime.device)
            step_result["profiler"] = _profile_flops(
                runtime, profile_example, rows[0], token_ids, steps
            )
            del profile_example
        results[f"T{steps}"] = step_result
        print(
            f"T={steps} latency={step_result['latency_seconds']['mean']:.3f}s "
            f"acc={step_result['accuracy']:.3f}",
            flush=True,
        )

    baseline_latency = results[f"T{args.steps[0]}"]["latency_seconds"]["mean"]
    for result in results.values():
        result["relative_latency"] = (
            result["latency_seconds"]["mean"] / baseline_latency
        )
    payload = {
        "kind": "cvrr_efficiency",
        "protocol": {
            "scope": "image preprocessing excluded; full model first-token path included",
            "steps": args.steps,
            "samples": args.samples,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": str(runtime.model.dtype),
            "device": str(runtime.device),
        },
        "runtime": runtime.metadata(),
        "parameters": _parameter_counts(runtime.model),
        "loaded_model_memory_bytes": base_allocated,
        "results": results,
        "meta": run_metadata(args.seed),
    }
    out = pathlib.Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
