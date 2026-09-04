"""Token-by-channel functional correction of the recurrent query LoRA."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from scripts.analysis_cvrr.core.data import load_vstar_rows, resolve_vstar_root, vstar_example
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


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
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--channel-group-size", type=int, default=7)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _find_q_projection(runtime: CVRRRuntime):
    found = [
        module
        for name, module in runtime.model.named_modules()
        if name.rsplit(".", 1)[-1] == "q_proj" and hasattr(module, "lora_A")
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one recurrent q_proj LoRA, found {len(found)}")
    return found[0]


def _capture_hook(module, trace, group_size: int, bucket: list[dict[str, Any]]):
    def hook(_module, args, result):
        x = args[0]
        base = module.get_base_layer()(x).float()
        delta = result.float() - base
        if delta.shape[0] != 1:
            raise RuntimeError("query-channel analysis requires batch size one")
        base = base[0]
        delta = delta[0]
        width = int(delta.shape[-1])
        if width % group_size:
            raise RuntimeError(
                f"q projection width {width} is not divisible by group size {group_size}"
            )
        groups = width // group_size
        bucket.append(
            {
                "delta_group_mse": delta.reshape(-1, groups, group_size)
                .square()
                .mean(dim=-1)
                .cpu(),
                "base_mse": base.square().mean(dim=-1, keepdim=True).cpu(),
                "width": width,
                "groups": groups,
            }
        )

    return hook


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps != 4:
        raise SystemExit("channel activation analysis expects canonical T=4")
    if args.channel_group_size < 1:
        raise SystemExit("--channel-group-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    from scripts.analysis_cvrr.core.common import run_metadata, set_seed

    set_seed(args.seed)
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    runtime = CVRRRuntime(
        checkpoint=args.checkpoint,
        processor_name=args.processor,
        device=args.device,
        max_visual_tokens=args.max_visual_tokens,
        beta=args.beta,
        offline=not args.allow_download,
    )
    q_projection = _find_q_projection(runtime)
    q_width = int(q_projection.get_base_layer().out_features)
    if q_width % args.channel_group_size:
        raise RuntimeError(
            f"q width {q_width} is not divisible by {args.channel_group_size}"
        )
    num_groups = q_width // args.channel_group_size
    torch = runtime.torch
    delta_sum = torch.zeros(
        3, args.max_question_tokens, num_groups, dtype=torch.float64
    )
    base_sum = torch.zeros(3, args.max_question_tokens, 1, dtype=torch.float64)
    counts = torch.zeros(3, args.max_question_tokens, 1, dtype=torch.int64)
    lengths: list[int] = []
    root = resolve_vstar_root(args.vstar_root)
    rows = load_vstar_rows(root)
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    started = time.monotonic()
    for position, (_, row) in enumerate(indexed, start=1):
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with torch.inference_mode():
            trace = runtime.extract(example)
            captured: list[dict[str, Any]] = []
            handle = q_projection.register_forward_hook(
                _capture_hook(q_projection, trace, args.channel_group_size, captured)
            )
            try:
                runtime.rollout(trace, steps=4)
            finally:
                handle.remove()
        if len(captured) != 3:
            raise RuntimeError(f"expected three q-projection calls, got {len(captured)}")
        length = int(captured[0]["delta_group_mse"].shape[0])
        if length > args.max_question_tokens:
            raise RuntimeError(
                f"question has {length} tokens, above --max-question-tokens "
                f"{args.max_question_tokens}"
            )
        start = args.max_question_tokens - length
        for step, values in enumerate(captured):
            delta_sum[step, start:] += values["delta_group_mse"].double()
            base_sum[step, start:] += values["base_mse"].double()
            counts[step, start:] += 1
        lengths.append(length)
        release_trace(trace)
        if position % 10 == 0 or position == len(indexed):
            elapsed = time.monotonic() - started
            eta = (len(indexed) - position) * elapsed / position / 60
            print(f"q-channel {position}/{len(indexed)} eta={eta:.1f} min", flush=True)

    payload = {
        "kind": "cvrr_q_channel_activation",
        "protocol": {
            "dataset": "V*",
            "steps": 4,
            "projection": "recurrent q_proj LoRA",
            "alignment": "text rows right-aligned to answer boundary",
            "metric": (
                "RMS LoRA correction in each contiguous channel group divided "
                "by RMS frozen q projection across all channels"
            ),
        },
        "runtime": runtime.metadata(),
        "n": len(indexed),
        "q_width": q_width,
        "channel_group_size": args.channel_group_size,
        "num_channel_groups": num_groups,
        "max_question_tokens": args.max_question_tokens,
        "question_token_lengths": lengths,
        "delta_group_mse_sum": delta_sum.tolist(),
        "base_mse_sum": base_sum.tolist(),
        "counts": counts.tolist(),
        "meta": run_metadata(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(out)
    print(f"wrote {out}", flush=True)
    return payload


def merge(paths: list[pathlib.Path]) -> dict[str, Any]:
    import numpy as np

    payloads = [json.loads(path.read_text()) for path in paths]
    reference = payloads[0]
    keys = ("q_width", "channel_group_size", "num_channel_groups", "max_question_tokens")
    for payload in payloads[1:]:
        for key in keys:
            if payload[key] != reference[key]:
                raise ValueError(f"incompatible {key}")
        if payload["protocol"] != reference["protocol"]:
            raise ValueError("incompatible channel-map protocol")
    delta = sum(np.asarray(payload["delta_group_mse_sum"]) for payload in payloads)
    base = sum(np.asarray(payload["base_mse_sum"]) for payload in payloads)
    counts = sum(np.asarray(payload["counts"]) for payload in payloads)
    output = dict(reference)
    output.update(
        {
            "n": sum(int(payload["n"]) for payload in payloads),
            "question_token_lengths": [
                length
                for payload in payloads
                for length in payload["question_token_lengths"]
            ],
            "delta_group_mse_sum": delta.tolist(),
            "base_mse_sum": base.tolist(),
            "counts": counts.tolist(),
            "source_shards": [str(path) for path in paths],
        }
    )
    return output


def _matrix(payload: dict[str, Any]):
    import numpy as np

    delta = np.asarray(payload["delta_group_mse_sum"], dtype=np.float64)
    base = np.asarray(payload["base_mse_sum"], dtype=np.float64)
    counts = np.asarray(payload["counts"], dtype=np.int64)[..., 0]
    ratio = np.sqrt(np.divide(delta, base, out=np.zeros_like(delta), where=base > 0))
    valid = counts.max(axis=0) > 0
    start = int(np.flatnonzero(valid)[0])
    return ratio[:, start:, :], counts[:, start:]



def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
