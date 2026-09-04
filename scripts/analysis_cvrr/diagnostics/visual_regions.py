"""Step-wise causal visual-region maps for persistent visual recurrence."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from scripts.analysis_cvrr.core.data import (
    LETTERS,
    load_vstar_rows,
    option_token_ids,
    resolve_vstar_root,
    vstar_example,
)
from scripts.analysis_cvrr.core.metrics import classification_result, scalar_summary
from scripts.analysis_cvrr.core.runtime import CVRRRuntime, release_trace


DEFAULT_INDICES = (0, 1, 2, 3, 115, 116, 117, 118)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--macro-grid", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _gold_margin(result: dict[str, Any]) -> float:
    return float(result["gold_margin"])


def _region_layout(trace, runtime: CVRRRuntime, macro_grid: int):
    torch = runtime.torch
    encoded = trace.example.multimodal
    grid = encoded["image_grid_thw"][0]
    merge = int(runtime.config.vision_config.spatial_merge_size)
    temporal = int(grid[0].item())
    height = int(grid[1].item()) // merge
    width = int(grid[2].item()) // merge
    image_mask = encoded["input_ids"].eq(int(runtime.config.image_token_id))
    image_positions = torch.nonzero(image_mask[0], as_tuple=False).flatten()
    expected = temporal * height * width
    if int(image_positions.numel()) != expected:
        raise RuntimeError(
            f"image grid mismatch: positions={image_positions.numel()} "
            f"grid={temporal}x{height}x{width}"
        )
    if temporal != 1:
        raise RuntimeError("V* causal maps currently expect one still image")
    rows = min(macro_grid, height)
    columns = min(macro_grid, width)
    regions = []
    token_region = torch.empty(expected, dtype=torch.long, device=image_positions.device)
    for row in range(rows):
        y0 = row * height // rows
        y1 = (row + 1) * height // rows
        for column in range(columns):
            x0 = column * width // columns
            x1 = (column + 1) * width // columns
            flat = [y * width + x for y in range(y0, y1) for x in range(x0, x1)]
            region_id = row * columns + column
            token_region[flat] = region_id
            key_mask = torch.zeros_like(trace.visual_rows)
            key_mask[0, image_positions[flat]] = True
            regions.append(
                {
                    "id": region_id,
                    "row": row,
                    "column": column,
                    "y": [y0, y1],
                    "x": [x0, x1],
                    "key_mask": key_mask,
                }
            )
    return {
        "height": height,
        "width": width,
        "rows": rows,
        "columns": columns,
        "regions": regions,
        "token_region": token_region,
    }


def _rollout_with_region_block(
    runtime: CVRRRuntime,
    trace,
    *,
    blocked_transition: int,
    key_mask,
    steps: int,
):
    scaffold = trace.first_full
    state = trace.r1
    for transition in range(2, steps + 1):
        recurrent_input = runtime._replace_rows(scaffold, trace.text_rows, state)
        with runtime.adapters(True):
            proposal, padding = runtime.model._recurrent_question_transition(
                recurrent_input,
                trace.mm_context,
                trace.text_rows,
                blocked_key_rows=(
                    key_mask if transition == blocked_transition else None
                ),
            )
        if bool(padding.any()):
            raise RuntimeError("batch-one visual-map analysis produced padding")
        state = state + runtime.beta * (proposal - state)
    return state


def _map_summary(group_map: list[list[float]]) -> dict[str, float]:
    flat = [float(value) for row in group_map for value in row]
    positive = [max(value, 0.0) for value in flat]
    total = sum(positive)
    ordered = sorted(positive, reverse=True)
    top_count = max(1, len(ordered) // 4)
    return {
        "mean_margin_drop": sum(flat) / len(flat),
        "max_margin_drop": max(flat),
        "positive_region_fraction": sum(value > 0 for value in flat) / len(flat),
        "top_quartile_positive_mass": sum(ordered[:top_count]) / total if total else 0.0,
    }


def _cosine(left: list[list[float]], right: list[list[float]]) -> float:
    import math

    a = [float(value) for row in left for value in row]
    b = [float(value) for row in right for value in row]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / max(norm_a * norm_b, 1e-12)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for transition in (2, 3, 4):
        key = f"T{transition}"
        output[key] = {
            metric: scalar_summary(
                (record["map_summary"][key][metric] for record in records),
                n_boot=10_000,
                seed=transition,
            )
            for metric in (
                "mean_margin_drop",
                "max_margin_drop",
                "positive_region_fraction",
                "top_quartile_positive_mass",
            )
        }
    output["successive_map_cosine"] = {
        "T2_to_T3": scalar_summary(
            (
                _cosine(record["group_maps"]["T2"], record["group_maps"]["T3"])
                for record in records
            ),
            n_boot=10_000,
            seed=23,
        ),
        "T3_to_T4": scalar_summary(
            (
                _cosine(record["group_maps"]["T3"], record["group_maps"]["T4"])
                for record in records
            ),
            n_boot=10_000,
            seed=34,
        ),
    }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps != 4:
        raise SystemExit("canonical causal maps require T=4")
    if args.macro_grid < 2:
        raise SystemExit("--macro-grid must be >=2")
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
    root = resolve_vstar_root(args.vstar_root)
    rows = load_vstar_rows(root)
    token_ids = option_token_ids(runtime.processor.tokenizer)
    records = []
    started = time.monotonic()
    for position, index in enumerate(args.indices, start=1):
        row = rows[index]
        example = vstar_example(runtime.processor, root, row, runtime.device)
        with runtime.torch.inference_mode():
            trace = runtime.extract(example)
            clean_state = runtime.rollout(trace, steps=4)[-1]
            clean = classification_result(
                runtime.token_scores(trace, clean_state, token_ids, path="strict"),
                LETTERS,
                row["label"],
            )
            layout = _region_layout(trace, runtime, args.macro_grid)
            group_maps = {}
            for transition in (2, 3, 4):
                values = [
                    [0.0 for _ in range(layout["columns"])]
                    for _ in range(layout["rows"])
                ]
                for region in layout["regions"]:
                    state = _rollout_with_region_block(
                        runtime,
                        trace,
                        blocked_transition=transition,
                        key_mask=region["key_mask"],
                        steps=4,
                    )
                    changed = classification_result(
                        runtime.token_scores(trace, state, token_ids, path="strict"),
                        LETTERS,
                        row["label"],
                    )
                    values[region["row"]][region["column"]] = (
                        _gold_margin(clean) - _gold_margin(changed)
                    )
                group_maps[f"T{transition}"] = values
        records.append(
            {
                "global_index": index,
                "question_id": str(row["question_id"]),
                "category": row.get("category"),
                "label": row["label"],
                "text": row["text"],
                "image": row["image"],
                "clean": clean,
                "visual_grid": [layout["height"], layout["width"]],
                "macro_grid": [layout["rows"], layout["columns"]],
                "group_maps": group_maps,
                "map_summary": {
                    key: _map_summary(value) for key, value in group_maps.items()
                },
            }
        )
        release_trace(trace)
        elapsed = time.monotonic() - started
        print(
            f"causal-map {position}/{len(args.indices)} elapsed={elapsed/60:.1f} min",
            flush=True,
        )
    payload = {
        "kind": "cvrr_causal_visual_map",
        "protocol": {
            "dataset": "V*",
            "indices": args.indices,
            "steps": 4,
            "macro_grid": args.macro_grid,
            "intervention": (
                "block one spatial region's question-query to image-key/value "
                "edges at exactly one recurrent transition"
            ),
            "score": "clean final gold margin minus intervened final gold margin",
        },
        "runtime": runtime.metadata(),
        "n": len(records),
        "summary": summarize(records),
        "records": records,
        "meta": run_metadata(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(out)
    print(f"wrote {out}", flush=True)
    return payload


def merge_payloads(paths: list[pathlib.Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    records = []
    seen = set()
    for payload in payloads:
        for record in payload["records"]:
            key = str(record["question_id"])
            if key not in seen:
                records.append(record)
                seen.add(key)
    output = dict(payloads[0])
    output["protocol"] = dict(output["protocol"])
    output["protocol"]["indices"] = [record["global_index"] for record in records]
    output["n"] = len(records)
    output["records"] = records
    output["summary"] = summarize(records)
    output["merge_inputs"] = [str(path) for path in paths]
    return output



def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
