"""Locate the causal visual-read boundary by bidirectional activation patching.

The input JSON must contain exact same-question, different-image pairs under a
top-level ``pairs`` list.  Each side has ``image`` and ``answer`` fields.  This
analysis runs the frozen pretrained Qwen2.5-VL backbone, patches image-token
rows at one layer at a time, and selects the onset of the strongest sustained
decline in the sample-normalized mediation curve.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

from scripts.analysis_cvrr.core.common import configure_huggingface, run_metadata, set_seed
from scripts.analysis_cvrr.core.data import LOCALIZATION_ANSWER_HINT, configure_visual_cap


@dataclass
class PatchExample:
    input_ids: Any
    attention_mask: Any
    pixel_values: Any
    image_grid_thw: Any
    own_answer_token: int
    other_answer_token: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--pairs", required=True)
    parser.add_argument(
        "--image-root",
        required=True,
        help="Extracted image root containing gqa/<image> or direct image names.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=8192)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _prompt(processor, question: str) -> str:
    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": question.strip() + LOCALIZATION_ANSWER_HINT,
                    },
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _first_answer_token(tokenizer, prompt: str, answer: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(
        prompt + answer.strip(), add_special_tokens=False
    ).input_ids
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"answer {answer!r} contributed no continuation token")
    return int(full_ids[len(prompt_ids)])


def _open_pair_image(root: pathlib.Path, name: str):
    from PIL import Image

    candidates = (root / "gqa" / name, root / name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"image {name!r} absent below {root}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _encode(processor, image, prompt: str, device, dtype) -> dict[str, Any]:
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    encoded["pixel_values"] = encoded["pixel_values"].to(dtype)
    return encoded


def _build_examples(processor, record: dict[str, Any], root, device, dtype):
    question = str(record["question"])
    prompt = _prompt(processor, question)
    left_token = _first_answer_token(
        processor.tokenizer, prompt, str(record["x"]["answer"])
    )
    right_token = _first_answer_token(
        processor.tokenizer, prompt, str(record["x_prime"]["answer"])
    )
    if left_token == right_token:
        return None
    encoded = {
        side: _encode(
            processor,
            _open_pair_image(root, str(record[side]["image"])),
            prompt,
            device,
            dtype,
        )
        for side in ("x", "x_prime")
    }
    left = PatchExample(
        input_ids=encoded["x"]["input_ids"],
        attention_mask=encoded["x"]["attention_mask"],
        pixel_values=encoded["x"]["pixel_values"],
        image_grid_thw=encoded["x"]["image_grid_thw"],
        own_answer_token=left_token,
        other_answer_token=right_token,
    )
    right = PatchExample(
        input_ids=encoded["x_prime"]["input_ids"],
        attention_mask=encoded["x_prime"]["attention_mask"],
        pixel_values=encoded["x_prime"]["pixel_values"],
        image_grid_thw=encoded["x_prime"]["image_grid_thw"],
        own_answer_token=right_token,
        other_answer_token=left_token,
    )
    return left, right


def _collect(model, item: PatchExample) -> dict[str, Any]:
    from cvrr.modeling_cvrr import _make_context, _run_layers

    embeddings = model.model.get_input_embeddings()(item.input_ids)
    # Repeat the stock multimodal embedding path explicitly so every layer
    # state is available for patching.
    image_features = model.model.get_image_features(
        item.pixel_values, item.image_grid_thw
    )
    import torch

    image_features = torch.cat(image_features, dim=0).to(
        embeddings.device, embeddings.dtype
    )
    image_mask, _ = model.model.get_placeholder_mask(
        item.input_ids,
        inputs_embeds=embeddings,
        image_features=image_features,
    )
    embeddings = embeddings.masked_scatter(image_mask, image_features)
    positions, _ = model.model.get_rope_index(
        item.input_ids,
        item.image_grid_thw,
        None,
        second_per_grid_ts=None,
        attention_mask=item.attention_mask,
    )
    context = _make_context(
        model.model.language_model,
        embeddings,
        positions,
        item.attention_mask,
    )
    hidden = embeddings
    states = []
    for layer in range(len(model.model.language_model.layers)):
        hidden = _run_layers(
            model.model.language_model,
            context,
            layer,
            layer + 1,
            hidden_states=hidden,
            use_cache=False,
        )
        states.append(hidden)
    visual = item.input_ids.eq(model.config.image_token_id)
    final_position = int(item.attention_mask.sum().item()) - 1
    return {
        "context": context,
        "states": states,
        "visual": visual,
        "final_position": final_position,
    }


def _logits(model, hidden, position: int):
    normalized = model.model.language_model.norm(hidden)
    return model.lm_head(normalized)[:, position]


def _margin(logits, own: int, other: int) -> float:
    return float((logits[:, other] - logits[:, own]).float().item())


def _pair_curve(model, left: PatchExample, right: PatchExample):
    from cvrr.modeling_cvrr import _run_layers

    cached = {"left": _collect(model, left), "right": _collect(model, right)}
    items = {"left": left, "right": right}
    n_layers = len(model.model.language_model.layers)
    for base_name, source_name in (("left", "right"), ("right", "left")):
        base_count = int(cached[base_name]["visual"].sum().item())
        source_count = int(cached[source_name]["visual"].sum().item())
        if base_count != source_count:
            raise ValueError(
                "paired images produce different visual-token counts: "
                f"{base_count} versus {source_count}"
            )

    clean = {}
    for name in ("left", "right"):
        item = items[name]
        data = cached[name]
        clean[name] = _margin(
            _logits(model, data["states"][-1], data["final_position"]),
            item.own_answer_token,
            item.other_answer_token,
        )

    curve = []
    for layer in range(n_layers):
        row = {}
        for direction, base_name, source_name in (
            ("forward", "left", "right"),
            ("backward", "right", "left"),
        ):
            base = cached[base_name]
            source = cached[source_name]
            item = items[base_name]
            hidden = base["states"][layer].clone()
            hidden[base["visual"]] = source["states"][layer][source["visual"]].to(
                hidden.dtype
            )
            if layer + 1 < n_layers:
                hidden = _run_layers(
                    model.model.language_model,
                    base["context"],
                    layer + 1,
                    None,
                    hidden_states=hidden,
                    use_cache=False,
                )
            patched = _margin(
                _logits(model, hidden, base["final_position"]),
                item.own_answer_token,
                item.other_answer_token,
            )
            row[direction] = patched - clean[base_name]
        row["mean"] = 0.5 * (row["forward"] + row["backward"])
        curve.append(row)
    return curve


def _strongest_sustained_decline(values: list[float], window: int):
    if window < 2 or len(values) <= window:
        raise ValueError("invalid sustained-decline window")
    drops = [
        values[index] - values[index + window]
        for index in range(len(values) - window)
    ]
    onset = max(range(len(drops)), key=drops.__getitem__)
    return onset, drops[onset]


def _bootstrap(curves, *, window: int, n_boot: int, seed: int):
    import random

    rng = random.Random(seed)
    width = len(curves[0])
    counts: dict[int, int] = {}
    for _ in range(n_boot):
        sample = [curves[rng.randrange(len(curves))] for _ in curves]
        mean = [sum(row[layer] for row in sample) / len(sample) for layer in range(width)]
        selected, _ = _strongest_sustained_decline(mean, window)
        counts[selected] = counts.get(selected, 0) + 1
    return {str(key): value for key, value in sorted(counts.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard specification")
    configure_huggingface(offline=not args.allow_download)
    set_seed(args.seed)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = torch.device(args.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=not args.allow_download,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor or args.model,
        use_fast=True,
        local_files_only=not args.allow_download,
    )
    configure_visual_cap(processor, model.config, args.max_visual_tokens)

    payload = json.loads(pathlib.Path(args.pairs).read_text())
    pair_records = payload.get("pairs")
    if not isinstance(pair_records, list):
        raise ValueError("pair JSON must contain a top-level 'pairs' list")
    if args.max_pairs:
        pair_records = pair_records[: args.max_pairs]
    pair_records = [
        record
        for index, record in enumerate(pair_records)
        if index % args.num_shards == args.shard_index
    ]
    image_root = pathlib.Path(args.image_root).expanduser().resolve()

    curves = []
    dropped_collision = 0
    started = time.monotonic()
    with torch.inference_mode():
        for index, record in enumerate(pair_records, start=1):
            built = _build_examples(
                processor, record, image_root, device, model.dtype
            )
            if built is None:
                dropped_collision += 1
                continue
            curves.append(_pair_curve(model, *built))
            if index % 10 == 0 or index == len(pair_records):
                elapsed = time.monotonic() - started
                eta = (len(pair_records) - index) * elapsed / max(index, 1) / 60
                print(
                    f"localization {index}/{len(pair_records)} eta={eta:.1f} min",
                    flush=True,
                )
    if not curves:
        raise RuntimeError("no usable localization pairs")

    usable = [curve for curve in curves if abs(float(curve[0]["mean"])) > 1e-6]
    if not usable:
        raise RuntimeError("all pair-level layer-0 mediation values are zero")
    normalized = [
        [float(row["mean"]) / float(curve[0]["mean"]) for row in curve]
        for curve in usable
    ]
    n_layers = len(curves[0])
    per_layer = []
    for layer in range(n_layers):
        raw = [float(curve[layer]["mean"]) for curve in curves]
        relative = [curve[layer] for curve in normalized]
        per_layer.append(
            {
                "layer": layer,
                "m": sum(raw) / len(raw),
                "sem": statistics.stdev(raw) / math.sqrt(len(raw)) if len(raw) > 1 else 0.0,
                "m_rel": sum(relative) / len(relative),
                "sem_rel": (
                    statistics.stdev(relative) / math.sqrt(len(relative))
                    if len(relative) > 1
                    else 0.0
                ),
            }
        )
    mean_curve = [row["m_rel"] for row in per_layer]
    selected, window_drop = _strongest_sustained_decline(mean_curve, args.window)
    result = {
        "kind": "cvrr_boundary_localization",
        "model": args.model,
        "protocol": {
            "intervention": "bidirectional image-token-row activation patching",
            "normalization": "each pair divided by its own layer-0 mediation",
            "selection": "onset of strongest sustained decline",
            "window": args.window,
            "max_visual_tokens": args.max_visual_tokens,
        },
        "n_pairs": len(curves),
        "n_normalized_pairs": len(usable),
        "dropped_first_token_collisions": dropped_collision,
        "selected_ell_star": selected,
        "window_drop": window_drop,
        "per_layer": per_layer,
        "bootstrap_counts": _bootstrap(
            normalized,
            window=args.window,
            n_boot=args.bootstrap,
            seed=args.seed,
        ),
        "raw_curves": curves,
        "meta": run_metadata(args.seed),
    }
    output = pathlib.Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    print(f"wrote {output}; selected ell*={selected}", flush=True)
    return result


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
