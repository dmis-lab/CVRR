"""Merge disjoint CVRR analysis shards and recompute paired statistics."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Callable

from scripts.analysis_cvrr.core.metrics import summarize_condition_records
from scripts.analysis_cvrr.recurrence.visual_access import _summarize as summarize_persistent
from scripts.analysis_cvrr.recurrence.depth import (
    _summarize_depth as summarize_depth,
    _summarize_factorial as summarize_factorial,
)
from scripts.analysis_cvrr.recurrence.reread import _summarize as summarize_reread
from scripts.analysis_cvrr.recurrence.components import summarize as summarize_components
from scripts.analysis_cvrr.diagnostics.transition import summarize as summarize_transition


SUPPORTED_KINDS = {
    "cvrr_causal_reliance",
    "cvrr_recurrence_mechanism",
    "cvrr_persistent_visual_ablation",
    "cvrr_reread_correction",
    "cvrr_component_ablation",
    "cvrr_transition_activation",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load(path: str) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved = pathlib.Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"result is not a JSON object: {resolved}")
    return resolved, payload


def _identity_subset(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime", {})
    return {
        "checkpoint": runtime.get("checkpoint"),
        "model_type": runtime.get("model_type"),
        "ell_star": runtime.get("ell_star"),
        "cell_layer": runtime.get("cell_layer"),
        "beta": runtime.get("beta"),
        "max_visual_tokens": runtime.get("max_visual_tokens"),
    }


def _protocol_subset(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    protocol = payload.get("protocol", {})
    if kind == "cvrr_causal_reliance":
        names = ("dataset", "steps", "conditions", "paths")
    elif kind == "cvrr_recurrence_mechanism":
        names = (
            "depth_dataset",
            "factorial_dataset",
            "depths",
            "factorial_max_steps",
        )
    elif kind == "cvrr_persistent_visual_ablation":
        names = ("dataset", "steps")
    elif kind == "cvrr_reread_correction":
        names = ("dataset", "steps", "read_on", "read_off")
    elif kind == "cvrr_component_ablation":
        names = ("dataset", "steps", "conditions", "ablation_scope")
    else:
        names = ("dataset", "steps", "metric", "supports")
    return {name: protocol.get(name) for name in names}


def _assert_compatible(payloads: list[dict[str, Any]]) -> str:
    kinds = {payload.get("kind") for payload in payloads}
    if len(kinds) != 1:
        raise ValueError(f"cannot merge different result kinds: {sorted(kinds)}")
    kind = str(next(iter(kinds)))
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported merge kind {kind!r}")
    reference = _identity_subset(payloads[0])
    reference_protocol = _protocol_subset(payloads[0], kind)
    for index, payload in enumerate(payloads[1:], start=1):
        current = _identity_subset(payload)
        if current != reference:
            raise ValueError(
                f"input {index} has incompatible runtime metadata:\n"
                f"reference={reference}\ncurrent={current}"
            )
        current_protocol = _protocol_subset(payload, kind)
        if current_protocol != reference_protocol:
            raise ValueError(
                f"input {index} has incompatible protocol:\n"
                f"reference={reference_protocol}\ncurrent={current_protocol}"
            )
    return kind


def _deduplicate(
    records: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    unique: dict[Any, dict[str, Any]] = {}
    for record in records:
        identifier = key(record)
        previous = unique.get(identifier)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate record {identifier!r}")
        unique[identifier] = record
    return [unique[identifier] for identifier in sorted(unique, key=str)]


def _merge_causal(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    from scripts.analysis_cvrr.causal.reliance import (
        _bypass_sensitivity,
        _path_interactions,
        _path_sensitivity,
    )

    records = _deduplicate(
        [record for payload in payloads for record in payload.get("records", [])],
        lambda record: str(record["question_id"]),
    )
    summary = summarize_condition_records(
        records, n_boot=bootstrap, seed=seed
    )
    output = dict(payloads[0])
    output.update(
        {
            "n": len(records),
            "summary": summary,
            "bypass_restoration": _bypass_sensitivity(summary),
            "path_sensitivity": _path_sensitivity(summary),
            "path_interactions": _path_interactions(
                records, n_boot=bootstrap, seed=seed
            ),
            "records": records,
        }
    )
    return output


def _merge_recurrence(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    protocol = payloads[0]["protocol"]
    steps = [int(value) for value in protocol["depths"]]
    max_steps = int(protocol["factorial_max_steps"])
    depth_records = _deduplicate(
        [
            record
            for payload in payloads
            for record in payload.get("depth", {}).get("records", [])
        ],
        lambda record: str(record["question_id"]),
    )
    factorial_records = _deduplicate(
        [
            record
            for payload in payloads
            for record in payload.get("factorial", {}).get("records", [])
        ],
        lambda record: (int(record["pair_index"]), str(record["direction"])),
    )
    output = dict(payloads[0])
    output["depth"] = {
        "n": len(depth_records),
        "summary": summarize_depth(
            depth_records, steps, n_boot=bootstrap, seed=seed
        ),
        "records": depth_records,
    }
    output["factorial"] = {
        "n": len(factorial_records),
        "summary": summarize_factorial(
            factorial_records,
            max_steps=max_steps,
            n_boot=bootstrap,
            seed=seed,
        ),
        "records": factorial_records,
    }
    return output


def _merge_persistent(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    steps = int(payloads[0]["protocol"]["steps"])
    records = _deduplicate(
        [record for payload in payloads for record in payload.get("records", [])],
        lambda record: str(record["question_id"]),
    )
    output = dict(payloads[0])
    output.update(
        {
            "n": len(records),
            "summary": summarize_persistent(
                records, steps=steps, bootstrap=bootstrap, seed=seed
            ),
            "records": records,
        }
    )
    return output


def _merge_reread(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    steps = int(payloads[0]["protocol"]["steps"])
    records = _deduplicate(
        [record for payload in payloads for record in payload.get("records", [])],
        lambda record: (int(record["pair_index"]), str(record["direction"])),
    )
    output = dict(payloads[0])
    output.update(
        {
            "n": len(records),
            "summary": summarize_reread(
                records,
                steps=steps,
                bootstrap=bootstrap,
                seed=seed,
            ),
            "records": records,
        }
    )
    return output


def _merge_components(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    records = _deduplicate(
        [record for payload in payloads for record in payload.get("records", [])],
        lambda record: str(record["question_id"]),
    )
    output = dict(payloads[0])
    output.update(
        {
            "n": len(records),
            "summary": summarize_components(
                records, bootstrap=bootstrap, seed=seed
            ),
            "records": records,
        }
    )
    return output


def _merge_transition(
    payloads: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    records = _deduplicate(
        [record for payload in payloads for record in payload.get("records", [])],
        lambda record: str(record["question_id"]),
    )
    reference_weights = payloads[0].get("weights")
    if any(payload.get("weights") != reference_weights for payload in payloads[1:]):
        raise ValueError("transition shards disagree on learned LoRA weights")
    output = dict(payloads[0])
    output.update(
        {
            "n": len(records),
            "summary": summarize_transition(
                records, bootstrap=bootstrap, seed=seed
            ),
            "records": records,
        }
    )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")
    loaded = [_load(path) for path in args.inputs]
    paths = [path for path, _ in loaded]
    payloads = [payload for _, payload in loaded]
    kind = _assert_compatible(payloads)
    if kind == "cvrr_causal_reliance":
        output = _merge_causal(payloads, bootstrap=args.bootstrap, seed=args.seed)
    elif kind == "cvrr_recurrence_mechanism":
        output = _merge_recurrence(
            payloads, bootstrap=args.bootstrap, seed=args.seed
        )
    elif kind == "cvrr_persistent_visual_ablation":
        output = _merge_persistent(
            payloads, bootstrap=args.bootstrap, seed=args.seed
        )
    elif kind == "cvrr_reread_correction":
        output = _merge_reread(
            payloads, bootstrap=args.bootstrap, seed=args.seed
        )
    elif kind == "cvrr_component_ablation":
        output = _merge_components(
            payloads, bootstrap=args.bootstrap, seed=args.seed
        )
    else:
        output = _merge_transition(
            payloads, bootstrap=args.bootstrap, seed=args.seed
        )
    output["merge"] = {
        "inputs": [str(path) for path in paths],
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    out = pathlib.Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, default=str))
    temporary.replace(out)
    print(f"wrote {out}", flush=True)
    return output


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
