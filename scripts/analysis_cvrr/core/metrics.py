"""Paired statistics shared by all CVRR analyses.

The primary uncertainty unit is an example (or a contrastive pair), never an
answer token.  Bootstrap intervals therefore resample complete records.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> list[float | None]:
    values = [float(value) for value in values]
    if not values:
        return [None, None]
    if len(values) == 1 or n_boot <= 0:
        value = values[0] if len(values) == 1 else sum(values) / len(values)
        return [value, value]
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    tail = (1.0 - level) / 2.0
    return [percentile(samples, tail), percentile(samples, 1.0 - tail)]


def scalar_summary(
    values: Iterable[float], *, n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_error": None,
            "ci95": [None, None],
            "positive_fraction": None,
        }
    ordered = sorted(values)
    value_mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - value_mean) ** 2 for value in values) / (
            len(values) - 1
        )
        standard_error = math.sqrt(variance / len(values))
    else:
        standard_error = 0.0
    return {
        "n": len(values),
        "mean": value_mean,
        "median": percentile(ordered, 0.5),
        "standard_error": standard_error,
        "ci95": bootstrap_mean_ci(values, n_boot=n_boot, seed=seed),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
    }


def exact_mcnemar_p(clean_correct: Sequence[bool], changed_correct: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value using the discordant-pair binomial."""

    if len(clean_correct) != len(changed_correct):
        raise ValueError("paired correctness arrays have different lengths")
    regress = sum(a and not b for a, b in zip(clean_correct, changed_correct))
    improve = sum((not a) and b for a, b in zip(clean_correct, changed_correct))
    discordant = regress + improve
    if discordant == 0:
        return 1.0
    lower = min(regress, improve)
    one_tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * one_tail)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise correction with monotonic adjusted p-values."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def classification_result(
    scores: Sequence[float], labels: Sequence[str], gold: str
) -> dict[str, Any]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same nonzero length")
    if gold not in labels:
        raise ValueError(f"gold label {gold!r} is absent from candidates")
    pred_index = max(range(len(scores)), key=lambda index: float(scores[index]))
    gold_index = labels.index(gold)
    best_wrong = max(
        float(score) for index, score in enumerate(scores) if index != gold_index
    )
    return {
        "scores": [float(score) for score in scores],
        "pred": labels[pred_index],
        "correct": pred_index == gold_index,
        "gold_margin": float(scores[gold_index]) - best_wrong,
    }


def summarize_condition_records(
    records: Sequence[dict[str, Any]],
    *,
    result_key: str = "results",
    clean_name: str = "clean",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize ``record[result_key][path][condition]`` paired outcomes."""

    paths = sorted(
        {
            path
            for record in records
            for path in record.get(result_key, {})
        }
    )
    output: dict[str, Any] = {}
    for path in paths:
        conditions = sorted(
            {
                condition
                for record in records
                for condition in record.get(result_key, {}).get(path, {})
            }
        )
        path_summary: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for condition in conditions:
            rows = [
                record[result_key][path][condition]
                for record in records
                if condition in record.get(result_key, {}).get(path, {})
            ]
            correctness = [bool(row["correct"]) for row in rows]
            margins = [float(row["gold_margin"]) for row in rows]
            summary: dict[str, Any] = {
                "n": len(rows),
                "accuracy": sum(correctness) / len(correctness),
                "accuracy_stats": scalar_summary(
                    (float(value) for value in correctness),
                    n_boot=n_boot,
                    seed=seed,
                ),
                "gold_margin": scalar_summary(
                    margins, n_boot=n_boot, seed=seed
                ),
            }
            paired_records = [
                record
                for record in records
                if condition in record.get(result_key, {}).get(path, {})
                and clean_name in record.get(result_key, {}).get(path, {})
            ]
            if condition != clean_name and paired_records:
                clean = [
                    record[result_key][path][clean_name] for record in paired_records
                ]
                changed = [
                    record[result_key][path][condition] for record in paired_records
                ]
                correctness_delta = [
                    float(bool(after["correct"])) - float(bool(before["correct"]))
                    for before, after in zip(clean, changed)
                ]
                margin_delta = [
                    float(after["gold_margin"]) - float(before["gold_margin"])
                    for before, after in zip(clean, changed)
                ]
                p_value = exact_mcnemar_p(
                    [bool(row["correct"]) for row in clean],
                    [bool(row["correct"]) for row in changed],
                )
                raw_p[condition] = p_value
                summary["vs_clean"] = {
                    "n": len(paired_records),
                    "accuracy_delta": scalar_summary(
                        correctness_delta, n_boot=n_boot, seed=seed
                    ),
                    "gold_margin_delta": scalar_summary(
                        margin_delta, n_boot=n_boot, seed=seed + 1
                    ),
                    "prediction_flip_fraction": sum(
                        before["pred"] != after["pred"]
                        for before, after in zip(clean, changed)
                    )
                    / len(paired_records),
                    "regress_fraction": sum(
                        bool(before["correct"]) and not bool(after["correct"])
                        for before, after in zip(clean, changed)
                    )
                    / len(paired_records),
                    "improve_fraction": sum(
                        not bool(before["correct"]) and bool(after["correct"])
                        for before, after in zip(clean, changed)
                    )
                    / len(paired_records),
                    "mcnemar_exact_p": p_value,
                }
            path_summary[condition] = summary
        adjusted = holm_adjust(raw_p)
        for condition, p_value in adjusted.items():
            path_summary[condition]["vs_clean"]["mcnemar_holm_p"] = p_value
        output[path] = path_summary
    return output


def grouped_scalar_summary(
    rows: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
    value: Callable[[dict[str, Any]], float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(float(value(row)))
    return {
        "/".join(map(str, group)): scalar_summary(
            values, n_boot=n_boot, seed=seed
        )
        for group, values in sorted(groups.items(), key=lambda item: item[0])
    }
