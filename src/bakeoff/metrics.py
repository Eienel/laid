from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    label: int
    probability_ai: float
    dataset: str = "unknown"
    generator: str = "unknown"
    degradation: str = "none"


@dataclass(frozen=True)
class Metrics:
    samples: int
    threshold: float
    true_real: int
    false_ai: int
    false_real: int
    true_ai: int
    real_recall: float
    ai_recall: float
    balanced_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def score(rows: Iterable[Prediction], threshold: float = 0.65) -> Metrics:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    tn = fp = fn = tp = 0
    for row in rows:
        if row.label not in (0, 1):
            raise ValueError(f"{row.sample_id}: label must be 0 or 1")
        if not 0.0 <= row.probability_ai <= 1.0:
            raise ValueError(f"{row.sample_id}: probability_ai must be between 0 and 1")
        predicted = int(row.probability_ai >= threshold)
        if row.label == 0 and predicted == 0:
            tn += 1
        elif row.label == 0:
            fp += 1
        elif predicted == 0:
            fn += 1
        else:
            tp += 1
    if tn + fp == 0 or tp + fn == 0:
        raise ValueError("balanced accuracy requires at least one real and one AI sample")
    real_recall = tn / (tn + fp)
    ai_recall = tp / (tp + fn)
    return Metrics(
        samples=tn + fp + fn + tp,
        threshold=threshold,
        true_real=tn,
        false_ai=fp,
        false_real=fn,
        true_ai=tp,
        real_recall=real_recall,
        ai_recall=ai_recall,
        balanced_accuracy=(real_recall + ai_recall) / 2,
    )


def grouped_scores(
    rows: list[Prediction], field: str, threshold: float = 0.65
) -> dict[str, dict[str, int | float | str]]:
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field))].append(row)
    results: dict[str, dict[str, int | float | str]] = {}
    for name, group in sorted(groups.items()):
        labels = {row.label for row in group}
        if labels != {0, 1}:
            results[name] = {
                "samples": len(group),
                "status": "not_scored",
                "reason": "slice does not contain both classes",
            }
        else:
            results[name] = score(group, threshold).to_dict()
    return results
