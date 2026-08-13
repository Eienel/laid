from __future__ import annotations

import csv
from pathlib import Path

from .metrics import Prediction


REQUIRED_COLUMNS = {"sample_id", "label", "probability_ai"}


def read_predictions(path: Path) -> list[Prediction]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required CSV columns: {', '.join(sorted(missing))}")
        rows = [
            Prediction(
                sample_id=row["sample_id"],
                label=int(row["label"]),
                probability_ai=float(row["probability_ai"]),
                dataset=row.get("dataset") or "unknown",
                generator=row.get("generator") or "unknown",
                degradation=row.get("degradation") or "none",
            )
            for row in reader
        ]
    if not rows:
        raise ValueError("prediction CSV is empty")
    return rows
