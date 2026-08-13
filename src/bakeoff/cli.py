from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import read_predictions
from .metrics import grouped_scores, score
from .resources import assert_safe, snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detector-bakeoff")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="inspect machine resource headroom")
    preflight.add_argument("--path", type=Path, default=Path.cwd())
    preflight.add_argument("--require-safe", action="store_true")
    preflight.add_argument("--min-ram-gb", type=float, default=2.0)
    preflight.add_argument("--min-disk-gb", type=float, default=20.0)

    evaluate = sub.add_parser("evaluate", help="score a detector prediction CSV")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.65)
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            state = (
                assert_safe(
                    args.path,
                    min_available_ram_gb=args.min_ram_gb,
                    min_free_disk_gb=args.min_disk_gb,
                )
                if args.require_safe
                else snapshot(args.path)
            )
            print(json.dumps(state.to_dict(), indent=2))
            return 0

        rows = read_predictions(args.predictions)
        report = {
            "overall": score(rows, args.threshold).to_dict(),
            "by_dataset": grouped_scores(rows, "dataset", args.threshold),
            "by_generator": grouped_scores(rows, "generator", args.threshold),
            "by_degradation": grouped_scores(rows, "degradation", args.threshold),
        }
        payload = json.dumps(report, indent=2)
        print(payload)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
