from __future__ import annotations

import argparse
import csv
import io
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATASET_IDS = {
    "sdxl": "rhythmghai/ai-vs-real-images-dataset",
    "multigen": "cartografia/unbiased-tiny-genimage",
}
MODEL_ID = "OwensLab/commfor-model-224"
OFFICIAL_COMMIT = "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"
MULTIGEN_DIRECTORIES = (
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Community Forensics baseline")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/laid/results/community-forensics-224"),
    )
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--dataset-kind", choices=("sdxl", "multigen"), default="sdxl")
    return parser.parse_args()


def find_class_directory(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name!r} directory under {root}, found {len(matches)}")
    return matches[0]


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stratified_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.parent.name].append(path)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    chosen: list[Path] = []
    names = sorted(groups)
    while len(chosen) < count and any(groups.values()):
        for name in names:
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].pop())
    if len(chosen) < count:
        raise RuntimeError(f"requested {count} images, but only found {len(chosen)}")
    return chosen


def random_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    if len(paths) < count:
        raise RuntimeError(f"requested {count} images, but only found {len(paths)}")
    return random.Random(seed).sample(paths, count)


def select_sdxl(root: Path, count: int, seed: int) -> list[tuple[Path, int, str]]:
    real_root = find_class_directory(root, "real_dataset")
    ai_root = find_class_directory(root, "Ai_generated_dataset")
    return [
        (path, 0, "sdxl")
        for path in stratified_sample(image_paths(real_root), count, seed)
    ] + [
        (path, 1, "sdxl")
        for path in stratified_sample(image_paths(ai_root), count, seed + 1)
    ]


def select_multigen(root: Path, count: int, seed: int) -> list[tuple[Path, int, str]]:
    nature = find_class_directory(root, "Nature")
    real_paths = random_sample(
        image_paths(nature), count * len(MULTIGEN_DIRECTORIES), seed
    )
    selected: list[tuple[Path, int, str]] = []
    for index, directory_name in enumerate(MULTIGEN_DIRECTORIES):
        generator_root = find_class_directory(root, directory_name)
        generator = directory_name.lower()
        real_start = index * count
        selected.extend(
            (path, 0, generator)
            for path in real_paths[real_start : real_start + count]
        )
        selected.extend(
            (path, 1, generator)
            for path in random_sample(image_paths(generator_root), count, seed + index + 1)
        )
    return selected


def degraded(image, name: str):
    from PIL import Image

    image = image.convert("RGB")
    if name == "none":
        return image
    if name == "jpeg75":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name == "downscale50":
        small = image.resize(
            (max(1, image.width // 2), max(1, image.height // 2)),
            Image.Resampling.LANCZOS,
        )
        return small.resize(image.size, Image.Resampling.LANCZOS)
    raise ValueError(f"unknown degradation: {name}")


def main() -> int:
    args = parse_args()
    if args.per_class < 2:
        raise ValueError("--per-class must be at least 2")
    if not args.input_root.exists():
        raise RuntimeError(
            f"Kaggle input root not found at {args.input_root}. Select Add Input and attach "
            f"{DATASET_IDS[args.dataset_kind]}."
        )

    import torch
    from PIL import Image
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; attach a Kaggle GPU before running the baseline")

    official_repo = Path("/kaggle/working/community-forensics")
    if not (official_repo / "models.py").exists():
        raise RuntimeError("the pinned Community Forensics source checkout is missing")
    sys.path.insert(0, str(official_repo))
    import models  # type: ignore[import-not-found]

    try:
        selected = (
            select_sdxl(args.input_root, args.per_class, args.seed)
            if args.dataset_kind == "sdxl"
            else select_multigen(args.input_root, args.per_class, args.seed)
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not discover {DATASET_IDS[args.dataset_kind]} below {args.input_root}; "
            "confirm it is attached with Add Input"
        ) from exc

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    device = torch.device("cuda:0")
    model = models.ViTClassifier.from_pretrained(MODEL_ID).to(device).eval()

    with Image.open(selected[0][0]) as warmup_source:
        warmup = preprocess(warmup_source.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(3):
            model(warmup)
    torch.cuda.synchronize()

    rows: list[dict[str, str | int | float]] = []
    latencies_ms: dict[str, float] = {}
    for degradation in ("none", "jpeg75", "downscale50"):
        tensors = []
        metadata = []
        for path, label, generator in selected:
            with Image.open(path) as source:
                tensors.append(preprocess(degraded(source, degradation)))
            metadata.append((path, label, generator))

        outputs: list[float] = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(tensors), args.batch_size):
                batch = torch.stack(tensors[offset : offset + args.batch_size]).to(device)
                probabilities = torch.sigmoid(model(batch)).flatten().cpu().tolist()
                outputs.extend(float(value) for value in probabilities)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms[degradation] = elapsed_ms / len(tensors)

        for (path, label, generator), probability in zip(metadata, outputs, strict=True):
            relative = path.relative_to(args.input_root).as_posix()
            rows.append(
                {
                    "sample_id": f"{relative}::{degradation}",
                    "label": label,
                    "probability_ai": probability,
                    "dataset": args.dataset_kind,
                    "generator": generator,
                    "degradation": degradation,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = args.output_dir / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bakeoff.cli",
            "evaluate",
            str(predictions),
            "--threshold",
            "0.65",
            "--output",
            str(report),
        ],
        check=True,
    )
    provenance = {
        "model": MODEL_ID,
        "official_source_commit": OFFICIAL_COMMIT,
        "dataset": DATASET_IDS[args.dataset_kind],
        "dataset_kind": args.dataset_kind,
        "sample_seed": args.seed,
        "samples_per_class": args.per_class,
        "prediction_rows": len(rows),
        "threshold": 0.65,
        "degradations": ["none", "jpeg75", "downscale50"],
        "mean_inference_ms_per_image": latencies_ms,
        "gpu": torch.cuda.get_device_name(0),
        "notes": [
            "Baseline candidate only; not the final LAID model.",
            (
                "Dataset covers SDXL and real Unsplash photos only."
                if args.dataset_kind == "sdxl"
                else "Dataset covers seven legacy-to-mid-generation model families, not FLUX or SD3."
            ),
            "Latency excludes model download, image decoding, and preprocessing.",
            "Three untimed CUDA warm-up passes run before latency measurement.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print(f"Results saved under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
