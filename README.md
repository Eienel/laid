# Local AI detector bounty: model bake-off

This repository starts with evidence, not the Chrome UI. Its first job is to
compare detector predictions under the bounty's exact decision rule:
`probability_ai >= 0.65` means AI-generated.

## Resource policy

The current workstation has about 8 GB RAM, four logical CPU threads, and no
 NVIDIA GPU. Therefore:

- local training is disabled;
- large datasets and PyTorch are not installed locally;
- only one model may be loaded at a time;
- local data-loader concurrency is capped at one;
- preflight defaults to at least 2 GB available RAM and 20 GB free disk;
- large training and full benchmark runs belong on a remote GPU machine.

## Kaggle quick start

Kaggle is the preferred free GPU runner for the bake-off. Download or import
[`notebooks/laid_kaggle_bootstrap.ipynb`](notebooks/laid_kaggle_bootstrap.ipynb)
into a new Kaggle notebook. In Notebook options, select a GPU accelerator and
turn Internet on, then choose **Run All**.

The first cell refuses to continue unless CUDA is actually available. The
notebook then clones this public repository, installs only the lightweight
benchmark package, checks Kaggle's RAM and disk headroom, and runs the tests.
Stop the Kaggle session when finished so idle time does not consume GPU quota.

Run the resource check before any model or dataset operation:

```powershell
$python = 'C:\Users\EIENEL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = "$PWD\src"
& $python -m bakeoff.cli preflight --require-safe
```

If available RAM is below the safety floor, the command exits without starting
the job. Do not bypass the floor for model inference.

## Prediction contract

Each detector adapter will output a CSV with these required columns:

- `sample_id`: stable sample identifier;
- `label`: `0` for real and `1` for AI-generated;
- `probability_ai`: calibrated probability from 0 through 1.

Optional slice columns are `dataset`, `generator`, and `degradation`.

Evaluate a prediction file:

```powershell
& $python -m bakeoff.cli evaluate examples/predictions.csv --threshold 0.65
```

The report includes overall balanced accuracy and dataset, generator, and
degradation slices. A slice containing only one class is explicitly marked as
not scored instead of producing a misleading number.

## Gates

- Bounty requirement: balanced accuracy >= 0.75 at threshold 0.65.
- Internal extension-build gate: balanced accuracy >= 0.80 at threshold 0.65.
- Modern-generator and web-degradation slices must also remain credible; a good
  aggregate score is not sufficient by itself.

## Next implementation milestone

Add adapters for FerretNet and Community Forensics, generate prediction files
on small public validation shards, then measure accuracy, model size, and
single-image latency. Full training and large evaluations remain remote-only.
