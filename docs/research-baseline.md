# Verified research baseline

Checked on 2026-08-13. This file records why each candidate is in the bake-off;
it is not evidence that the candidate meets the bounty threshold.

## Primary candidates

### FerretNet

- Official source: <https://github.com/xigua7105/FerretNet>
- Paper: <https://arxiv.org/abs/2509.20890>
- License: Apache-2.0.
- Reported architecture: 1.06M parameters, 2.38 GFLOPs, 256-pixel input.
- Reason to test: its local-pixel-dependency residual is complementary to a
  semantic detector and unusually small for browser deployment.
- Unverified item: ONNX/WebGPU export and accuracy at a fixed 0.65 threshold.

### Community Forensics

- Official source: <https://github.com/JeongsooP/Community-Forensics>
- Paper: <https://arxiv.org/abs/2411.04125>
- Official 224 weights: <https://huggingface.co/OwensLab/commfor-model-224>
- License: MIT.
- Verified source-weight size: 86,678,644 bytes.
- Reason to test: training covers 2.7M images from 4,803 generators, making it
  the best available teacher candidate for broad generator diversity.
- Unverified item: exported browser latency and accuracy under our slices.

### SAFE

- Official source: <https://github.com/ouxiang-li/SAFE>
- Paper: <https://arxiv.org/abs/2408.06741>
- License: Apache-2.0.
- Reported architecture: 1.44M parameters.
- Reason to test: lightweight baseline with artifact-preserving preprocessing
  and training augmentations designed to reduce content bias.
- Unverified item: ONNX/WebGPU export and fixed-threshold calibration.

### Nonescape Mini

- Weights: <https://huggingface.co/e3ntity/nonescape-v0>
- License: Apache-2.0.
- Verified mini weight size: 86,666,672 bytes.
- Reason to test: accessible model artifacts and documented JavaScript/ONNX
  support make it useful as an early end-to-end pipeline baseline.
- Unverified item: independent performance on the selected benchmark slices.

## Evaluation sources

- GenImage: <https://github.com/GenImage-Dataset/GenImage>
- AIGIBench: <https://github.com/HorizonTEL/AIGIBench>
- Synthbuster: <https://github.com/qbammey/synthbuster>
- Community Forensics evaluation data:
  <https://github.com/JeongsooP/Community-Forensics>
- Real-world robustness dataset:
  <https://zenodo.org/records/14963880>

No complete dataset is to be downloaded to the local workstation. Small,
balanced, license-compatible validation shards should be materialized on a
remote runner and copied back only as prediction CSVs and summary reports.

## First-pass external slice

- Kaggle source: <https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset>
- Verified size: 248,868,789 bytes with 250 SDXL and 745 real Unsplash images.
- License shown by Kaggle: CC BY-NC-SA 4.0.
- Use: evaluation only; select 100 samples per class across five content
  categories and do not redistribute the images.
- LAID-owned robustness slices: untouched, JPEG quality 75, and 50% downscale
  followed by upscale.
- Limitation: a single generator family cannot establish broad generalization;
  this is a pipeline and failure-analysis baseline only.

## Second-pass multi-generator slice

- Kaggle source: <https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage>
- Verified size: 2,525,870,358 bytes and 23,329 files.
- Composition: 5,828 real images plus 2,500 each from ADM, BigGAN, GLIDE,
  Midjourney, Stable Diffusion 1.5, VQDM, and Wukong.
- Bias control: AI files were normalized to JPEG quality 96 and real images
  were selected for matching compression and dimensions.
- Sampling: 100 AI and 100 distinct real images per generator comparison,
  repeated for clean, JPEG quality 75, and downscale stress.
- Limitation: this adds generator breadth but does not represent newer FLUX,
  Stable Diffusion 3, or other recent commercial generators.
