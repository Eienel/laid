# Remote evaluation runner specification

The workstation is the control plane. Model conversion, training and full
evaluation require a separate GPU runner.

Minimum recommended remote runner:

- Linux with an NVIDIA GPU and at least 16 GB VRAM;
- at least 32 GB system RAM;
- at least 250 GB free disk for initial curated shards;
- Python 3.11 or 3.12;
- one detector process per GPU during the first bake-off.

Every remote run must produce:

1. a prediction CSV matching the contract in the root README;
2. model source URL, source commit and weight checksum;
3. preprocessing and class-order metadata;
4. peak CPU RAM and peak GPU VRAM;
5. median and p95 single-image latency at batch size one;
6. an ONNX export parity report against the original PyTorch outputs.

Do not tune against final test partitions. Calibration data, model-selection
data and final generator-disjoint test data must be separate. The threshold is
always reported at 0.65 even when a diagnostic threshold sweep is performed.
