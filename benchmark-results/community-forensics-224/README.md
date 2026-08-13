# Community Forensics 224: first external baseline

Verified from the Kaggle artifacts on 2026-08-13. Re-evaluating all 600
prediction rows locally at threshold 0.65 reproduced `report.json` exactly.
The archive contained 600 unique IDs, 100 real and 100 AI samples for each of
three conditions, with no missing or out-of-range probabilities.

## Result

| Condition | Balanced accuracy | Real recall | AI recall |
| --- | ---: | ---: | ---: |
| Clean | 0.990 | 0.980 | 1.000 |
| JPEG quality 75 | 0.950 | 1.000 | 0.900 |
| Downscale 50% and restore | 0.985 | 0.970 | 1.000 |
| Combined | 0.975 | 0.983 | 0.967 |

The model comfortably clears the bounty's 0.75 requirement and LAID's 0.80
internal gate on this slice. This is not evidence of broad generalization:
the test set contains SDXL images and real Unsplash photos only.

## Failure analysis and LAID direction

- JPEG quality 75 caused ten AI false negatives and reduced AI recall from
  1.00 to 0.90. Several probabilities collapsed far below 0.65, so a simple
  threshold adjustment is unlikely to be a safe robustness fix.
- Two clean real-image false positives recurred after downscaling. This points
  to persistent content or source bias rather than a one-off transform error.
- The next candidate comparison must include newer and multiple generator
  families, browser-like recompression, screenshots, and browser CPU/ONNX
  latency. Community Forensics remains a teacher/baseline, not the final LAID
  model.

Raw prediction rows are intentionally not committed because their sample paths
refer to a third-party evaluation dataset. The submitted ZIP is identified by
the SHA-256 digest in `provenance.json`.
