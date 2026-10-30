# Final pipeline validation

Validated locally on **2026-10-30**, Apple M1 Pro / 16 GiB, Python 3.11.15, ONNX Runtime 1.31.1, CPU provider, four threads. This separates executable pipeline evidence from the older model-selection benchmark.

## Actual inference

`./colorize.sh input` processed all five research pages with the final model-3-only worker. It created **five folders, 25 RGB PNGs**, and `report.html`, `metrics.csv`, `manifest.json`.

| Page | Final-script processing seconds |
|---|---:|
| 1194-001.png | 3.503 |
| 1194-003.png | 4.403 |
| 1194-004.png | 3.700 |
| 1194-005.png | 3.443 |
| 1194-006.png | 3.497 |

Total worker image processing: **18.546 seconds**. Initial model loading: **0.776 seconds**, separate from these image times. These jobs include all five variants, saves and validation. They are different measurements from the older 2.662-second repeated warm median and the 2.853-second four-model batch median, which used the earlier output stages.

Verification confirmed:

- All five raw and all five recommended PNGs were **byte-identical to the previously approved benchmark outputs** before removing the old experiments.
- All 25 PNGs decode with the correct original dimensions and RGB mode.
- Every source checksum is unchanged.
- Each binary mask contains only 0 and 255; recommended pixels inside it equal original RGB, and pixels outside it equal the prior ink-only RGB exactly.
- All 52 gallery links/image references resolve to existing local files.
- A repeat command skipped all five complete, checksum-verified pages.
- Real installed Tesseract found both dialogue lines in a new synthetic page; no additional model inference was needed for this OCR check.

Exact per-file hashes and metrics are in [final-validation.json](research/final-validation.json). Runtime results stay local in `output/input/`; source artwork and generated PNGs are not published.

## Automated tests

`.venv/bin/python -m unittest discover -s tests -v`: **13 passed, zero skipped**.

Tests cover five-folder output and pixel invariants, unchanged source bytes, verified resume, rerunning only a corrupted page, changed-input rejection, JPEG data with a PNG filename, proportion restoration, PNG-name collisions, wrong reviewed-region checksums, input/output nesting, worker timeout cleanup, continuing after an ordinary page failure, pinned downloads and SHA256 failure, low-disk guards, exact dark ink, antialias text restoration, out-of-bounds regions and OCR background gating.

Synthetic worker tests use a small fake colorizer and mock disk availability; they do not download weights or claim real model quality. The five-page run above is the separate live ONNX inference evidence. No cross-platform run or fresh dependency bootstrap on another machine is claimed.

## Cleanup

The working project is about **0.55 GiB locally**, including selected weights, one runtime, user inputs and the five-variant results, compared with about 20 GiB of experiment assets before cleanup. Only source/documentation/research metadata is committed. Removed items include rejected-model weights/environments, installer caches, reference atlases, source archives, historical galleries, redundant trial outputs and the obsolete multi-model runner. Input files were retained unchanged. The cleanup record is [cleanup.json](research/cleanup.json).
