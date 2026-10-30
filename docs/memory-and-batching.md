# Memory and batch behavior

Measured on **2026-10-30**, M1 Pro with **16 GiB unified memory / 10 CPU cores**, using the existing optimized **Core ML GPU backend**, not MLX. These measurements cover cached model loading and inference at the fixed 512 × 512 canvas. The original memory study used pages up to 1200 × 1753 pixels; the tensor-batch study below used a separate set of your pages. First conversion/compilation and arbitrary larger source pages can require more memory.

## RSS was not the full memory figure

Earlier results recorded about 0.88 GiB peak worker RSS. That measures resident process pages and did not establish the full GPU workload footprint. macOS `proc_pid_rusage` physical-footprint accounting revealed substantially higher use.

An unguarded 15-page test loaded at about **0.37 GiB worker physical footprint**, then rose to **5.23 GiB peak** after 15 pages. Its peak RSS was only about **0.63 GiB** in that particular run. Different runs can have different RSS because resident/file-backed/compressed memory differs; RSS and physical footprint must not be treated as interchangeable or added together.

Repeated network-only predictions also grew in footprint. Additional autorelease pools did not stop it, and deleting the sessions in the same process released only part of it. These observations locate retention in the inference/runtime path but do not identify a specific upstream defect. Exiting the worker releases its process-owned allocations, so the folder runner now recycles GPU workers between pages.

## Guarded 20-page run

The production CLI processed 20 pages, reusing the five original test pages four times under distinct input names, with all five output variants. Cached loading and two worker reloads are included in wall time.

| Measurement | Observed |
|---|---:|
| Worker physical footprint immediately after loading | About 0.36–0.38 GiB |
| Configured recycling threshold | 3 GiB |
| Maximum OS-recorded worker physical footprint | 3.21 GiB |
| Maximum worker RSS | 1.11 GiB |
| Maximum sampled parent + child physical footprint | 3.28 GiB |
| Complete pages / output PNGs | 20 / 100 |
| Worker loads | Three: initial load and two reloads |
| Total wall time, including reloads | 30.42 s |
| Average wall time per page | 1.52 s |
| Throughput | About 39 pages/minute |

This is an **observed peak**, not a guaranteed maximum for all inputs. The 3 GiB threshold is checked after a page, so a page can overshoot it. It also excludes memory attributed to shared Core ML services or the rest of macOS. Allowing roughly 4 GiB for this workload at these page sizes is a planning allowance, not a universal requirement or a process hard limit.

## Current batch support

Folder batching keeps one model worker active and processes images sequentially, replacing the worker when its footprint reaches the threshold. When physical accounting is unavailable, it falls back to recycling after eight images. Parent run metadata and `metrics.csv` now include current RSS, physical footprint, and lifetime peak physical footprint; model-load memory and reload durations are recorded in `worker_loads`.

The repeat-image benchmark applies the same 3 GiB policy for longer trial counts. The first image after a worker reload is marked cold and excluded from its warm median; model reload costs remain recorded separately.

```sh
./colorize.sh "0" --backend coreml-gpu --output output/0-gpu

# A different soft per-worker threshold; this does not invalidate completed pages:
./colorize.sh "0" --backend coreml-gpu --output output/0-gpu --gpu-worker-memory-gib 3
```

Changing the threshold affects operational behavior, not model weights or output geometry. CPU behavior and legacy CPU resume remain unchanged. GPU result hashes/masks are validated across reloads just as with an uninterrupted worker.

## Real tensor batches 1, 2 and 4

The experimental benchmark now compiles separate static ONNX/Core ML graphs for **N = 1, 2, 4**, then submits each group in one SAM call and one generator call. Profiling confirmed two Core ML execution sections and no ONNX CPU nodes per prediction. The production folder command keeps N = 1.

Two runs processed the same **24 distinct pages from `0/`** in orders **1 → 2 → 4** and **4 → 2 → 1**. Each configuration therefore completed 48 pages. Source dimensions ranged from 728 × 1023 to a 1456 × 1017 spread; all images used the same padded 512² generator and 1024² SAM contract. Every page ran automatic English OCR and saved all five PNG variants at its original dimensions.

| Tensor batch | Complete workflow pages/min | Individual-run range | Prediction stage pages/min | Peak worker footprint | Peak sampled process-tree footprint |
|---|---:|---:|---:|---:|---:|
| **1** | **37.84** | 35.13–40.99 | 143.19 | 3.44 GiB | **3.47 GiB** |
| **2** | **31.73** | 29.28–34.63 | 90.03 | 4.28 GiB | **4.33 GiB** |
| **4** | **26.60** | 21.94–33.77 | 68.61 | 4.78 GiB | **4.82 GiB** |

The combined rates use `48 × 60 / (run_1_seconds + run_2_seconds)`. Complete-workflow time includes initial cached worker loading, reading, canvas preparation, prediction, OCR, ink/text/luminance processing, all five saves, normal output validation and worker reloads. Folder discovery, controller imports, graph preparation, first compilation/profiling probes, gallery/CSV generation, and final cross-configuration pixel comparisons are excluded. Prediction-stage time includes both networks, tensor preparation, finite checks and conversion to RGB; it excludes OCR and the five output saves.

Workers loaded at **0.36–0.42 GiB physical footprint** before prediction. During processing, the same **3 GiB soft recycling policy** was applied between whole groups. Across the two 24-page runs, N = 1 needed 3 and 4 worker loads, N = 2 needed 4 and 4, and N = 4 needed 6 and 6. Every N = 4 group exceeded the soft threshold and required a new worker before the next group. Setup/probe peaks remained below 4.14 GiB sampled process-tree footprint across both runs; the larger 4.82 GiB peak occurred during measured processing. An experimental 8 GiB worker abort guard was never reached.

**Keep tensor batch 1 for this system and pipeline.** Batch 2 averaged about **16% fewer pages/minute** and batch 4 about **30% fewer**, while both used more memory. Prediction throughput itself also declined. These results describe the current Core ML graphs with the current recycling policy; a different runtime or policy requires a new measurement. One N = 4 group took 21 seconds in the first run, versus roughly 6–9 seconds for most groups. The reverse-order repeat is included, and the range shows the observed variability.

All **144 page jobs / 720 saved PNGs** passed decoding, geometry and source-hash checks. Batched raw output differed from batch 1 by at most **1/255 per channel**, with maximum mean absolute channel error per page below 0.00004 on the 0–255 scale. All text masks were identical across batch sizes. Restored text pixels matched the source exactly, and pixels outside the mask matched ink-only exactly. A separate production-CLI regression on the five reviewed research pages produced **25 PNGs byte-identical** to the previous GPU reference.

Reproduce the experiment:

```sh
# Existing environment and Tesseract required; creates five variants per configuration:
.venv/bin/python benchmark_batches.py "0" --pages 24

# Repeat in reverse order using compiled caches:
.venv/bin/python benchmark_batches.py "0" --pages 24 --batches 4 2 1
```

Results and private image outputs go to `.cache/tensor-batch-runs/<timestamp>/`. The project lock prevents concurrent model jobs. The benchmark checks storage headroom, monitors physical footprint, recycles at 3 GiB between groups, and aborts a worker above 8 GiB or if immediately free disk falls below 2 GiB. Its process-tree samples include the controller, model worker and transient OCR children; the OS lifetime peak is used for the worker figure. Shared Core ML services and other macOS applications remain outside these measurements, so the table is an observed workload peak rather than a guaranteed maximum RAM requirement.

The public [tensor-batch measurement record](research/tensor-batches.json) contains both run orders, input hashes/dimensions, exact model input shapes, GPU placement, loads, group timings, memory readings and validation results. It contains no source images, generated artwork or model weights.

## Measurement method and boundaries

`memory.py` reads the stable macOS `rusage_info_v4` ABI through `libproc`, using standard-library `ctypes`; no memory-monitoring dependency or developer SDK is required at runtime. `ri_phys_footprint` is current OS-accounted process footprint and `ri_lifetime_max_phys_footprint` is the OS-recorded process peak. The validation observer sampled the parent/children every 60 ms; its combined figure is explicitly a sampled maximum. This does not isolate every shared helper or GPU-driver allocation and is not total machine RAM.

Apple GPUs share system RAM with the CPU; GPU allocations do not form a separate VRAM allowance that can simply be added on top of the machine's 16 GiB. See [Apple's unified-memory description](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

The [measurement record](research/memory-batch-study.json) preserves the stage readings, unguarded comparison and guarded run. Automated tests verify the distinction between RSS and footprint, read-failure handling, threshold/fallback decisions, and worker replacement without rerunning completed pages.
