# Memory and batch behavior

Measured on **2026-10-30**, M1 Pro with **16 GiB unified memory / 10 CPU cores**, using the existing optimized **Core ML GPU backend**, not MLX. These measurements cover cached model loading and inference at the fixed 512 × 512 canvas on pages up to 1200 × 1753 pixels. First conversion/compilation and arbitrary larger source pages can require more memory.

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

**This is not tensor batching:** the compiled network inputs have batch dimension 1. Simultaneous batch sizes 2/4 and multiple GPU workers have not been benchmarked and have no established RAM peak or speedup here. The 20-page folder result should not be presented as a multi-image tensor benchmark.

The existing warm single-image comparison was about **2.6× faster than CPU**; the guarded folder run achieves about **1.52 s/page including reloads** on reviewed pages. At the same throughput, 44 pages would take about **67 seconds**, as an estimate rather than a measured 44-page result. OCR, larger source images, initial device warm-up, first compilation and other apps can add time. There is no measured additional speedup from increasing tensor batch size. Overlapping CPU OCR/PNG work with GPU inference is a possible future optimization; it is not implemented or assigned a promised speedup.

## Measurement method and boundaries

`memory.py` reads the stable macOS `rusage_info_v4` ABI through `libproc`, using standard-library `ctypes`; no memory-monitoring dependency or developer SDK is required at runtime. `ri_phys_footprint` is current OS-accounted process footprint and `ri_lifetime_max_phys_footprint` is the OS-recorded process peak. The validation observer sampled the parent/children every 60 ms; its combined figure is explicitly a sampled maximum. This does not isolate every shared helper or GPU-driver allocation and is not total machine RAM.

Apple GPUs share system RAM with the CPU; GPU allocations do not form a separate VRAM allowance that can simply be added on top of the machine's 16 GiB. See [Apple's unified-memory description](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

The [measurement record](research/memory-batch-study.json) preserves the stage readings, unguarded comparison and guarded run. Automated tests verify the distinction between RSS and footprint, read-failure handling, threshold/fallback decisions, and worker replacement without rerunning completed pages.
