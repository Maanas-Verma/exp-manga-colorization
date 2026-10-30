# Apple GPU acceleration for model 3

Implemented and tested on **2026-10-30**, Apple M1 Pro / 16 GiB, ONNX Runtime 1.31.1. The same model weights now support an optional optimized Apple GPU backend. CPU remains the default so existing runs keep their backend and resume behavior.

```sh
./colorize.sh "0" --backend coreml-gpu --output output/0-gpu
```

Use a separate output folder when changing backend. GPU/CPU outputs are very close but not byte-identical, so mixing them under one run identity would make checksums and measurements misleading. Historical runs without a backend field are treated as CPU.

## Measured results

One cold task and three warm repeats on `1194-005.png`, sequentially, with identical 512-pixel preprocessing, reviewed masks and all five variant saves/validation:

| Execution | Initial model load/setup | First image task | Warm median, full task | Warm median, inference |
|---|---:|---:|---:|---:|
| CPU baseline | 1.90 s | 3.80 s | **3.05 s** | **2.38 s** |
| Optimized Core ML GPU, fresh cache | 46.12 s | 3.42 s | **1.17 s** | **0.32 s** |

This is about **2.6× faster per warm full task** and **7.4× faster for warm inference** in this comparison. Task times include image decode, preparation, all variants, saving and validation; model load/setup and parent gallery rendering are separate. Source resolutions, text restoration and the five output folders are unchanged.

The finalized CLI also ran all five original pages: **2.05, 1.08, 1.20, 1.05, 1.06 seconds**, with cached model loading at **1.63 seconds**. All 25 output PNGs and source hashes were validated.

An additional unreviewed page, including both Tesseract passes, measured **3.43 s CPU versus 1.39 s GPU** for the warm full-task median. Its CPU/GPU text masks were identical. OCR and PNG output still run on CPU; accelerating the network does not accelerate those stages.

**Cold latency matters.** First GPU setup converts/compiles models and took about 46 seconds here. Core ML can also defer device initialization until the first prediction: the unreviewed-page test observed a **15.64-second first GPU image**, despite later images being much faster. A fresh one-image invocation is therefore not guaranteed to beat CPU. This mode is most useful for multiple pages or repeated predictions in the existing worker. Warm timings are not a promise of first-launch latency, and vary with other apps and system load.

See [exact measurements](research/apple-gpu-benchmark.json). These are local exploratory measurements, not a cross-device benchmark.

## Why enabling Core ML alone was insufficient

The installed ONNX Runtime already included `CoreMLExecutionProvider`. A simple provider switch could use most operators, but the original SAM graph split into **19 Core ML partitions** separated by ONNX CPU reshape/transpose operations. The generator split into **two partitions** around six unsupported `Abs` operators. These boundaries caused repeated device handoffs; the automatic Core ML device mode did not outperform CPU in the initial same-image trial.

Profiling identified 36 reshapes and 18 transposes in the SAM window partition/unpartition paths. They use six-dimensional intermediate tensors. The final preparation folds static shape calculations and rewrites each reshape–transpose–reshape triple:

```text
Original reshape: [a,b,c,d,e,f]
Transpose:        [0,1,3,2,4,5]
Final reshape:    unchanged target

Equivalent reshape: [a*b,c,d,e*f]
Transpose:          [0,2,1,3]
Final reshape:      unchanged target
```

Grouping adjacent dimensions preserves element order and gives the same result without rank-six tensors. All **18 window patterns** are rewritten. The six generator `Abs(x)` nodes become `max(x, x * -1)`, equivalent for the finite float32 values used by these models.

The optimized models profiled as **one Core ML section per network**, with no intervening ONNX CPU node events. Unit tests execute the original and rewritten synthetic graphs through ONNX CPU and check exact equality, including an inferred `-1` dimension. Unexpected patterns or branches are rejected rather than rewritten speculatively.

## Runtime and cache design

- Original downloads remain pinned/checksum-verified in `.cache/models/` and are never modified.
- `accelerate.py` prepares local static ONNX copies in `.cache/coreml-onnx/`, keyed by source manifest and rewrite version.
- Input dimensions are fixed to this pipeline's contract: batch 1, 512 × 512 generator canvas, 1024 × 1024 SAM input, and 64²/32² SAM features.
- Core ML uses `MLProgram`, `CPUAndGPU`, static input shapes, and compiled caching under `.cache/coreml/`. Lower-precision GPU accumulation is disabled.
- `CPUAndGPU` allows Apple's runtime to place operations on CPU/GPU; it does not mean every instruction is guaranteed to run on the GPU. The profiler confirms Core ML execution rather than ONNX CPU fallback. Neural Engine is not selected by this mode.
- Missing Core ML support and provider initialization failure produce an explicit error. Python's automatic provider retry is disabled for GPU sessions so a runtime GPU error cannot silently switch the session to CPU. CPU is available explicitly with `--backend cpu`.
- Converted/compiled caches require additional disk space beyond the original 300.3 MB weights. The first uncached GPU load checks 2 GiB of cache headroom plus the configured free-space reserve. Cache artifacts and test images are excluded from Git; discarded experimental caches were removed.

The runtime adds `onnx`, `ml-dtypes` and `typing-extensions` for preparing and validating the converted graphs. It does not install PyTorch, MLX, coremltools, a new model checkpoint or a cloud service. This implementation retains ONNX and uses the [official Core ML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

## Color and text validation

Across the five original pages, raw CPU/GPU output differed by at most **2/255 per channel**. Mean absolute channel differences were about **0.002–0.004/255**, with no meaningful palette change visible at that scale. Minor floating-point rounding means GPU PNG hashes differ from CPU PNG hashes.

Every recommended image was checked: masked pixels equal original source RGB exactly, and pixels outside the mask equal that GPU run's ink-only image exactly. Source images remained unchanged. Worker RSS does not include all Core ML helper/device allocations, so these measurements do not establish a total system/GPU peak memory requirement.

## Repeat on another image

```sh
.venv/bin/python benchmark.py "path/to/page.jpeg"
```

The benchmark holds the same project lock, runs backends sequentially, uses the same complete image function, releases each worker, and writes timings/results under `.cache/benchmark-runs/`. Model loading is separate from task timing. The default is four trials per backend; on a machine without Core ML use `--backends cpu`.
