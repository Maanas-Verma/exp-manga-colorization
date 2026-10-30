# Manga Light Colorizer

Colorize a folder of black-and-white manga with **Manga Light Colorizer (model 3)** and produce five separate output folders. Runs locally using ONNX Runtime on CPU or an optimized **Apple GPU backend through Core ML**; no prompt, colored reference, CUDA, PyTorch or cloud inference is required.

Model 3 was selected after assessing 14 candidates on an Apple M1 Pro with 16 GiB RAM. Its raw colors and shading were preferred, and source-text restoration fixes softened dialogue without replacing brightness across the artwork. See the [research and selection record](docs/research.md), [measured results](docs/research/experiments.json) and [pipeline architecture](docs/architecture.md).

## Run

Requirements: macOS/Linux, Python 3.11 or 3.12, and internet access for the first dependency/model download. The tested machine is an M1 Pro with 16 GiB RAM. The two model files total **300,318,868 bytes (300.3 MB)**; runtime dependencies need additional space. New/unreviewed pages require local English Tesseract OCR:

```sh
# macOS; skip if already installed
brew install tesseract

# Put your images in a folder, then run:
./colorize.sh input

# Any folder name, including spaces:
./colorize.sh "/path/to/My Manga" --output "/path/to/results"

# Faster Apple Silicon GPU mode; use a separate output for existing CPU runs:
./colorize.sh input --backend coreml-gpu --output output/input-gpu

# If your default Python is newer than the tested versions:
PYTHON=python3.11 ./colorize.sh input
```

The script creates `.venv`, installs pinned [requirements](requirements.txt) when needed, and downloads only the selected model's two checksum-verified ONNX files into `.cache/models/`. Valid downloads are reused. Inputs are never modified. Sessions are loaded once and reused sequentially across pages.

The CPU backend remains the default and keeps existing runs compatible. `--backend coreml-gpu` specializes the ONNX graphs and uses Core ML with `CPUAndGPU` compute units. On this M1 Pro, a repeated same-image test improved from **3.05 s to 1.17 s per complete five-output task**, and network inference from **2.38 s to 0.32 s**. First GPU setup took about **46 seconds** to prepare/compile caches; later loads reused them. Cold first-image latency is higher than warm latency. See [measurements, precision and implementation](docs/acceleration.md).

By default, results go to `output/INPUT_FOLDER_NAME/`. Open `report.html` there for the gallery, completion status and links. Every output PNG has the original decoded dimensions.

```text
output/input/
├── 01_raw_model_output/
├── 02_original_text_recommended/
├── 03_ink_only_previous/
├── 04_text_restoration_mask/
├── 05_original_luminance_older_method/
├── report.html
├── metrics.csv
└── manifest.json
```

Each folder contains one PNG per input image. Recommended output starts with ink-only and copies original RGB where the mask is white. Pixels outside the mask remain **exactly equal** to ink-only. OCR locates text; it never rewrites words. The older full-luminance version preserves source brightness across the page and can wash out model shading.

## Text restoration

The five research pages have [reviewed text regions](assets/text-regions.json), matched by **source SHA256**, so renamed copies still work. They cover English dialogue, captions and selected small English effects. Logos, Japanese lettering and oversized artwork effects are excluded. White letter halos are filled to include glyphs; gray artwork between floating words stays colored.

New pages use automatic English Tesseract detection with conservative white-background checks. **Inspect the generated mask:** automatic detection can miss short words, unusual fonts, rotated text and effects. Its coverage is not equivalent to the five reviewed pages. Supply reviewed boxes for a new collection with:

```sh
./colorize.sh "My Manga" --regions-file my-text-regions.json --output results-reviewed
```

Use the format in `assets/text-regions.json`: `images` maps names to `sha256` and a `regions` list, each with a pixel box `[left, top, right, bottom]` in the decoded, EXIF-oriented image. Right/bottom bounds are exclusive. An explicitly supplied file must match every input checksum; mismatches stop before inference.

## Resume and options

```sh
./colorize.sh input                          # skips complete, hash-verified outputs
./colorize.sh input --force                  # reruns a matching collection
./colorize.sh "My Manga" --recursive         # preserves subfolder paths
./colorize.sh input --threads 4 --timeout 1800
./colorize.sh input --min-free-gib 2          # default free-disk reserve

# If you manage the environment yourself:
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python colorize.py input
```

Inputs/settings must match an existing run to resume; use a new `--output` folder after changing images, settings or reviewed regions. Corrupted/missing outputs are regenerated. Supported formats: PNG, JPEG, WebP, BMP, TIFF. Decoding follows actual bytes, including JPEG named `.png`. EXIF orientation and transparency are normalized onto white. TIFF uses the first frame. PNG filename collisions are rejected before processing.

Inference uses a proportionally resized **512 × 512 white-padded canvas**, then restores original dimensions. This is not native-resolution inference. One isolated CPU worker handles the collection. The default timeout is 30 minutes for model loading and each complete image job; a project lock prevents simultaneous runs. Storage guards preserve a configurable **2 GiB free-disk reserve** after estimated writes and cap project runtime/cache/results at 35 GiB. Input/output directories must be separate without nesting.

The reserve is a buffer for other applications, not model storage or RAM usage. The earlier multi-model benchmark used 20 GiB; that floor was removed from the production default. Use `--min-free-gib 20` to retain it or `--min-free-gib 0` to disable the extra reserve (estimated writes and the project storage cap still apply). First GPU compilation also checks 2 GiB of temporary/cache headroom; converted graphs and compiled models are local caches, separate from the original weights. Changing the reserve does not invalidate completed pages. On macOS the script measures immediately free filesystem space; Finder/System Settings can show a larger available figure that includes purgeable data. See [Apple's explanation](https://support.apple.com/guide/disk-utility/get-detailed-information-about-a-disk-dskutl1005/mac).

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic images to cover decoding, proportions, text/ink preservation, output hashes, filename collisions, reviewed regions, timeout handling and selective resume. Actual inference was separately verified on all five research pages; see [validation](docs/validation.md). Final-script timings include all five outputs and validation, and differ from the original benchmark.

To benchmark one of your own images on both backends, including all five variant saves:

```sh
.venv/bin/python benchmark.py "input/1194-005.png"
```

This runs one cold task and three warm repeats per backend, sequentially, and saves results under `.cache/benchmark-runs/`. On systems without Core ML use `--backends cpu`. Extra OCR on unreviewed pages also contributes to task time.

## License and attribution

Pipeline code is GPL-3.0; see [LICENSE](LICENSE) and [third-party attribution](THIRD_PARTY.md). Model weights come from [sharky172/manga-light-colorizer](https://huggingface.co/sharky172/manga-light-colorizer) under the publisher's **CC BY-NC-SA 4.0** license. Weights, input manga pages, generated images, environments and caches are excluded from Git. This public repository contains code, region coordinates, documentation and compact research records.
