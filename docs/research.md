# Research and model selection

Decision recorded **2026-10-30**. Selected model: **#3 Manga Light Colorizer**, using its raw palette, source dark ink and selective original-text restoration. This is a decision for the tested manga pages on an **Apple M1 Pro / 16 GiB unified memory**, not a universal ranking of colorizers.

After selection, an optimized Apple GPU execution path was implemented for this same model. The original selection measurements below remain CPU measurements; the separate [acceleration study](acceleration.md) records Core ML results, first-compilation cost and numerical comparisons.

## Why model 3

The user preferred model 3's raw output over the other local results. It gave coherent filled colors, useful shading and better character palettes on the comparison page. The original black-and-white layout stayed recognizable, unlike diffusion editors that regenerated text and facial details. It needed neither a colored character reference nor a prompt. Both ONNX networks total 300.3 MB and run on CPU with a small dependency set.

The initial single-image warm median was **2.662 seconds**, with **3.445 seconds cold completion**, **0.565 seconds model load** and **2.23 GiB process-lifetime peak RSS**. The four-model/five-page batch later measured a model 3 median of **2.853 seconds/page**, 13.900 seconds processing total and **2.51 GiB worker peak RSS**. These are measured results from this host, not predicted requirements or total system memory.

Model 1 was faster and much smaller, but its colors were less preferred. Model 2 gave little visible improvement over model 1 while taking longer. Model 7 produced rich colors but changed text, faces, hair and character palettes; its batch median was 24.651 seconds/page. The user ultimately chose model 3 alone after reviewing the candidates.

## Research scope and method

- Initial single-image sample: `1194-005.png`, 1100 × 1606, including close-ups, action and dialogue.
- One cold task plus three warm tasks per completed single-image model. Warm median uses those three repeated tasks; installation/download/model loading are separate.
- Common 512 × 512 proportional resize with white padding; output cropped and resized back to source dimensions. This limits small-text detail and is not native-resolution inference.
- Final-candidate batch: five pages (`1194-001`, `003`, `004`, `005`, `006`), one pass per page for models 1, 2, 3 and 7; **20/20 jobs completed**. Batch medians compare different pages and are not repeated warm benchmarks.
- CPU inference for direct colorizers; Apple MPS/MLX or experimental CPU/Core ML ports where documented. Device work was synchronized where applicable. RSS and Metal allocations were reported separately, never added.
- Resource limits: 30 minutes per image/load stage, 35 GiB task data, preserve 20 GiB free. Model 9's fallback hit the disk guard; this is not a demonstrated numerical failure of its CPU implementation.
- For reference models, official character artwork and a volume-cover crop formed a derived atlas. Approximate segmentation, semantic hints and Mac adaptations limit comparisons with author examples. These trials do not establish upstream CUDA parity.

The compact [experiments record](research/experiments.json) preserves exact timings, statuses and visual assessments from both benchmark cohorts and the batch. [Model metadata](research/model-catalog.json) retains known revisions, requirements and provenance. Large checkpoint copies, installer caches, reference images, unsuccessful outputs and obsolete benchmark implementations were removed after extracting these records.

## All 14 candidates

Sizes are decimal downloaded weight estimates from the research, unless qualified. They exclude Python environments, temporary compilation, OS caches and runtime memory. A completed trial means the local configured experiment produced usable files; color quality is a visual judgment on this sample.

| # | Model/source | Weights and runtime | Local outcome and decision |
|---|---|---|---|
| 1 | [Manga Colorization v2 INT8](https://huggingface.co/lookx2/manga-colorization) | 32.5 MB; ONNX CPU, no reference | Completed; warm 1.122 s. Fast vivid fills, softened lettering, some bleed and character palette guesses. Retained for five-page comparison, then dropped when #3 was chosen. |
| 2 | [v2 FP16 ONNX](https://huggingface.co/Faridzar/manga-colorization-v2-onnx) | 61.7 MB; ONNX CPU, no reference | Completed; warm 2.305 s. Nearly the same colors/structure as INT8, little visible gain. Completed five-page comparison, then dropped. |
| 3 | [Manga Light Colorizer](https://huggingface.co/sharky172/manga-light-colorizer) | **300.3 MB**, generator + real SAM encoder; ONNX CPU | Completed; warm 2.662 s. Preferred coherent palette and complete fills; **selected**. Text softness handled separately. |
| 4 | [DDColor Tiny](https://huggingface.co/piddnad/ddcolor_paper_tiny) | 220.4 MB; PyTorch CPU | Completed; warm 1.136 s. Mostly white interiors with sparse yellow/orange edge coloring. User dropped it. General-image training was a weak fit for this manga sample. |
| 5 | [DDColor Artistic](https://huggingface.co/piddnad/ddcolor_artistic) | 911.9 MB; PyTorch CPU | Completed; warm 2.469 s. Sparse green/orange contours and limited filled color. User dropped it. |
| 6 | [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) + [Anime Lineart ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime) | About 3.46 GB selected FP16 components; PyTorch MPS/Diffusers | Completed; warm 19.358 s. Mostly monochrome, small magenta accents, distorted dialogue. Tested control scale 1.0, img2img strength 0.65, inverted existing line art. User rejected the useful-color result at these settings. |
| 7 | [FLUX.2 Klein 4B MLX 4-bit](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit) | About 4.61 GB; MLX/MFLUX, float16 compute, low-memory mode, four steps | Completed; warm 23.693 s. Rich colors but regenerated text/details and incorrect character colors. Five-page batch completed; #3 was preferred. Stage reloads were included in image time. |
| 8 | [FLUX 3 Image Klein MLX 6-bit](https://huggingface.co/VisualInference/FLUX-3-Image-Klein-MLX-6bit) | About 9.01 GB; `flux3-mlx`, 16 steps | Completed; warm 158.444 s, MLX peak 9.44 GB. Flat beige/red colors, misplaced accents, regenerated details. User rejected it. Publisher M4 Pro memory claims were not treated as M1 measurements. |
| 9 | [MangaNinja Core ML](https://huggingface.co/VioletXF/manganinja-coreml) | About 4.94 GB plus compilation; seven components, colored reference | **Invalid GPU output**: all raw trials black. FP16 reference features/conditional residuals became non-finite. Early CPU diagnostics stayed finite; a CPU/Neural Engine fallback stopped at the 20 GiB disk floor. No usable completed local color output; dropped. |
| 10 | [BasicPBC-Light](https://github.com/ykdai/BasicPBC) | 50.4 MB core + 21.1 MB RAFT asset; PyTorch CPU, reference/segmentation preparation | Completed; warm 2.897 s. Flat region fills, missed/merged regions, dark dialogue. Native scatter compatibility work and derived reference preparation limit the trial. Retained CLIP parameters were covered by strict checkpoint loading, so no separate 1.4 GB CLIP checkpoint was required. User dropped it. |
| 11 | [BasicPBC-Ref](https://github.com/ykdai/BasicPBC) | 90.6 MB core; PyTorch CPU, reference colors/line art/labels/semantic hints | Completed; warm 2.616 s. Sparse flat fills and incorrect matches. Coarse semantic hints were specific to the test page and were not a general automatic solution. User dropped it. |
| 12 | [DACoN v1.1](https://github.com/kzmngt/DACoN) | About 1.36 GB including DINOv2-Large; PyTorch CPU, closed line regions, reference colors | Completed; warm 8.303 s. Patchy orange/black/purple regions and obscured dialogue. Reference/segmentation preparation limits this full-page experiment. User dropped it. |
| 13 | [ColorizeDiffusion v1.5](https://huggingface.co/tellurion/ColorizeDiffusion) | **8.28 GB total**: 4.33 GB switch checkpoint + required 3.94 GB OpenCLIP; experimental PyTorch MPS port | Completed; warm 26.781 s. Vivid but large green/yellow fills, distorted text/faces and inconsistent palettes. MPS parity with author's CUDA path unverified. User rejected it. Initial 4.33 GB estimate omitted its required backbone and was corrected. |
| 14 | [MangaDiT](https://github.com/CyberAgentAILab/MangaDiT) | About 33.74 GB standard stack; transformer alone 23.80 GB; colored reference | **Blocked at preflight**, not benchmarked. Standard transformer exceeded 16 GiB physical RAM, and the custom attention path had no locally validated compatible quantization. Published A100 80 GB setup was not evidence of a working 16 GiB Mac route. |

## Selected-model provenance

Source revision: `2fb022c4ce55632b7671a1df306f63984928e36a` from `sharky172/manga-light-colorizer`.

| File | Bytes | SHA256 |
|---|---:|---|
| `v6_generator.onnx` | 191,335,312 | `48284fcf0b7a606270702630f559af88eecf95bc6cdec1ff8bce8663d12b4bb6` |
| `v6_sam_encoder.onnx` | 108,983,556 | `97c4cad5814e1fb12c13d1b23d3969dd9e2fce92d818539fe7db575140333a34` |

See [model manifest](../assets/model.json), [original environment lock](research/original-onnx-requirements.txt), [upstream model card](https://huggingface.co/sharky172/manga-light-colorizer), [inference contract](https://huggingface.co/sharky172/manga-light-colorizer/blob/2fb022c4ce55632b7671a1df306f63984928e36a/inference.py), and [license declaration](https://huggingface.co/sharky172/manga-light-colorizer/blob/2fb022c4ce55632b7671a1df306f63984928e36a/LICENSE). The selected weights remain CC BY-NC-SA 4.0; the publisher's inference code is GPL-3.0.

## Why the output method changed

1. **Raw output:** preferred color and shading, but page reduction blurred small lettering.
2. **Whole-page source luminance:** replaced predicted CIELAB L with original L. Source text edges became clear, but colored shading became pale and less attractive. Correcting text this way changed brightness everywhere.
3. **Ink only:** copied the darkest source pixels and feathered edges. It retained raw shading outside source ink, but gray antialias edges and blurred backgrounds around letters remained. Hence text was less clear than in whole-page luminance output.
4. **Selective original text:** copied source RGB in reviewed dialogue/caption masks while keeping ink-only pixels everywhere else. Both dark glyphs and their gray/white edges were restored, without changing the page's general palette or shading.

Automatic bubble filling was also tested. Incomplete outlines let white regions extend into hair/faces. The final five-page masks use reviewed boxes, source-white support and filled letter halos. New-page OCR remains a convenience that requires mask inspection. The source words are copied, never reconstructed from OCR strings.

All five recommended outputs were checked pixelwise: inside masks they equal source RGB; outside masks they equal ink-only RGB. Input checksums and raw/ink/luminance baselines remained unchanged. The final script reran all five pages and reproduced the approved raw and recommended PNGs byte for byte.

## Limits and future work

The research used a small page set, not a labeled colorization dataset. There is no ground-truth color score or claim of canonical palette accuracy. Visual preference is the user's review; colorfulness/ink diagnostics are not perceptual quality metrics. Reference-model results depend on preparation, and experimental Mac adaptations were not validated against upstream CUDA outputs. Unsupported local conversions are not judgments on upstream models in general.

The 512-pixel inference canvas still limits artwork detail. Source-text copying improves selected text but does not undo every model change. New characters may receive plausible but incorrect colors. OCR can miss text or select a wrong region. Higher-resolution or tiled inference, learned bubble detection, and canonical-color references remain separate research directions; none are claimed implemented here.
