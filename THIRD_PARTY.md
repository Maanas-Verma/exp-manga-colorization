# Third-party attribution

## Manga Light Colorizer

- Publisher: **sharky172**, https://huggingface.co/sharky172/manga-light-colorizer
- Pinned revision: `2fb022c4ce55632b7671a1df306f63984928e36a`.
- Tensor contract and normalization follow the publisher's `inference.py`; this wrapper adds padding, five variants, text restoration and folder orchestration.
- Publisher inference code: **GNU GPL v3**. This pipeline is distributed under GPL-3.0.
- Model weights: **CC BY-NC-SA 4.0**. Their license remains separate from the code license.
- [Upstream LICENSE](https://huggingface.co/sharky172/manga-light-colorizer/blob/2fb022c4ce55632b7671a1df306f63984928e36a/LICENSE).
- [Weight license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

ONNX files are downloaded directly from the pinned publisher source, verified by size/SHA256, and excluded from Git. `assets/model.json` records their exact provenance.

## Dependencies and research

ONNX Runtime, NumPy, OpenCV, Pillow and Tesseract retain their upstream licenses; dependency source is not vendored. Python versions are pinned in `requirements.txt`.

Model publishers are credited in `docs/research.md` and `docs/research/model-catalog.json`. Input manga pages, character references and generated artwork are local data excluded from Git.
