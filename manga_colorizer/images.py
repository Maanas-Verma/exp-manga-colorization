# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path: Path):
    with Image.open(path) as opened:
        actual_format = opened.format
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
        result = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return result, actual_format


def letterbox(original, size=512):
    scale = size / max(original.size)
    width, height = (max(1, round(v * scale)) for v in original.size)
    resized = cv2.resize(np.asarray(original), (width, height), interpolation=cv2.INTER_AREA)
    x, y = (size - width) // 2, (size - height) // 2
    canvas = np.full((size, size, 3), 255, np.uint8)
    canvas[y:y + height, x:x + width] = resized
    return Image.fromarray(canvas), (x, y, x + width, y + height)


def restore_size(prediction, box, original_size):
    if not isinstance(prediction, Image.Image):
        prediction = Image.fromarray(prediction)
    return prediction.convert("RGB").crop(box).resize(original_size, Image.Resampling.LANCZOS)


def retain_luminance(original, predicted):
    source = np.asarray(original, dtype=np.float32) / 255
    colors = np.asarray(predicted, dtype=np.float32) / 255
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    predicted_lab = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB)
    predicted_lab[..., 0] = source_lab[..., 0]
    rgb = cv2.cvtColor(predicted_lab, cv2.COLOR_LAB2RGB)
    # Lab chroma at L=0 can clip to colored non-black pixels. Keep source ink exact.
    ink = np.asarray(original.convert("L")) < 18
    rgb[ink] = source[ink]
    return Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8))


def retain_ink(original, predicted, black_threshold=24, edge_threshold=96):
    """Restore source ink; keep predicted colors/shading outside dark source pixels."""
    if original.size != predicted.size:
        raise ValueError('Ink restoration requires matching image dimensions')
    if not 0 <= black_threshold < edge_threshold <= 255:
        raise ValueError('Invalid ink thresholds')
    source = np.asarray(original.convert('RGB'))
    colors = np.asarray(predicted.convert('RGB'), dtype=np.float32)
    gray = np.asarray(original.convert('L'), dtype=np.float32)
    alpha = np.clip((edge_threshold-gray)/(edge_threshold-black_threshold), 0, 1)[..., None]
    # Feathered edges can darken the raw result without lifting its shadows.
    target = np.minimum(colors, source)
    rgb = np.clip(np.round(colors*(1-alpha)+target*alpha), 0, 255).astype(np.uint8)
    rgb[gray <= black_threshold] = source[gray <= black_threshold]
    return Image.fromarray(rgb)


def validate_output(path, expected_size):
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != tuple(expected_size):
            raise ValueError(f"Invalid output: {path}")


def validate_prediction(original, prediction):
    source = np.asarray(original.convert('L'))
    result = np.asarray(prediction.convert('RGB'))
    if source.std() > 10 and np.ptp(result, axis=(0, 1)).max() < 2:
        raise ValueError('Collapsed constant-color prediction for a nonempty page; unusable model output')
