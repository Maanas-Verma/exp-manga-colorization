# SPDX-License-Identifier: GPL-3.0-only
"""Pinned Manga Light generator and SAM encoder, using CPU ONNX Runtime."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure_models(cache, check_storage):
    manifest = json.loads((ROOT/'assets/model.json').read_text())
    cache = Path(cache);cache.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        path = cache/Path(item['path']).name
        if path.exists() and path.stat().st_size == item['size'] and sha256(path) == item['sha256']:
            continue
        check_storage(item['size'])
        url = f'https://huggingface.co/{manifest["repo"]}/resolve/{manifest["revision"]}/{item["path"]}'
        print(f'Downloading {path.name} ({item["size"]/1e6:.1f} MB)', flush=True)
        temporary = path.with_suffix('.onnx.partial')
        started = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024**2):
                    if time.monotonic()-started > 1800:
                        raise TimeoutError('Model download exceeded 30 minutes')
                    check_storage(len(chunk));stream.write(chunk)
            if temporary.stat().st_size != item['size'] or sha256(temporary) != item['sha256']:
                raise ValueError(f'Model checksum mismatch: {path.name}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


class MangaLight:
    def __init__(self, cache, threads=4):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        cache = Path(cache)
        self.generator = ort.InferenceSession(str(cache/'v6_generator.onnx'), options, providers=['CPUExecutionProvider'])
        self.sam = ort.InferenceSession(str(cache/'v6_sam_encoder.onnx'), options, providers=['CPUExecutionProvider'])

    def predict(self, canvas):
        gray = np.asarray(canvas.convert('L'), dtype=np.float32)
        sam_gray = cv2.resize(gray/255, (1024,1024), interpolation=cv2.INTER_LINEAR)
        sam_input = np.repeat(sam_gray[None,None], 3, axis=1).astype(np.float32)
        sam_outputs = self.sam.run(None, {self.sam.get_inputs()[0].name:sam_input})
        if not all(np.isfinite(output).all() for output in sam_outputs):
            raise ValueError('SAM encoder returned non-finite features')
        output = self.generator.run(None, {
            'L_bw': (gray/127.5-1)[None,None].astype(np.float32),
            'sam_level0':sam_outputs[0], 'sam_level1':sam_outputs[1],
            'wd14_embedding':np.zeros((1,1024), np.float32),
        })[0]
        if output.shape != (1,3,canvas.height,canvas.width) or not np.isfinite(output).all():
            raise ValueError(f'Invalid RGB prediction: {output.shape}')
        rgb = (output[0].transpose(1,2,0)+1)/2
        return Image.fromarray(np.clip(np.round(rgb*255), 0,255).astype(np.uint8))
