# SPDX-License-Identifier: GPL-3.0-only
"""Pinned Manga Light generator and SAM encoder with CPU/Core ML execution."""
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
    def __init__(self, cache, threads=4, backend='cpu', profile_dir=None):
        import onnxruntime as ort
        if backend not in ('cpu','coreml-gpu'):
            raise ValueError(f'Unknown execution backend: {backend}')
        if backend!='cpu' and 'CoreMLExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('This ONNX Runtime installation does not include CoreMLExecutionProvider; use --backend cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        cache = Path(cache)
        models=cache;optimization=None
        providers=['CPUExecutionProvider']
        self.backend=backend;self.last_timings={}
        if backend!='cpu':
            from .accelerate import FIXED_DIMENSIONS,prepare_coreml_models
            models,optimization=prepare_coreml_models(cache)
            units='CPUAndGPU'
            # Our canvas and SAM feature contract are fixed; specialize dynamic
            # ONNX dimensions so Core ML can compile efficient static graphs.
            for name,size in FIXED_DIMENSIONS.items():
                options.add_free_dimension_override_by_name(name,size)
            compiled=cache.parent/'coreml'/ort.__version__/sha256(ROOT/'assets/model.json')[:16]/'gpu-rewrite-v1'
            compiled.mkdir(parents=True,exist_ok=True)
            providers=[('CoreMLExecutionProvider',{'ModelFormat':'MLProgram','MLComputeUnits':units,
                'RequireStaticInputShapes':'1','EnableOnSubgraphs':'0','ModelCacheDirectory':str(compiled),
                'AllowLowPrecisionAccumulationOnGPU':'0'}),'CPUExecutionProvider']
        if profile_dir:
            profile_dir=Path(profile_dir);profile_dir.mkdir(parents=True,exist_ok=True)
            options.enable_profiling=True
        def session(name):
            if profile_dir:options.profile_file_prefix=str(profile_dir/name)
            result=ort.InferenceSession(str(models/name),options,providers=providers)
            if backend!='cpu' and 'CoreMLExecutionProvider' not in result.get_providers():
                raise RuntimeError(f'Core ML initialization fell back to CPU for {name}; acceleration was not enabled')
            if backend!='cpu':result.disable_fallback()
            return result
        self.generator=session('v6_generator.onnx')
        self.sam=session('v6_sam_encoder.onnx')
        self.execution={'backend':backend,'onnxruntime':ort.__version__,
                        'generator_providers':self.generator.get_providers(),'sam_providers':self.sam.get_providers()}
        if optimization:self.execution['graph_optimization']=optimization['changes']

    def predict(self, canvas):
        gray = np.asarray(canvas.convert('L'), dtype=np.float32)
        sam_gray = cv2.resize(gray/255, (1024,1024), interpolation=cv2.INTER_LINEAR)
        sam_input = np.repeat(sam_gray[None,None], 3, axis=1).astype(np.float32)
        sam_started=time.perf_counter()
        sam_outputs = self.sam.run(None, {self.sam.get_inputs()[0].name:sam_input})
        sam_seconds=time.perf_counter()-sam_started
        if not all(np.isfinite(output).all() for output in sam_outputs):
            raise ValueError('SAM encoder returned non-finite features')
        generator_started=time.perf_counter()
        output = self.generator.run(None, {
            'L_bw': (gray/127.5-1)[None,None].astype(np.float32),
            'sam_level0':sam_outputs[0], 'sam_level1':sam_outputs[1],
            'wd14_embedding':np.zeros((1,1024), np.float32),
        })[0]
        self.last_timings={'sam_seconds':sam_seconds,'generator_seconds':time.perf_counter()-generator_started}
        if output.shape != (1,3,canvas.height,canvas.width) or not np.isfinite(output).all():
            raise ValueError(f'Invalid RGB prediction: {output.shape}')
        rgb = (output[0].transpose(1,2,0)+1)/2
        return Image.fromarray(np.clip(np.round(rgb*255), 0,255).astype(np.uint8))
