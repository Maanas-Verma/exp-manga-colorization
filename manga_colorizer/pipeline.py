# SPDX-License-Identifier: GPL-3.0-only
"""Folder orchestration, isolated CPU worker, verified outputs and local gallery."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import html
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from urllib.parse import quote

import numpy as np
from PIL import Image

from .images import letterbox, load_image, restore_size, retain_ink, retain_luminance, validate_output, validate_prediction
from .model import ROOT, MangaLight, ensure_models, sha256
from .text import check_ocr, restore_text, reviewed_regions

VERSION = '1.0.0'
VARIANTS = {
    'raw':('01_raw_model_output','Raw model output'),
    'text':('02_original_text_recommended','Original text (recommended)'),
    'ink':('03_ink_only_previous','Ink only (previous)'),
    'mask':('04_text_restoration_mask','Text restoration mask'),
    'luminance':('05_original_luminance_older_method','Original luminance (older method)'),
}
EXTENSIONS = {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}


def write_json(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def check_storage(output,additional=0):
    locations = [ROOT/'.cache',ROOT/'.venv',ROOT/'output']
    if not output.is_relative_to(ROOT/'output'):locations.append(output)
    used = sum(path.stat().st_size for directory in locations if directory.exists()
               for path in directory.rglob('*') if path.is_file() and not path.is_symlink())
    if used+additional > 35*1024**3:
        raise RuntimeError('Project runtime/cache/results would exceed the 35 GiB limit')
    if shutil.disk_usage(output if output.exists() else output.parent).free-additional < 20*1024**3:
        raise RuntimeError('This operation must preserve at least 20 GiB free disk space')


def discover(input_dir,recursive=False):
    paths = sorted(path for path in (input_dir.rglob('*') if recursive else input_dir.iterdir())
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:raise ValueError('Input folder contains no supported images')
    images=[];names=set()
    for path in paths:
        relative=path.relative_to(input_dir);name=relative.with_suffix('.png').as_posix()
        if name.casefold() in names:
            raise ValueError(f'Input filenames would collide as PNG outputs: {relative}. Rename one source file.')
        names.add(name.casefold());decoded,format_name=load_image(path)
        images.append({'name':relative.as_posix(),'output_name':name,'path':str(path),
                       'sha256':sha256(path),'size':list(decoded.size),'format':format_name})
    return images


def save_png(image,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.png.tmp')
    try:
        image.convert('RGB').save(temporary,format='PNG')
        validate_output(temporary,image.size);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def process_image(model,job):
    started=time.perf_counter();source=Path(job['path'])
    if sha256(source)!=job['sha256']:raise ValueError('Input changed before processing')
    original,format_name=load_image(source)
    canvas,box=letterbox(original,512)
    inference_started=time.perf_counter();prediction=model.predict(canvas)
    inference_seconds=time.perf_counter()-inference_started
    validate_prediction(canvas,prediction)
    raw=restore_size(prediction,box,original.size);ink=retain_ink(original,raw)
    restored,mask,text_info=restore_text(original,ink,raw,job['regions'],job['language'])
    variants={'raw':raw,'ink':ink,'text':Image.fromarray(restored),
              'mask':Image.fromarray(mask).convert('RGB'),'luminance':retain_luminance(original,raw)}
    artifacts={}
    for key,image in variants.items():
        relative=Path(VARIANTS[key][0])/job['output_name'];destination=Path(job['output'])/relative
        save_png(image,destination)
        artifacts[key]={'path':relative.as_posix(),'sha256':sha256(destination)}
    if sha256(source)!=job['sha256']:raise ValueError('Input changed during processing')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system()!='Darwin':rss*=1024
    return {'status':'complete','input_sha256':job['sha256'],'original_size':list(original.size),
            'input_format':format_name,'inference_canvas':[512,512],'content_box':list(box),
            'inference_seconds':inference_seconds,'processing_seconds':time.perf_counter()-started,
            'worker_lifetime_peak_rss_bytes':int(rss),'text_restoration':text_info,'artifacts':artifacts}


def worker(connection,cache,threads):
    try:
        started=time.perf_counter();model=MangaLight(cache,threads)
        connection.send({'status':'loaded','model_load_seconds':time.perf_counter()-started})
        while True:
            job=connection.recv()
            if job is None:break
            try:connection.send(process_image(model,job))
            except Exception as error:connection.send({'status':'failed','error':f'{type(error).__name__}: {error}'})
    except (EOFError,BrokenPipeError):pass
    except Exception as error:
        connection.send({'status':'failed','error':f'{type(error).__name__}: {error}'})
    finally:connection.close()


class Worker:
    def __init__(self,cache,threads,timeout):
        context=multiprocessing.get_context('spawn')
        self.connection,child=context.Pipe();self.timeout=timeout
        self.process=context.Process(target=worker,args=(child,str(cache),threads),daemon=True)
        self.process.start();child.close()

    def receive(self):
        if not self.connection.poll(self.timeout):
            self.close();raise TimeoutError(f'Worker exceeded {self.timeout:g} seconds')
        try:return self.connection.recv()
        except EOFError as error:raise RuntimeError('Model worker ended unexpectedly') from error

    def close(self):
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=5)
            if self.process.is_alive():self.process.kill();self.process.join(timeout=5)
        self.connection.close()


def valid_result(result,image,output):
    try:
        if result['status']!='complete' or result['input_sha256']!=image['sha256']:return False
        for key in VARIANTS:
            artifact=result['artifacts'][key];path=output/artifact['path']
            validate_output(path,image['size'])
            if sha256(path)!=artifact['sha256']:return False
        return True
    except (KeyError,ValueError,OSError):return False


def report(output,manifest):
    results=manifest['results'];images=manifest['images']
    complete=sum(result.get('status')=='complete' for result in results.values())
    records=io.StringIO();writer=csv.writer(records)
    writer.writerow(['image','status','processing_seconds','inference_seconds','text_method','restored_fraction','error'])
    rows=[]
    for image in images:
        result=results.get(image['name'],{'status':'pending'});status=result['status']
        writer.writerow([image['name'],status,result.get('processing_seconds'),result.get('inference_seconds'),
                         result.get('text_restoration',{}).get('method'),result.get('text_restoration',{}).get('restored_fraction'),result.get('error','')])
        cells=[]
        for key,(_,label) in VARIANTS.items():
            if status=='complete':
                url=quote(result['artifacts'][key]['path'],safe='/')
                cells.append(f'<td><a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(label)}"></a></td>')
            else:cells.append(f'<td>{html.escape(status)}</td>')
        method=result.get('text_restoration',{}).get('method','')
        rows.append(f'<tr><th>{html.escape(image["name"])}<br>{html.escape(method)}<br>{html.escape(result.get("error",""))}</th>{"".join(cells)}</tr>')
    (output/'metrics.csv').write_text(records.getvalue())
    headers=''.join(f'<th>{html.escape(label)}</th>' for _,label in VARIANTS.values())
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Manga Light Colorizer results</title>
<style>:root{{font:16px system-ui;color-scheme:light dark}}body{{margin:24px}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}th,td{{padding:8px;vertical-align:top;border-bottom:1px solid #8885}}th:first-child{{width:130px;overflow-wrap:anywhere}}img{{width:100%;height:auto;background:white}}p{{line-height:1.5}}.scroll{{overflow:auto}}@media(max-width:900px){{table{{min-width:1100px}}}}</style>
<h1>Manga Light Colorizer · {complete}/{len(images)} complete</h1>
<p>Five outputs per page. The recommended version adds original text to the ink-only image. The mask is binary: white pixels come from the source, black pixels keep the ink-only result. Full luminance replacement is the older comparison and can wash out the preferred shading.</p>
<p>Inference uses a 512 × 512 padded canvas; PNGs recover original dimensions. Reviewed masks are used for known source checksums. Automatic English OCR on new pages can miss text; inspect the mask. Logos, Japanese lettering and oversized effects are outside the reviewed masks.</p>
<p><a href="manifest.json">Run details and checksums</a> · <a href="metrics.csv">Per-image metrics</a>. Timings include decode, inference, all five variants, PNG saves and validation; model loading is separate.</p>
<div class="scroll"><table><thead><tr><th>Image</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></html>'''
    temporary=output/'report.html.tmp';temporary.write_text(content);temporary.replace(output/'report.html')
    write_json(output/'manifest.json',manifest)


def run(args):
    input_dir=Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():raise ValueError('Provide an existing input folder')
    output=(Path(args.output).expanduser() if args.output else ROOT/'output'/input_dir.name).resolve()
    if output.is_relative_to(input_dir) or input_dir.is_relative_to(output):
        raise ValueError('Input and output folders must be separate, without nesting')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'.cache/models';cache.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.cache/run.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another colorization is already running in this project') from error
        images=discover(input_dir,args.recursive)
        region_file=Path(args.regions_file).resolve() if args.regions_file else ROOT/'assets/text-regions.json'
        known=reviewed_regions(region_file)
        if args.regions_file and any(image['sha256'] not in known for image in images):
            raise ValueError('Explicit reviewed-region file does not match every input checksum')
        if any(image['sha256'] not in known for image in images):check_ocr(args.ocr_language)
        settings={'version':VERSION,'input_dir':str(input_dir),'size':512,'threads':args.threads,
                  'regions_sha256':sha256(region_file),'ocr_language':args.ocr_language,
                  'model_manifest_sha256':sha256(ROOT/'assets/model.json')}
        path=output/'manifest.json'
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest['settings']!=settings or manifest['images']!=images:
                raise ValueError('Inputs or pipeline settings changed. Choose a new --output folder.')
        else:
            if output.exists() and any(output.iterdir()):raise ValueError('Output folder is not an existing pipeline run; choose an empty folder')
            output.mkdir(parents=True,exist_ok=True)
            manifest={'settings':settings,'images':images,'results':{},'created_at':datetime.now(timezone.utc).isoformat()}
        for folder,_ in VARIANTS.values():(output/folder).mkdir(exist_ok=True)
        report(output,manifest)
        pending=[image for image in images if args.force or not valid_result(manifest['results'].get(image['name'],{}),image,output)]
        if not pending:
            print(f'All {len(images)} pages already valid. Report: {output/"report.html"}');return 0
        check_storage(output)
        manifest['model']=ensure_models(cache,lambda additional=0:check_storage(output,additional))
        manifest['dependencies']={name:importlib.metadata.version(name) for name in ['numpy','pillow','opencv-python-headless','onnxruntime']}
        manifest['python']=platform.python_version()
        client=Worker(cache,args.threads,args.timeout)
        try:
            loaded=client.receive()
            if loaded['status']!='loaded':raise RuntimeError(loaded['error'])
            manifest['model_load_seconds']=loaded['model_load_seconds']
            for index,image in enumerate(images,1):
                if image not in pending:
                    print(f'[{index}/{len(images)}] skipped verified {image["name"]}',flush=True);continue
                # Reserve uncompressed space for all variants before processing this page.
                check_storage(output,image['size'][0]*image['size'][1]*15)
                manifest['results'][image['name']]={'status':'running'};report(output,manifest)
                started=time.perf_counter()
                client.connection.send({**image,'output':str(output),'regions':known.get(image['sha256']),'language':args.ocr_language})
                try:result=client.receive()
                except (TimeoutError,RuntimeError) as error:
                    manifest['results'][image['name']]={'status':'failed','error':str(error)};report(output,manifest);raise
                result['elapsed_seconds']=time.perf_counter()-started
                manifest['results'][image['name']]=result
                if result['status']=='complete' and not valid_result(result,image,output):
                    result.update(status='failed',error='Output verification failed in parent process')
                report(output,manifest)
                print(f'[{index}/{len(images)}] {result["status"]}: {image["name"]} · {result.get("processing_seconds",0):.2f}s',flush=True)
        finally:client.close()
        manifest['finished_at']=datetime.now(timezone.utc).isoformat();report(output,manifest)
        print(f'Report: {output/"report.html"}')
        return 0 if all(result.get('status')=='complete' for result in manifest['results'].values()) else 1


def main():
    parser=argparse.ArgumentParser(description='Colorize a folder with Manga Light Colorizer and save five separate output folders.')
    parser.add_argument('input',help='Input image folder')
    parser.add_argument('--output',help='Default: PROJECT/output/INPUT_FOLDER_NAME')
    parser.add_argument('--recursive',action='store_true',help='Include subfolders, preserving relative paths')
    parser.add_argument('--regions-file',help='Optional source-checksummed reviewed text regions JSON')
    parser.add_argument('--ocr-language',default='eng',help='Installed Tesseract language; English filtering is currently used')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=1800,help='Model load or complete image worker timeout in seconds')
    parser.add_argument('--force',action='store_true',help='Rerun images in an otherwise matching output folder')
    args=parser.parse_args()
    if args.threads<1 or args.timeout<=0:parser.error('Threads and timeout must be positive')
    try:return run(args)
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print(f'Error: {error}',file=sys.stderr);return 1
    except KeyboardInterrupt:
        print('Interrupted; completed files can be resumed with the same command.',file=sys.stderr);return 130
