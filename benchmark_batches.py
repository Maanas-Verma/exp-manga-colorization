#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Experimental real Core ML tensor batches; includes OCR and five output saves."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from manga_colorizer.images import load_image,letterbox
from manga_colorizer.memory import memory_snapshot,recycle_gpu_worker
from manga_colorizer.model import ROOT,MangaLight,ensure_models,sha256
from manga_colorizer.pipeline import discover,check_storage,finish_image_outputs,write_json
from manga_colorizer.text import reviewed_regions,check_ocr


def tensor_worker(connection,batch_size,profile_dir=None):
    try:
        start=time.perf_counter()
        model=MangaLight(ROOT/'.cache/models',backend='coreml-gpu',batch_size=batch_size,profile_dir=profile_dir)
        connection.send({'status':'loaded','seconds':time.perf_counter()-start,'memory':memory_snapshot(),
                         'execution':model.execution,'inputs':{
                             'sam':[(x.name,x.shape) for x in model.sam.get_inputs()],
                             'generator':[(x.name,x.shape) for x in model.generator.get_inputs()]}})
        while True:
            message=connection.recv()
            if message is None:break
            start=time.perf_counter();prepared=[]
            for job in message['jobs']:
                if sha256(job['path'])!=job['sha256']:raise ValueError('Input checksum changed')
                original,format_name=load_image(Path(job['path']));canvas,box=letterbox(original,512)
                prepared.append((job,original,format_name,canvas,box))
            network_start=time.perf_counter()
            predictions=model.predict_batch([item[3] for item in prepared])
            network_seconds=time.perf_counter()-network_start
            results=[]
            if not message.get('probe'):
                for (job,original,format_name,canvas,box),prediction in zip(prepared,predictions,strict=True):
                    results.append(finish_image_outputs(job,original,format_name,canvas,box,prediction))
            result={'status':'complete','seconds':time.perf_counter()-start,'inference_seconds':network_seconds,
                    'network_timings':model.last_timings,'memory':memory_snapshot(),'pages':results}
            if message.get('probe'):
                result['profiles']=[model.sam.end_profiling(),model.generator.end_profiling()]
            connection.send(result)
            del prepared,predictions,results,result
    except (EOFError,BrokenPipeError):pass
    except Exception as error:
        connection.send({'status':'failed','error':f'{type(error).__name__}: {error}','memory':memory_snapshot()})
    finally:connection.close()


class TensorClient:
    def __init__(self,batch_size,profile_dir=None):
        context=multiprocessing.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=tensor_worker,args=(child,batch_size,profile_dir),daemon=True)
        self.process.start();child.close()
        self.peaks={'worker_physical_bytes':0,'worker_rss_bytes':0,'sampled_tree_physical_bytes':0}

    def sample(self):
        worker=memory_snapshot(self.process.pid)
        self.peaks['worker_physical_bytes']=max(self.peaks['worker_physical_bytes'],worker.get('peak_physical_footprint_bytes') or 0)
        self.peaks['worker_rss_bytes']=max(self.peaks['worker_rss_bytes'],worker.get('rss_bytes') or 0)
        # Includes this observer, its model worker and transient Tesseract children.
        listing=subprocess.run(['ps','-Ao','pid=,ppid='],capture_output=True,text=True,check=True,timeout=5)
        pairs=[tuple(map(int,line.split())) for line in listing.stdout.splitlines()]
        descendants={os.getpid()}
        while True:
            expanded=descendants|{pid for pid,parent in pairs if parent in descendants}
            if expanded==descendants:break
            descendants=expanded
        physical=sum(memory_snapshot(pid).get('physical_footprint_bytes') or 0 for pid in descendants)
        self.peaks['sampled_tree_physical_bytes']=max(self.peaks['sampled_tree_physical_bytes'],physical)
        if (worker.get('physical_footprint_bytes') or 0)>8*1024**3:
            raise RuntimeError('Experiment stopped: model worker exceeded 8 GiB physical footprint')
        if shutil.disk_usage(ROOT).free<2*1024**3:
            raise RuntimeError('Experiment stopped: less than 2 GiB immediately free disk space')

    def receive(self):
        deadline=time.monotonic()+1800
        while True:
            self.sample()
            if self.connection.poll(.06):
                try:result=self.connection.recv()
                except EOFError as error:raise RuntimeError('Tensor worker ended unexpectedly') from error
                if result['status']=='failed':raise RuntimeError(result['error'])
                memory=result.get('memory',{})
                self.peaks['worker_physical_bytes']=max(self.peaks['worker_physical_bytes'],memory.get('peak_physical_footprint_bytes') or 0)
                return result
            if time.monotonic()>deadline:raise TimeoutError('Tensor worker exceeded 30 minutes')
            if not self.process.is_alive():raise RuntimeError('Tensor worker exited')

    def close(self):
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=5)
            if self.process.is_alive():self.process.kill();self.process.join(timeout=5)
        self.connection.close()


def profile_summary(paths):
    events=[event for path in paths for event in json.loads(Path(path).read_text())]
    providers={}
    for event in events:
        provider=event.get('args',{}).get('provider')
        if provider:providers[provider]=providers.get(provider,0)+1
    if providers.get('CoreMLExecutionProvider')!=2 or providers.get('CPUExecutionProvider',0):
        raise RuntimeError(f'Expected one GPU section per network with no CPU nodes, got {providers}')
    return providers


def compare_outputs(work,images,batch_size):
    records=[]
    for image in images:
        relative=Path('01_raw_model_output')/image['output_name']
        baseline=np.asarray(Image.open(work/'batch-1'/relative).convert('RGB')).astype(np.int16)
        actual=np.asarray(Image.open(work/f'batch-{batch_size}'/relative).convert('RGB')).astype(np.int16)
        if baseline.shape!=actual.shape:raise ValueError('Batched output geometry changed')
        difference=np.abs(actual-baseline)
        if not np.isfinite(actual).all():raise ValueError('Non-finite batched pixels')
        folder=work/f'batch-{batch_size}'
        mask=np.asarray(Image.open(folder/'04_text_restoration_mask'/image['output_name']).convert('L'))
        if not np.isin(mask,[0,255]).all():raise ValueError('Text mask is not binary')
        source=np.asarray(load_image(Path(image['path']))[0])
        restored=np.asarray(Image.open(folder/'02_original_text_recommended'/image['output_name']).convert('RGB'))
        ink=np.asarray(Image.open(folder/'03_ink_only_previous'/image['output_name']).convert('RGB'))
        selected=mask>0
        if not np.array_equal(restored[selected],source[selected]):raise ValueError('Saved text pixels differ from source')
        if not np.array_equal(restored[~selected],ink[~selected]):raise ValueError('Saved text restoration changes pixels outside mask')
        reference_mask=np.asarray(Image.open(work/'batch-1'/'04_text_restoration_mask'/image['output_name']).convert('L'))
        record={'image':image['name'],'mean_absolute_channel_error':float(difference.mean()),
                'max_channel_error':int(difference.max()),'mask_pixel_differences':int(np.count_nonzero(mask!=reference_mask)),
                'saved_text_pixels_match_source':True,'saved_pixels_outside_mask_match_ink':True}
        records.append(record)
        # A different batch must not materially alter per-page color predictions.
        if record['mean_absolute_channel_error']>.1 or record['max_channel_error']>8:
            raise ValueError(f'Tensor batch changes colors materially: {record}')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--pages',type=int,default=24)
    parser.add_argument('--batches',type=int,nargs='+',choices=[1,2,4],default=[1,2,4])
    args=parser.parse_args()
    if platform.system()!='Darwin':parser.error('This tensor-batch experiment requires macOS with Core ML')
    if len(set(args.batches))!=len(args.batches):parser.error('Specify each batch size only once')
    if args.pages<4 or args.pages%4:parser.error('--pages must be a positive multiple of 4')
    if not args.input.is_dir():parser.error('Provide an existing input folder')
    work=ROOT/'.cache/tensor-batch-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.cache/run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another colorization is running')
        images=discover(args.input.resolve())[:args.pages]
        if len(images)!=args.pages:parser.error('Not enough input images')
        regions=reviewed_regions(ROOT/'assets/text-regions.json')
        if any(image['sha256'] not in regions for image in images):check_ocr('eng')
        check_storage(work,3*1024**3)
        ensure_models(ROOT/'.cache/models',lambda additional=0:check_storage(work,additional))
        document={'date':datetime.now(timezone.utc).isoformat(),'machine':platform.machine(),'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True).strip()),
                  'canvas':[512,512],'pages_per_configuration':len(images),'worker_recycle_gib':3,
                  'timing_scope':'Initial cached model load, image reading, preprocessing, actual tensor inference, OCR, five PNG saves, validation, worker reloads; first graph compilation/probe excluded',
                  'images':[{key:image[key] for key in ('name','sha256','size')} for image in images],
                  'configurations':{}}
        for batch_size in args.batches:
            record={'status':'running','batch_size':batch_size,'loads':[],'groups':[],'peaks':{}}
            document['configurations'][str(batch_size)]=record
            def jobs_for(group):
                return [{**image,'regions':regions.get(image['sha256']),'language':'eng',
                         'output':str(work/f'batch-{batch_size}')} for image in group]
            client=None
            try:
                # Prepare static-shape compilation and verify actual execution placement.
                print(f'Batch {batch_size}: compiling/profiling GPU graphs',flush=True)
                setup_start=time.perf_counter();client=TensorClient(batch_size,str(work/f'profile-{batch_size}'))
                initial_load=client.receive()
                client.connection.send({'probe':True,'jobs':jobs_for(images[:batch_size])})
                probe=client.receive();record['gpu_profile']=profile_summary(probe['profiles'])
                record['inputs']=initial_load['inputs'];record['setup_peak']=dict(client.peaks)
                record['setup_load_seconds']=initial_load['seconds'];record['setup_probe_seconds']=probe['seconds']
                client.close();client=None;record['setup_seconds']=time.perf_counter()-setup_start
                start=time.perf_counter();completed=0;last_memory=None
                for offset in range(0,len(images),batch_size):
                    if client is None or recycle_gpu_worker('coreml-gpu',last_memory,3,completed):
                        if client:
                            for key,value in client.peaks.items():record['peaks'][key]=max(record['peaks'].get(key,0),value)
                            client.close()
                        client=TensorClient(batch_size);loaded=client.receive();record['loads'].append(loaded);completed=0
                    client.connection.send({'jobs':jobs_for(images[offset:offset+batch_size])})
                    result=client.receive();last_memory=result['memory'];completed+=batch_size
                    record['groups'].append(result)
                    for key,value in client.peaks.items():record['peaks'][key]=max(record['peaks'].get(key,0),value)
                    print(f'Batch {batch_size}: {offset+batch_size}/{len(images)} pages, {result["seconds"]:.3f}s/group, footprint {(last_memory["physical_footprint_bytes"] or 0)/1024**3:.2f} GiB',flush=True)
                    write_json(work/'results.json',document)
                record['wall_seconds']=time.perf_counter()-start
                record['pages_per_minute']=len(images)*60/record['wall_seconds']
                record['network_only_pages_per_minute']=len(images)*60/sum(group['inference_seconds'] for group in record['groups'])
                record['status']='complete'
                for image in images:
                    if sha256(image['path'])!=image['sha256']:raise ValueError('Input changed during experiment')
            except Exception as error:
                record['status']='failed';record['error']=f'{type(error).__name__}: {error}'
                print(f'Batch {batch_size}: FAILED: {record["error"]}',flush=True)
            finally:
                if client:client.close()
                write_json(work/'results.json',document)
            print(f'Batch {batch_size}: {record.get("pages_per_minute",0):.2f} pages/min, {record["peaks"]}',flush=True)
        if document['configurations'].get('1',{}).get('status')=='complete':
            for size,record in document['configurations'].items():
                if record['status']!='complete':continue
                try:record['quality_against_batch_1']=compare_outputs(work,images,int(size))
                except Exception as error:
                    record['status']='failed';record['error']=f'{type(error).__name__}: {error}'
            write_json(work/'results.json',document)
        print(f'Results: {work/"results.json"}',flush=True)


if __name__=='__main__':main()
