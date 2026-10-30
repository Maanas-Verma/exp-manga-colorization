#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Repeat the same image on CPU and Apple GPU, including all five output saves."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import statistics

from manga_colorizer.model import ROOT,ensure_models,sha256
from manga_colorizer.pipeline import Worker,check_storage
from manga_colorizer.memory import recycle_gpu_worker
from manga_colorizer.text import check_ocr,reviewed_regions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--trials',type=int,default=4,help='One cold image plus warm repeats (default: 4 total)')
    parser.add_argument('--backends',nargs='+',choices=['cpu','coreml-gpu'],default=['cpu','coreml-gpu'])
    args=parser.parse_args()
    if args.trials<2:parser.error('Use at least two trials')
    source=args.image.expanduser().resolve()
    if not source.is_file():parser.error('Provide an existing image')
    digest=sha256(source);regions=reviewed_regions(ROOT/'assets/text-regions.json').get(digest)
    if regions is None:check_ocr('eng')
    work=ROOT/'.cache/benchmark-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.cache/run.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another colorization is running; benchmark after it completes')
        check_storage(work,2*1024**3)
        ensure_models(ROOT/'.cache/models',lambda additional=0:check_storage(work,additional))
        results={'image':source.name,'input_sha256':digest,'trials_per_backend':args.trials,'backends':{}}
        for backend in args.backends:
            client=Worker(ROOT/'.cache/models',4,1800,backend)
            try:
                loaded=client.receive()
                if loaded['status']!='loaded':raise RuntimeError(loaded.get('error','Model load failed'))
                trials=[];loads=[loaded];completed_in_worker=0;last_memory=loaded.get('memory')
                for index in range(args.trials):
                    reloaded=False
                    if completed_in_worker and recycle_gpu_worker(backend,last_memory,3,completed_in_worker):
                        client.close();client=Worker(ROOT/'.cache/models',4,1800,backend)
                        loaded=client.receive()
                        if loaded['status']!='loaded':raise RuntimeError(loaded.get('error','Model reload failed'))
                        loads.append(loaded);completed_in_worker=0;reloaded=True
                    folder=work/backend/f'trial-{index}'
                    client.connection.send({'path':str(source),'sha256':digest,'regions':regions,'language':'eng',
                                            'output':str(folder),'output_name':'image.png'})
                    result=client.receive()
                    if result['status']!='complete':raise RuntimeError(result.get('error','Image failed'))
                    result['cold_trial']=index==0 or reloaded
                    completed_in_worker+=1;last_memory=result.get('memory')
                    trials.append(result)
                    print(f'{backend} {index+1}/{args.trials}: {result["processing_seconds"]:.3f}s task, {result["inference_seconds"]:.3f}s inference',flush=True)
                warm=[trial for trial in trials if not trial['cold_trial']]
                results['backends'][backend]={'model_load_seconds':loads[0]['model_load_seconds'],'worker_loads':loads,
                    'execution':loaded.get('execution'),'trials':trials,
                    'warm_task_median_seconds':statistics.median(t['processing_seconds'] for t in warm) if warm else None,
                    'warm_inference_median_seconds':statistics.median(t['inference_seconds'] for t in warm) if warm else None}
                (work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            finally:client.close()
        print(f'Results: {work/"results.json"}')


if __name__=='__main__':main()
