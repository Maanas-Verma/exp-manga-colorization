# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark
from manga_colorizer.model import sha256


class BenchmarkMemoryTests(unittest.TestCase):
    def test_reloaded_first_trial_is_excluded_from_warm_median(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'page.png';source.write_bytes(b'fixture');instances=[]
            class FakeWorker:
                def __init__(self,*_):self.index=len(instances);self.job=None;self.completed=0;self.connection=SimpleNamespace(send=self.send);instances.append(self)
                def send(self,job):self.job=job
                def receive(self):
                    if self.job is None:return {'status':'loaded','model_load_seconds':.1,'memory':{'physical_footprint_bytes':100*1024**2}}
                    self.completed+=1
                    seconds=10 if self.completed==1 else 1+2*self.index
                    return {'status':'complete','processing_seconds':seconds,'inference_seconds':seconds/2,
                            'memory':{'physical_footprint_bytes':(1 if self.completed==1 else 4)*1024**3}}
                def close(self):pass
            with patch.object(benchmark,'ROOT',root),patch.object(benchmark,'Worker',FakeWorker),patch.object(benchmark,'check_storage'),patch.object(benchmark,'ensure_models'),patch.object(benchmark,'reviewed_regions',return_value={sha256(source):[]}),patch.object(sys,'argv',['benchmark.py',str(source),'--backends','coreml-gpu']):
                benchmark.main()
            record=json.loads(next((root/'.cache/benchmark-runs').glob('*/results.json')).read_text())['backends']['coreml-gpu']
            self.assertEqual(len(record['worker_loads']),2)
            self.assertEqual([t['cold_trial'] for t in record['trials']],[True,False,True,False])
            self.assertEqual(record['warm_task_median_seconds'],2)


if __name__=='__main__':unittest.main()
