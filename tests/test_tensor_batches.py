# SPDX-License-Identifier: GPL-3.0-only
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from manga_colorizer.model import MangaLight
from benchmark_batches import profile_summary,compare_outputs


class TensorBatchTests(unittest.TestCase):
    def test_two_networks_receive_true_batches_and_keep_page_order(self):
        for size in (1,2,4):
            with self.subTest(batch_size=size):
                model=object.__new__(MangaLight);model.batch_size=size
                model.sam=Mock();model.generator=Mock()
                model.sam.get_inputs.return_value=[SimpleNamespace(name='rgb')]
                values=[37,89,173,241][:size]
                canvases=[Image.new('RGB',(8,8),(value,value,value)) for value in values]
                def sam_run(_,feed):
                    self.assertEqual(feed['rgb'].shape,(size,3,1024,1024))
                    for index,value in enumerate(values):
                        np.testing.assert_allclose(feed['rgb'][index],value/255,atol=1e-7)
                    return [np.zeros((size,2,4,4),np.float32),np.zeros((size,2,2,2),np.float32)]
                def generator_run(_,feed):
                    self.assertEqual(feed['L_bw'].shape,(size,1,8,8))
                    self.assertEqual(feed['sam_level0'].shape[0],size)
                    self.assertEqual(feed['sam_level1'].shape[0],size)
                    np.testing.assert_array_equal(feed['wd14_embedding'],np.zeros((size,1024),np.float32))
                    return [np.repeat(feed['L_bw'],3,axis=1)]
                model.sam.run.side_effect=sam_run;model.generator.run.side_effect=generator_run
                results=model.predict_batch(canvases)
                for image,value in zip(results,values,strict=True):np.testing.assert_array_equal(np.asarray(image),value)
                self.assertEqual(model.sam.run.call_count,1);self.assertEqual(model.generator.run.call_count,1)
                self.assertGreaterEqual(model.last_timings['sam_seconds'],0)

    def test_mismatched_batch_is_rejected_before_inference(self):
        model=object.__new__(MangaLight);model.batch_size=2;model.sam=Mock()
        canvas=Image.new('RGB',(8,8))
        with self.assertRaisesRegex(ValueError,'compiled tensor batch'):model.predict_batch([canvas])
        with self.assertRaisesRegex(ValueError,'identical dimensions'):model.predict_batch([canvas,Image.new('RGB',(16,8))])
        with self.assertRaisesRegex(ValueError,'predict_batch'):model.predict(canvas)
        model.sam.run.assert_not_called()

    def test_single_predict_and_batch_api_agree(self):
        model=object.__new__(MangaLight);model.batch_size=1
        canvas=Image.new('RGB',(8,8));prediction=Image.new('RGB',(8,8),(42,74,99))
        model.predict_batch=Mock(return_value=[prediction])
        self.assertIs(model.predict(canvas),prediction);model.predict_batch.assert_called_once_with([canvas])

    def test_gpu_profile_rejects_cpu_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'profile.json'
            events=[{'args':{'provider':'CoreMLExecutionProvider'}}]*2
            path.write_text(json.dumps(events))
            self.assertEqual(profile_summary([path]),{'CoreMLExecutionProvider':2})
            path.write_text(json.dumps(events+[{'args':{'provider':'CPUExecutionProvider'}}]))
            with self.assertRaisesRegex(RuntimeError,'no CPU nodes'):profile_summary([path])

    def test_saved_text_invariants_are_checked_after_batch_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(4,4),(180,180,180)).save(source)
            mask=np.zeros((4,4),np.uint8);mask[:2,:2]=255
            ink=np.full((4,4,3),42,np.uint8);restored=ink.copy();restored[mask>0]=180
            for size in (1,2):
                raw=np.full((4,4,3),21,np.uint8)
                if size==2:raw[0,0,0]=22
                images={'01_raw_model_output':Image.fromarray(raw),
                        '02_original_text_recommended':Image.fromarray(restored),
                        '03_ink_only_previous':Image.fromarray(ink),'04_text_restoration_mask':Image.fromarray(mask)}
                for name,image in images.items():
                    folder=root/f'batch-{size}'/name;folder.mkdir(parents=True);image.save(folder/'page.png')
            records=[{'name':'page.png','output_name':'page.png','path':str(source)}]
            result=compare_outputs(root,records,2)[0]
            self.assertEqual(result['max_channel_error'],1);self.assertEqual(result['mask_pixel_differences'],0)
            restored[0,0]=0
            Image.fromarray(restored).save(root/'batch-2'/'02_original_text_recommended'/'page.png')
            with self.assertRaisesRegex(ValueError,'text pixels differ'):compare_outputs(root,records,2)


if __name__=='__main__':unittest.main()
