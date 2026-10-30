# SPDX-License-Identifier: GPL-3.0-only
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from manga_colorizer import model, pipeline
from manga_colorizer.images import letterbox, load_image, restore_size, retain_ink, validate_output
from manga_colorizer.text import region_mask, restore_text, detect_regions


class FakeModel:
    def predict(self,canvas):
        return ImageOps.colorize(canvas.convert('L'),(0,0,0),(190,130,80))


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'input';source.mkdir();(root/'assets').mkdir()
        known={}
        for index in range(2):
            image=Image.new('RGB',(140,180),'white');draw=ImageDraw.Draw(image)
            draw.rectangle((30,30,45,60),fill='black');draw.rectangle((80,100,100,140),fill=(150,150,150))
            path=source/f'page-{index}.png';image.save(path)
            known[path.name]={'sha256':model.sha256(path),'regions':[{'box':[20,20,60,75]}]}
        (root/'assets/text-regions.json').write_text(json.dumps({'images':known}))
        (root/'assets/model.json').write_text('{}')
        return source

    def args(self,root):
        return SimpleNamespace(input=str(root/'input'),output=str(root/'output/input'),recursive=False,
                               regions_file=None,ocr_language='eng',threads=4,timeout=1800,min_free_gib=2,force=False)

    def test_five_folders_resume_and_only_corrupt_page_reprocessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root);args=self.args(root);calls=[]
            class FakeWorker:
                def __init__(self,*_):self.connection=SimpleNamespace(send=self.send);self.job=None
                def send(self,job):self.job=job
                def receive(self):
                    if self.job is None:return {'status':'loaded','model_load_seconds':0.01}
                    calls.append(self.job['name']);return pipeline.process_image(FakeModel(),self.job)
                def close(self):pass
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'Worker',FakeWorker),patch.object(pipeline,'ensure_models',return_value={'files':[]}),patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)):
                self.assertEqual(pipeline.run(args),0)
                output=Path(args.output);manifest=json.loads((output/'manifest.json').read_text())
                self.assertEqual(len([path for path in output.iterdir() if path.is_dir()]),5)
                self.assertEqual(len(list(output.rglob('*.png'))),10)
                for image in manifest['images']:
                    result=manifest['results'][image['name']]
                    self.assertTrue(pipeline.valid_result(result,image,output))
                    self.assertEqual(model.sha256(Path(image['path'])),image['sha256'])
                    selected=np.asarray(load_image(output/result['artifacts']['mask']['path'])[0].convert('L'))>0
                    restored=np.asarray(load_image(output/result['artifacts']['text']['path'])[0])
                    original=np.asarray(load_image(Path(image['path']))[0]);ink=np.asarray(load_image(output/result['artifacts']['ink']['path'])[0])
                    np.testing.assert_array_equal(restored[selected],original[selected])
                    np.testing.assert_array_equal(restored[~selected],ink[~selected])
                args.min_free_gib=3
                self.assertEqual(pipeline.run(args),0);self.assertEqual(len(calls),2)
                result=manifest['results']['page-0.png'];(output/result['artifacts']['mask']['path']).write_bytes(b'corrupt')
                self.assertEqual(pipeline.run(args),0);self.assertEqual(calls,['page-0.png','page-1.png','page-0.png'])
                Path(root/'input/page-0.png').write_bytes(Path(root/'input/page-1.png').read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):pipeline.run(args)

    def test_jpeg_signature_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'wrong-extension.png';Image.new('RGB',(20,30),'white').save(path,format='JPEG')
            digest=model.sha256(path);image,format_name=load_image(path)
            self.assertEqual(format_name,'JPEG');self.assertEqual(model.sha256(path),digest)
            canvas,box=letterbox(image);self.assertEqual(canvas.size,(512,512))
            self.assertEqual(restore_size(canvas,box,image.size).size,image.size)

    def test_png_filename_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for extension in ['png','jpg']:Image.new('RGB',(20,30)).save(root/f'page.{extension}')
            with self.assertRaisesRegex(ValueError,'collide'):pipeline.discover(root)

    def test_reviewed_regions_reject_wrong_source_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root);args=self.args(root)
            regions=root/'bad-regions.json';regions.write_text('{"images":{}}');args.regions_file=str(regions)
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'ensure_models') as prepare:
                with self.assertRaisesRegex(ValueError,'checksum'):pipeline.run(args)
                prepare.assert_not_called()

    def test_input_output_nesting_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root);args=self.args(root);args.output=str(root/'input/results')
            with self.assertRaisesRegex(ValueError,'separate'):pipeline.run(args)

    def test_timeout_closes_worker(self):
        client=pipeline.Worker.__new__(pipeline.Worker);client.timeout=.01;client.connection=Mock();client.connection.poll.return_value=False;client.close=Mock()
        with self.assertRaises(TimeoutError):client.receive()
        client.close.assert_called_once()

    def test_storage_guard_rejects_low_free_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**3)):
                with self.assertRaisesRegex(RuntimeError,'1.00 GiB free; 2 GiB reserve'):pipeline.check_storage(root)

    def test_benchmark_reserve_is_optional_and_estimated_writes_are_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=19*1024**3)):
                pipeline.check_storage(root,additional=100*1024**2)
                with self.assertRaisesRegex(RuntimeError,'19.00 GiB free; 20 GiB reserve'):
                    pipeline.check_storage(root,min_free_gib=20)
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=2*1024**3)):
                with self.assertRaisesRegex(RuntimeError,'estimated new files'):
                    pipeline.check_storage(root,additional=100*1024**2)
                pipeline.check_storage(root,additional=100*1024**2,min_free_gib=0)

    def test_worker_records_failure_and_continues(self):
        connection=Mock();connection.recv.side_effect=[{'name':'bad'},{'name':'good'},None]
        with patch.object(pipeline,'MangaLight',return_value=FakeModel()),patch.object(pipeline,'process_image',side_effect=[ValueError('bad image'),{'status':'complete'}]):
            pipeline.worker(connection,'unused',4)
        sent=[call.args[0]['status'] for call in connection.send.call_args_list]
        self.assertEqual(sent,['loaded','failed','complete'])

    def test_model_download_is_pinned_and_checksum_verified(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'assets').mkdir();data=b'onnx test'
            manifest={'repo':'test/model','revision':'pinned-commit','files':[{'path':'models/tiny.onnx','size':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
            (root/'assets/model.json').write_text(json.dumps(manifest));check=Mock()
            with patch.object(model,'ROOT',root),patch.object(model.urllib.request,'urlopen',return_value=io.BytesIO(data)) as request:
                model.ensure_models(root/'cache',check)
                self.assertIn('/resolve/pinned-commit/',request.call_args.args[0])
            with patch.object(model,'ROOT',root),patch.object(model.urllib.request,'urlopen') as request:
                model.ensure_models(root/'cache',check);request.assert_not_called()
            (root/'cache/tiny.onnx').unlink()
            with patch.object(model,'ROOT',root),patch.object(model.urllib.request,'urlopen',return_value=io.BytesIO(b'bad bytes')):
                with self.assertRaisesRegex(ValueError,'checksum'):model.ensure_models(root/'cache',check)
            self.assertFalse((root/'cache/tiny.onnx.partial').exists())


class TextTests(unittest.TestCase):
    def test_antialias_text_restored_without_gray_artwork(self):
        original=Image.new('RGB',(140,100),(150,150,150));draw=ImageDraw.Draw(original)
        draw.ellipse((20,20,90,70),fill='white');draw.rectangle((38,30,46,59),fill=(180,180,180));draw.rectangle((40,32,44,57),fill='black')
        raw=Image.new('RGB',original.size,(130,85,65));ink=retain_ink(original,raw)
        restored,mask,info=restore_text(original,ink,raw,[{'box':[10,10,125,85]}])
        selected=mask>0
        self.assertTrue(selected[31,39]);self.assertTrue(selected[45,42]);self.assertFalse(selected[45,115])
        np.testing.assert_array_equal(restored[selected],np.asarray(original)[selected])
        np.testing.assert_array_equal(restored[~selected],np.asarray(ink)[~selected])
        self.assertEqual(info['method'],'reviewed')

    def test_ink_keeps_raw_bright_source_and_exact_black_source(self):
        source=Image.new('RGB',(3,1));source.putdata([(0,0,0),(60,60,60),(200,200,200)])
        raw=Image.new('RGB',(3,1),(80,130,90));result=np.asarray(retain_ink(source,raw))
        np.testing.assert_array_equal(result[0,0],[0,0,0]);np.testing.assert_array_equal(result[0,2],[80,130,90])

    def test_region_outside_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):region_mask(Image.new('RGB',(20,20)),[{'box':[0,0,21,20]}])

    def test_ocr_finds_white_dialogue_and_rejects_colored_art(self):
        tsv='level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t60\t60\t70\t20\t95\tHELLO\n'
        original=Image.new('RGB',(300,400),'white')
        with patch('manga_colorizer.text.check_ocr',return_value='tesseract'),patch('manga_colorizer.text.subprocess.run',return_value=SimpleNamespace(stdout=tsv.encode())):
            self.assertEqual(len(detect_regions(original,original)),1)
            self.assertEqual(detect_regions(original,Image.new('RGB',original.size,(130,80,40))),[])


if __name__=='__main__':unittest.main()
