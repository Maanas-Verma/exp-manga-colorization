# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto

from manga_colorizer.accelerate import rewrite_graph
from manga_colorizer.model import MangaLight


class AccelerationTests(unittest.TestCase):
    def graph(self,dimensions,dynamic=False):
        a,b,c,d,e,f=dimensions;total=int(np.prod(dimensions))
        window=[-1,*dimensions[1:]] if dynamic else dimensions
        nodes=[helper.make_node('Reshape',['x','window_shape'],['window']),
               helper.make_node('Transpose',['window'],['transposed'],perm=[0,1,3,2,4,5]),
               helper.make_node('Reshape',['transposed','output_shape'],['signed']),
               helper.make_node('Abs',['signed'],['result'])]
        graph=helper.make_graph(nodes,'window-test',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,total])],
            [helper.make_tensor_value_info('result',TensorProto.FLOAT,[a*b*d,c,e,f])],
            [numpy_helper.from_array(np.array(window,np.int64),'window_shape'),
             numpy_helper.from_array(np.array([a*b*d,c,e,f],np.int64),'output_shape')])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
        return onnx.shape_inference.infer_shapes(model)

    def test_rank_rewrite_and_abs_are_exact_in_onnx_cpu_execution(self):
        for dimensions in [[1,2,3,2,2,4],[2,3,2,2,3,1]]:
            for dynamic in [False,True]:
                with self.subTest(dimensions=dimensions,dynamic=dynamic):
                    model=self.graph(dimensions,dynamic)
                    data=np.random.default_rng(42).normal(size=(1,int(np.prod(dimensions)))).astype(np.float32)
                    expected=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider']).run(None,{'x':data})[0]
                    changes=rewrite_graph(model)
                    actual=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider']).run(None,{'x':data})[0]
                    np.testing.assert_array_equal(expected,actual)
                    self.assertEqual(changes,{'rank_six_windows_rewritten':1,'absolute_values_rewritten':1})
                    for node in model.graph.node:
                        if node.op_type=='Transpose':self.assertEqual(len(node.attribute[0].ints),4)

    def test_branched_window_is_rejected(self):
        model=self.graph([1,2,3,2,2,4]);model.graph.node.append(helper.make_node('Identity',['window'],['other']))
        with self.assertRaisesRegex(ValueError,'Unexpected SAM window pattern'):rewrite_graph(model)

    def test_gpu_request_cannot_silently_run_as_cpu(self):
        with patch.object(ort,'get_available_providers',return_value=['CPUExecutionProvider']):
            with self.assertRaisesRegex(RuntimeError,'does not include CoreML'):MangaLight(Path('unused'),backend='coreml-gpu')
        with tempfile.TemporaryDirectory() as temporary:
            cache=Path(temporary)/'models';session=Mock()
            session.get_providers.return_value=['CoreMLExecutionProvider','CPUExecutionProvider']
            with patch.object(ort,'get_available_providers',return_value=['CoreMLExecutionProvider','CPUExecutionProvider']),patch.object(ort,'InferenceSession',return_value=session),patch('manga_colorizer.accelerate.prepare_coreml_models',return_value=(cache,{'changes':{}})):
                MangaLight(cache,backend='coreml-gpu')
                self.assertEqual(session.disable_fallback.call_count,2)

    def test_cpu_sessions_do_not_prepare_gpu_models(self):
        session=Mock();session.get_providers.return_value=['CPUExecutionProvider']
        with patch.object(ort,'InferenceSession',return_value=session) as create,patch('manga_colorizer.accelerate.prepare_coreml_models') as prepare:
            model=MangaLight(Path('unused'),backend='cpu')
            self.assertEqual(model.execution['backend'],'cpu');prepare.assert_not_called()
            self.assertEqual(create.call_count,2)
            for call in create.call_args_list:self.assertEqual(call.kwargs['providers'],['CPUExecutionProvider'])


if __name__=='__main__':unittest.main()
