# SPDX-License-Identifier: GPL-3.0-only
"""Equivalent ONNX graph rewrites that avoid Core ML/CPU partition boundaries."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .model import ROOT,sha256

REWRITE_VERSION=1
FIXED_DIMENSIONS={'batch':1,'height':512,'width':512,'sam_h0':64,'sam_w0':64,'sam_h1':32,'sam_w1':32}


def rewrite_graph(model):
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    producers={value:node for node in model.graph.node for value in node.output}
    consumers=defaultdict(list)
    for node in model.graph.node:
        for value in node.input:consumers[value].append(node)
    arrays={value.name:numpy_helper.to_array(value) for value in model.graph.initializer
            if value.data_type in (TensorProto.INT64,TensorProto.INT32)}
    shapes={value.name:[d.dim_value for d in value.type.tensor_type.shape.dim]
            for value in [*model.graph.input,*model.graph.value_info,*model.graph.output]}
    windows=0
    for node in model.graph.node:
        if node.op_type!='Transpose':continue
        permutation=next((attr for attr in node.attribute if attr.name=='perm'),None)
        if permutation is None or list(permutation.ints)!=[0,1,3,2,4,5]:continue
        before=producers.get(node.input[0]);after=consumers[node.output[0]]
        if before is None or before.op_type!='Reshape' or len(consumers[before.output[0]])!=1 or len(after)!=1 or after[0].op_type!='Reshape':
            raise ValueError('Unexpected SAM window pattern; use the CPU backend')
        dimensions=arrays[before.input[1]].astype(int).tolist()
        if -1 in dimensions:
            source_shape=shapes.get(before.input[0])
            if not source_shape or not all(source_shape):raise ValueError('Unknown static SAM window size')
            dimensions[dimensions.index(-1)]=int(np.prod(source_shape))//int(np.prod([d for d in dimensions if d!=-1]))
        if len(dimensions)!=6 or not all(d>0 for d in dimensions):raise ValueError('Invalid static SAM window shape')
        # [a,b,c,d,e,f] -> transpose(0,1,3,2,4,5) is exactly
        # [a*b,c,d,e*f] -> transpose(0,2,1,3), followed by the same final reshape.
        a,b,c,d,e,f=dimensions
        name=f'{before.input[1]}_coreml4_{windows}'
        model.graph.initializer.append(numpy_helper.from_array(np.array([a*b,c,d,e*f],np.int64),name))
        before.input[1]=name;before.output[0]+='_coreml4'
        node.input[0]=before.output[0];node.output[0]+='_coreml4';after[0].input[0]=node.output[0]
        del permutation.ints[:];permutation.ints.extend([0,2,1,3]);windows+=1
    nodes=[];absolute_values=0;negative='_coreml_negative_one'
    for node in model.graph.node:
        if node.op_type=='Abs':
            intermediate=node.output[0]+'_negative'
            nodes.append(helper.make_node('Mul',[node.input[0],negative],[intermediate],name=node.name+'_negative'))
            nodes.append(helper.make_node('Max',[node.input[0],intermediate],list(node.output),name=node.name+'_max'))
            absolute_values+=1
        else:nodes.append(node)
    if absolute_values:model.graph.initializer.append(numpy_helper.from_array(np.array(-1,np.float32),negative))
    del model.graph.node[:];model.graph.node.extend(nodes)
    del model.graph.value_info[:]  # Old rank-six annotations are no longer valid.
    used={value for node in model.graph.node for value in node.input}|{value.name for value in model.graph.output}
    # Static specialization folded control flow in these pinned exports.
    if not any(attr.type==onnx.AttributeProto.GRAPH for node in nodes for attr in node.attribute):
        retained=[value for value in model.graph.initializer if value.name in used]
        del model.graph.initializer[:];model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    return {'rank_six_windows_rewritten':windows,'absolute_values_rewritten':absolute_values}


def prepare_coreml_models(cache):
    import onnx
    import onnxruntime as ort
    cache=Path(cache)
    destination=cache.parent/'coreml-onnx'/f'v{REWRITE_VERSION}'/sha256(ROOT/'assets/model.json')[:16]
    destination.mkdir(parents=True,exist_ok=True);record=destination/'conversion.json'
    specification=json.loads((ROOT/'assets/model.json').read_text())
    expected={'rewrite_version':REWRITE_VERSION,'onnxruntime':ort.__version__,
              'source_sha256':{Path(item['path']).name:item['sha256'] for item in specification['files']}}
    if record.exists():
        metadata=json.loads(record.read_text())
        if metadata.get('settings')==expected and all((destination/name).exists() and sha256(destination/name)==digest for name,digest in metadata.get('output_sha256',{}).items()) and len(metadata.get('output_sha256',{}))==2:
            return destination,metadata
    details={};digests={}
    for name,source_digest in expected['source_sha256'].items():
        source=cache/name
        if sha256(source)!=source_digest:raise ValueError(f'Pinned source checksum changed: {name}')
        print(f'Preparing Apple GPU graph: {name}',flush=True)
        basic=destination/('basic-'+name)
        options=ort.SessionOptions();options.intra_op_num_threads=4
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.log_severity_level=3;options.optimized_model_filepath=str(basic)
        for key,value in FIXED_DIMENSIONS.items():options.add_free_dimension_override_by_name(key,value)
        session=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider']);del session
        try:
            model=onnx.shape_inference.infer_shapes(onnx.load(basic))
            changes=rewrite_graph(model)
            required={'rank_six_windows_rewritten':18 if 'sam' in name else 0,
                      'absolute_values_rewritten':0 if 'sam' in name else 6}
            if changes!=required:raise ValueError('Pinned graph patterns changed; use --backend cpu')
            temporary=destination/(name+'.tmp');onnx.save(model,temporary);temporary.replace(destination/name)
            details[name]=changes;digests[name]=sha256(destination/name)
        finally:basic.unlink(missing_ok=True)
    metadata={'settings':expected,'changes':details,'output_sha256':digests}
    temporary=record.with_suffix('.json.tmp');temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(record)
    return destination,metadata
