"""Full validation shard, four offsets, arbitrary trained horizon/context."""
import argparse,json,hashlib
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v15 import build_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--validation',type=Path);p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--sequence-length',type=int);a=p.parse_args()
 if (a.checkpoint or a.sequence_length) and not a.output:p.error('Alternate checkpoint/context requires explicit --output')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);run.recurrence=recurrence
 r=json.loads((a.run/'results.json').read_text());args=r['arguments'];cfg=r.get('study_config')
 if cfg is None:
  assert args['variant']=='mlp' and args['width']==96 and args['layers']==4
  cfg=dict(name='original_mlp',family='mlp')
 training_length=args.get('sequence_length',r.get('sequence_length',64))
 length=a.sequence_length or training_length
 model=build_model(args['seed'],cfg).cuda().eval();checkpoint=torch.load(a.checkpoint or a.run/'checkpoint.pt',map_location='cuda',weights_only=True)
 assert checkpoint['step']<=args['steps']
 if not a.checkpoint:assert checkpoint['step']==args['steps']
 evaluation_step=checkpoint['step'];model.load_state_dict(checkpoint['model']);del checkpoint
 _,val,_=run.prepare_data(Path(args['data']),a.run)
 if a.validation:val=torch.from_numpy(np.frombuffer(a.validation.read_bytes(),dtype=np.uint8).copy()).long()
 rows=[];collected_all=[]
 signed_original=cfg.get('geometry_equation')=='original' and not cfg.get('geometry_input_sigmoid')
 batch_size=4 if signed_original else (64 if length<=64 else 16)
 for offset in (0,length//4,length//2,3*length//4):
  starts=torch.arange(offset,len(val)-length,length);windows=val[starts[:,None]+torch.arange(length+1)];count=len(windows)
  probe_windows=windows[:min(64 if length<=64 else 16,count)].cuda()
  if (-count)%batch_size:windows=torch.cat((windows,windows[:(-count)%batch_size]))
  windows=windows.reshape(-1,batch_size,length+1).cuda();static=windows[0].clone()
  with torch.no_grad():
   def per_window(x):return F.cross_entropy(model(x[:,:-1]).flatten(0,1),x[:,1:].flatten(),reduction='none').view(-1,length).mean(1)
   actual=per_window(probe_windows);expected=torch.cat([per_window(chunk) for chunk in probe_windows.split(4)])
   assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
   batch_check='PASS';batch_failure=None
   try:torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
   except AssertionError as error:
    if not signed_original:raise
    batch_check='FP32_BATCH_SENSITIVE';batch_failure=str(error)
   batch_error=float((actual-expected).abs().max())
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):losses=per_window(static)
  collected=[]
  for batch in windows:static.copy_(batch);graph.replay();collected.append(losses.clone())
  measured=torch.cat(collected)[:count];assert torch.isfinite(measured).all();collected_all.append(measured)
  rows.append(dict(offset=offset,targets=count*length,loss=float(measured.mean()),batch4_vs_eval_max_error=batch_error,batch_check=batch_check,batch_failure=batch_failure))
  del graph
 result=dict(evaluation_step=evaluation_step,training_sequence_length=training_length,loss=float(torch.cat(collected_all).mean()),targets=sum(v['targets'] for v in rows),by_offset=rows,sequence_length=length,
             evaluation_batch_size=batch_size,original_signed_equation_batch4=signed_original,note='Four overlapping tilings of one held-out shard; state resets each window. Not four independent test sets.')
 result['validation_source']=str(a.validation or Path(args['data'])/'val.bytes')
 result['validation_sha256']=hashlib.sha256((a.validation or Path(args['data'])/'val.bytes').read_bytes()).hexdigest()
 source=Path(__file__);snapshot=a.run/'evaluation_sources';snapshot.mkdir(exist_ok=True)
 (snapshot/source.name).write_bytes(source.read_bytes())
 result['evaluation_source_sha256']={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}
 if signed_original:result['numerical_note']='Use training batch4 for all signed-original-equation runs. A batch64-vs4 diagnostic is retained for every offset; near-zero coordinate sums can amplify FP32 GEMM reduction differences. No weights or training settings changed. Other families retain original evaluation batch and strict batch check.'
 (a.output or a.run/'tiled.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
