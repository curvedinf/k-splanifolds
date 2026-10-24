"""Check and benchmark the integrated export under torchrun TP1/2/4."""
import argparse,hashlib,json,os,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v19 import build_model
from rank1_int8_tp import convert,inference_recurrence


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint-run',dest='run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=4);p.add_argument('--windows',type=int,default=128);p.add_argument('--repeats',type=int,default=3);p.add_argument('--full-eval',action='store_true');p.add_argument('--collective',choices=['rccl','custom'],default='rccl');p.add_argument('--validation',type=Path);a=p.parse_args()
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 if world>1:dist.init_process_group('nccl',device_id=torch.device('cuda',local))
 custom=None
 if world>1 and a.collective=='custom':
  from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
  cpu_group=dist.new_group(backend='gloo');custom=CustomAllreduce(cpu_group,torch.device('cuda',local),max_size=8*1024*1024)
  assert not custom.disabled,'Custom all-reduce unavailable; no silent fallback'
 def reduce(x):
  result=custom.custom_all_reduce(x);assert result is not None;return result
 a.out.mkdir(parents=True,exist_ok=True);meta=json.loads((a.run/'completed.json').read_text());cfg=meta['study_config'];model=build_model(meta['arguments']['seed'],cfg).cuda().eval();model.load_state_dict(torch.load(a.run/'checkpoint.pt',weights_only=True,map_location=f'cuda:{local}')['model'])
 # Existing exploratory validation only. Final confirmation interval is not used to tune export.
 valpath=a.validation or Path(meta['arguments']['data'])/'val.bytes';data=torch.from_numpy(np.frombuffer(valpath.read_bytes(),dtype=np.uint8).copy()).long()
 starts=torch.cat([torch.arange(offset,len(data)-64,64) for offset in (0,16,32,48)]) if a.full_eval else torch.arange(0,len(data)-64,64)[:a.windows]
 windows=data[starts[:,None]+torch.arange(65)];count=len(windows)
 if (-count)%a.batch:windows=torch.cat((windows,windows[:(-count)%a.batch]))
 windows=windows.reshape(-1,a.batch,65).cuda();x=windows[0,:,:-1].contiguous();target=windows[0,:,1:].contiguous()
 result=dict(nonlinear_scope='KS2 warp, Horner polynomial and ReLU-squared; RWKV controller nonlinearities remain FP32',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('rank1_int8_tp.py'),Path(__file__).with_name('rank1_int8_attention_tp.py'))},torch_version=torch.__version__,hip_version=torch.version.hip,device=torch.cuda.get_device_name(),collective=a.collective,world=world,batch=a.batch,context=64,seed=meta['arguments']['seed'],checkpoint=str(a.run/'checkpoint.pt'),validation_sha256=hashlib.sha256(valpath.read_bytes()).hexdigest(),windows=count,state_dtype='float32',cases=[])
 def barrier():
  if world>1:dist.barrier()
 def maximum(v):
  q=torch.tensor(v,device='cuda')
  if world>1:dist.all_reduce(q,op=dist.ReduceOp.MAX)
  return float(q)
 with torch.inference_mode():
  run.recurrence=recurrence;reference=model(x);run.recurrence=inference_recurrence;fastref=model(x)
  torch.testing.assert_close(fastref,reference,atol=3e-5,rtol=3e-5)
  for name in ('baseline','fp32','fp16','int8'):
   x.copy_(windows[0,:,:-1]);target.copy_(windows[0,:,1:])
   run.recurrence=recurrence if name=='baseline' else inference_recurrence
   current=model if name=='baseline' else convert(model,rank,world,name,reduce if custom else None)
   # Three identical forwards initialize collectives and kernels before graph capture.
   for _ in range(3):logits=current(x)
   finite=bool(torch.isfinite(logits).all());maxerr=float((logits-reference).abs().max());rms=float((logits-reference).square().mean().sqrt())
   if not finite:
    result['cases'].append(dict(nonlinear=name,status='NONFINITE',max_logit_error=maxerr));continue
   # Validate distributed summation against the same quantization groups on one GPU.
   tp_error=None;tp_rms=None;ffn_errors=[];attention_errors=[]
   if name!='baseline' and world>1:
    single=convert(model,0,1,name)
    torch.manual_seed(543);probe=torch.randn(a.batch,64,96,device='cuda')
    ffn_errors=[];attention_errors=[]
    for sharded,unsharded in zip(current.blocks,single.blocks):
     actual_ffn=sharded.ffn(probe);expected_ffn=unsharded.ffn(probe)
     ffn_errors.append(float((actual_ffn-expected_ffn).abs().max()))
     torch.testing.assert_close(actual_ffn,expected_ffn,atol=1e-4,rtol=2e-5)
     local_attention=sharded.attention
     first=torch.randn_like(probe) if unsharded.attention.value_control is not None else None
     local_first=first[...,local_attention.head_start:local_attention.head_end].contiguous() if first is not None and local_attention.heads else None
     attention_actual,_=local_attention(probe,local_first);attention_expected,_=unsharded.attention(probe,first)
     attention_errors.append(float((attention_actual-attention_expected).abs().max()))
     torch.testing.assert_close(attention_actual,attention_expected,atol=5e-5,rtol=5e-5)
    expected=single(x);tp_error=float((logits-expected).abs().max())
    tp_rms=float((logits-expected).square().mean().sqrt());del single
   barrier();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with (custom.capture() if custom and name!='baseline' else nullcontext()), torch.cuda.graph(graph):
    output=current(x);loss=F.cross_entropy(output.flatten(0,1),target.flatten(),reduction='none').view(a.batch,64).mean(1)
   for _ in range(5):graph.replay()
   replay_error=float((output-logits).abs().max())
   torch.testing.assert_close(output,logits,atol=3e-4,rtol=3e-4)
   times=[]
   for repeat in range(a.repeats):
    barrier();torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(100):graph.replay()
    torch.cuda.synchronize();times.append(maximum((time.perf_counter()-begin)*10))
   scores=[]
   for batch in windows:
    x.copy_(batch[:,:-1]);target.copy_(batch[:,1:]);graph.replay();scores.append(loss.clone())
   score=float(torch.cat(scores)[:count].mean());assert math_isfinite(score)
   row=dict(nonlinear=name,status='PASS',loss=score,max_logit_error=maxerr,rms_logit_error=rms,tp_vs_same_quant_tp1_max_error=tp_error,tp_vs_same_quant_tp1_rms=tp_rms,ffn_partition_errors=ffn_errors,attention_partition_errors=attention_errors,graph_vs_eager_max_error=replay_error,ms=times,trainable_parameters=sum(p.numel() for p in current.parameters() if p.requires_grad),int8_buffers=sum(b.numel() for b in current.buffers() if b.dtype==torch.int8),float_parameters=sum(p.numel() for p in current.parameters()))
   result['cases'].append(row)
   if rank==0:print(json.dumps(row),flush=True);(a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
   del graph,current
 if rank==0:
  result['status']='PASS';(a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 if custom:custom.close()
 if world>1:dist.destroy_process_group()


def math_isfinite(v):return v==v and abs(v)!=float('inf')
if __name__=='__main__':main()
