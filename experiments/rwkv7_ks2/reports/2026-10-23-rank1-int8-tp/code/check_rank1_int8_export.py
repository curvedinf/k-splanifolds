"""Verify every TP shard and all precision modes after loading the portable export."""
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
import run  # Establish repository package root before model imports.
from study_models_v19 import build_model
from rank1_int8_tp import convert
from export_rank1_int8_tp import load_export


def main():
 p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--checkpoint-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
 from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
 group=dist.new_group(backend='gloo');comm=CustomAllreduce(group,torch.device('cuda',rank),max_size=8*1024*1024);assert not comm.disabled
 def reduce(x):
  out=comm.custom_all_reduce(x);assert out is not None;return out
 meta=json.loads((a.checkpoint_run/'completed.json').read_text());base=build_model(meta['arguments']['seed'],meta['study_config']).cuda().eval();base.load_state_dict(torch.load(a.checkpoint_run/'checkpoint.pt',weights_only=True,map_location=f'cuda:{rank}')['model'])
 torch.manual_seed(12);x=torch.randint(0,256,(4,64),device='cuda');checks=[]
 with torch.inference_mode():
  for precision in ['fp32','fp16','int8']:
   expected=convert(base,rank,world,precision,reduce);loaded=load_export(a.export,rank,f'cuda:{rank}',precision,reduce)
   left=expected.state_dict();right=loaded.state_dict();assert left.keys()==right.keys()
   for key in left:assert torch.equal(left[key],right[key]),key
   aout=expected(x);bout=loaded(x);torch.testing.assert_close(aout,bout,atol=0,rtol=0)
   checks.append(dict(precision=precision,state_tensors=len(left),max_logit_error=float((aout-bout).abs().max())))
   del expected,loaded
 allchecks=[None]*world;dist.all_gather_object(allchecks,dict(rank=rank,checks=checks))
 if rank==0:a.out.write_text(json.dumps(dict(status='PASS',ranks=allchecks),indent=2)+'\n');print('PASS: all four shards and all three precision settings reload exactly',flush=True)
 comm.close();dist.destroy_process_group()
if __name__=='__main__':main()
