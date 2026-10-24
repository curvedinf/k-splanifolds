"""Evaluate the exact same 64 target bytes with 64 versus 256 bytes of context."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence
from study_models_v6 import build_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--context',type=int,choices=[64,256],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);run.recurrence=recurrence
 r=json.loads((a.run/'results.json').read_text());model=build_model(r['arguments']['seed'],r['study_config']).cuda().eval()
 cp=torch.load(a.run/'checkpoint.pt',map_location='cuda',weights_only=True);assert cp['step']==r['arguments']['steps'];model.load_state_dict(cp['model']);del cp
 _,val,_=run.prepare_data(Path(r['arguments']['data']),a.run);all_losses=[];rows=[];batch_size=16
 for offset in (0,16,32,48):
  common=torch.arange(offset,len(val)-256,64);starts=common+256-a.context;windows=val[starts[:,None]+torch.arange(a.context+1)];count=len(windows)
  if (-count)%batch_size:windows=torch.cat((windows,windows[:(-count)%batch_size]))
  windows=windows.reshape(-1,batch_size,a.context+1).cuda();static=windows[0].clone()
  with torch.no_grad():
   def per_window(x):return F.cross_entropy(model(x[:,:-1])[:,-64:].reshape(-1,256),x[:,1:][:,-64:].flatten(),reduction='none').view(-1,64).mean(1)
   actual=per_window(static);expected=torch.cat([per_window(chunk) for chunk in static.split(4)]);torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
   batch_error=float((actual-expected).abs().max());graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):losses=per_window(static)
  measured=[]
  for batch in windows:static.copy_(batch);graph.replay();measured.append(losses.clone())
  measured=torch.cat(measured)[:count];assert torch.isfinite(measured).all();all_losses.append(measured)
  rows.append(dict(offset=offset,targets=count*64,loss=float(measured.mean()),batch_error=batch_error));del graph
 result=dict(context=a.context,scored_suffix=64,common_reference_context=256,loss=float(torch.cat(all_losses).mean()),targets=sum(row['targets'] for row in rows),offsets=rows,training_sequence_length=r['sequence_length'],training_batch_size=r['batch_size'],training_steps=r['arguments']['steps'],validation_sha256=r['data']['val']['sha256'],note='Identical target positions across both context lengths; the extra 192 bytes are causal prefix only.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
