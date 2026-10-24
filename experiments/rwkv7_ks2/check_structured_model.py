"""Trained logit/gradient equivalence and compact projection storage check."""
import argparse,copy,json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from study_models_v10 import build_model
from structured_codes import install


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--real-tokens',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(12849)
 r=json.loads((a.run/'results.json').read_text());cfg=dict(r['study_config'],fused_geometry=False,fused_partial=False)
 if a.device=='cuda':
  from gpu_recurrence import recurrence
  run.recurrence=recurrence
 model=build_model(r['arguments']['seed'],cfg);model.load_state_dict(torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
 compact=install(copy.deepcopy(model));model=model.to(a.device);compact=compact.to(a.device)
 tokens=torch.randint(256,(2,17),device=a.device)
 if a.real_tokens:
  _,val,_=run.prepare_data(Path(r['arguments']['data']),a.run)
  pos=torch.randint(len(val)-65,(2,),generator=torch.Generator().manual_seed(419))
  tokens=val[pos[:,None]+torch.arange(65)].to(a.device)
 target=tokens[:,1:].flatten()
 def calculate(m):
  y=m(tokens[:,:-1]);loss=F.cross_entropy(y.flatten(0,1),target);loss.backward()
  return y.detach(),loss.item(),{k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
 y,loss,g=calculate(model);z,other,h=calculate(compact)
 torch.testing.assert_close(y,z,atol=5e-5,rtol=5e-5);assert g.keys()==h.keys()
 rows=[]
 for name in g:
  error=(g[name]-h[name]).abs().max().item();relative=(g[name]-h[name]).norm().item()/max(g[name].norm().item(),1e-20)
  scale=g[name].abs().max().item()
  rows.append(dict(name=name,max_absolute=error,relative_l2=relative,max_absolute_relative_to_tensor_max=error/max(scale,1e-20)))
 out=dict(status='DIAGNOSTIC',device=a.device,input_source='exploratory validation' if a.real_tokens else 'random token stress',trained_steps=r['arguments']['steps'],logit_max_error=(y-z).abs().max().item(),original_loss=loss,compact_loss=other,original_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),compact_buffer_bytes=sum(b.numel()*b.element_size() for b in compact.buffers()),parameters=sum(p.numel() for p in compact.parameters()),gradient_rows=rows,source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['structured_codes.py','structured_codes_gpu.py','check_structured_model.py']})
 failures=[x for x in rows if x['relative_l2']>1e-4 and x['max_absolute']>1e-7]
 out['gradient_tolerance']='Per-tensor relativeL2 <=1e-4 or maxabsolute <=1e-7; all component projection equations checked independently.'
 out['gradient_failures']=failures;out['status']='PASS' if not failures else 'FAIL'
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='gradient_rows'}),flush=True)
 assert not failures, failures
if __name__=='__main__':main()
