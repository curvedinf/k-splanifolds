"""Independent GPU checks of generalized KS2 kernels against eager autograd."""
import argparse,copy,hashlib,json,random,statistics,types
from pathlib import Path
import torch
from torch.nn import functional as F
import triton.testing
import run
from gpu_recurrence import recurrence
from experiments.common.ks2_stabilized import KS2StabilizedCell
from study_models_v3 import TermCell
from study_models_v10 import power_basis,convert_coefficients
from study_models_v14 import build_model
from study_fusion import with_fused_geometry
from ks2_general_gpu import install


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(8123);run.recurrence=recurrence;rows=[];failures=[]
 for rank in (1,2,4,8,16):
  for basis in ('hermite','powers','taylor'):
   for terms in ('full','spine'):
    cell=KS2StabilizedCell(5,5,3,rank,1)
    if terms=='spine':cell=TermCell(cell,dict(geometry_terms='spine'))
    if basis!='hermite':
     with torch.no_grad():
      for name in ('geo_spine','geo_t_basis_edge'):
       if hasattr(cell,name):getattr(cell,name).copy_(convert_coefficients(getattr(cell,name),basis))
     cell.study_basis=basis;cell.spine_basis=types.MethodType(power_basis,cell)
    for trainability in ('all','warps','frozen'):
     ref=copy.deepcopy(cell).cuda()
     for param_name,param in ref.named_parameters(recurse=False):
      param.requires_grad_(trainability=='all' or (trainability=='warps' and ('raw_scale' in param_name or 'raw_shift' in param_name)))
     fused=install(copy.deepcopy(ref));pr=[p for p in ref.parameters(recurse=False) if p.requires_grad];pf=[p for p in fused.parameters(recurse=False) if p.requires_grad]
     for scenario in ('normal','wide','zero','tiny','clamp'):
      x=torch.randn(17,3,rank,device='cuda')
      if scenario=='wide':x*=8
      elif scenario=='zero':x.zero_()
      elif scenario=='tiny':x*=1e-16
      elif scenario=='clamp':x*=1e5
      x.requires_grad_(True);y=x.detach().clone().requires_grad_(True);u=torch.randn_like(x)
      target=ref.geo_step(x);actual=fused.geo_step(y)
      forward_failure=None
      try:torch.testing.assert_close(actual,target,atol=5e-5,rtol=4e-4)
      except AssertionError as error:forward_failure=str(error)
      expected=torch.autograd.grad((target*u).sum(),(x,*pr));measured=torch.autograd.grad((actual*u).sum(),(y,*pf))
      if forward_failure:
       label=f'{rank}_{basis}_{terms}_{trainability}_{scenario}_forward'
       payload=dict(label=label,rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,gradient_index=None,error=forward_failure)
       failures.append(payload)
       tensors=dict(info=payload,x=x.detach().cpu(),upstream=u.detach().cpu(),state={n:p.detach().cpu() for n,p in ref.named_parameters(recurse=False)},parameter_names=[n for n,p in ref.named_parameters(recurse=False) if p.requires_grad],expected=[v.detach().cpu() for v in expected],actual=[v.detach().cpu() for v in measured],forward_expected=target.detach().cpu(),forward_actual=actual.detach().cpu())
       torch.save(tensors,a.out.parent/(label+'.pt'));print(json.dumps(payload),flush=True)
      errors=[]
      for i,(g,h) in enumerate(zip(expected,measured)):
       failure=None
       try:torch.testing.assert_close(h,g,atol=7e-5,rtol=5e-4)
       except AssertionError as error:failure=str(error)
       record=dict(index=i,max_absolute=(h-g).abs().max().item(),relative_l2=(h-g).double().norm().item()/max(g.double().norm().item(),1e-30))
       errors.append(record)
       if failure:
        label=f'{rank}_{basis}_{terms}_{trainability}_{scenario}_{i}'
        payload=dict(label=label,rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,gradient_index=i,error=failure,metrics=record)
        failures.append(payload)
        tensors=dict(info=payload,x=x.detach().cpu(),upstream=u.detach().cpu(),state={n:p.detach().cpu() for n,p in ref.named_parameters(recurse=False)},parameter_names=[n for n,p in ref.named_parameters(recurse=False) if p.requires_grad],expected=[v.detach().cpu() for v in expected],actual=[v.detach().cpu() for v in measured])
        torch.save(tensors,a.out.parent/(label+'.pt'))
        print(json.dumps(payload),flush=True)
      rows.append(dict(rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,forward_max_absolute=(actual-target).abs().max().item(),gradient_errors=errors))
    print(json.dumps(dict(rank=rank,basis=basis,terms=terms,status='SCANNED')),flush=True)
 a.out.write_text(json.dumps(dict(status='DIAGNOSTIC',component_cases=len(rows),rows=rows,failures=failures),indent=2)+'\n')
if __name__=='__main__':main()
