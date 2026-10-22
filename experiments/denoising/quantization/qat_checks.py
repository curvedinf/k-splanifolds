from pathlib import Path
import json,torch
from qat_geometry import make_model
from int8_geometry import IntegerGeometry
R=Path(__file__).parent;torch.set_num_threads(8);rows=[]
for seed in [11,22,33]:
 for kind in ['qat_combined','qat_powers']:
  q=make_model(kind,seed);q.load_state_dict(torch.load(R/f'qat_results/{kind}_{seed}.pt',weights_only=False)['state_dict']);q.eval()
  deployed=IntegerGeometry('combined' if kind=='qat_combined' else 'powers_all')
  deployed.load_state_dict({k:v for k,v in q.state_dict().items() if not k.startswith('range_')});deployed.eval();deployed.ranges={name:getattr(q,'range_'+name).clone() for name in ['basis','atten','ex']};deployed.prepare_weights()
  x=torch.randn(128,169,generator=torch.Generator().manual_seed(2026));before={k:v.clone() for k,v in q.state_dict().items() if k.startswith('range_')}
  with torch.no_grad():a=q(x);b=deployed(x)
  torch.testing.assert_close(a,b,atol=0,rtol=0)
  q.zero_grad(set_to_none=True);out=q(x);torch.testing.assert_close(out,a,atol=0,rtol=0);out.square().mean().backward()
  grad={k:dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,nonzero=int(torch.count_nonzero(p.grad)) if p.grad is not None else 0) for k,p in q.named_parameters()}
  assert all(d['present'] and d['finite'] and d['nonzero']>0 for d in grad.values()),grad
  assert all(torch.equal(before[k],q.state_dict()[k]) for k in before)
  rows.append(dict(seed=seed,kind=kind,integer_export_max_error=float((a-b).abs().max()),grad_enabled_forward_max_error=float((out.detach()-a).abs().max()),gradient_groups=grad,scales_frozen_in_eval=True))
  print('CHECK',seed,kind,'passed',flush=True)
(R/'qat_results/export_gradient_checks.json').write_text(json.dumps(rows,indent=2))
