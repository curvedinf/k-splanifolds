import json
from pathlib import Path
import torch
from bounded_cubic import make_model
R=Path(__file__).parent;torch.set_num_threads(8);rows=[]
for seed in [11,22,33]:
 m=make_model('bounded',seed);m.load_state_dict(torch.load(R/f'cubic_results/bounded_{seed}.pt',weights_only=False)['state_dict']);m.capture=True
 g=torch.Generator().manual_seed(123)
 cases={f'random_{scale}':(torch.randn(128,32,8,generator=g)*scale).clamp(-60000,60000) for scale in [0.,1e-7,1e-5,.01,1.,10.,100.,1000.,10000.,60000.]}
 cases.update(positive_60000=torch.full((128,32,8),60000.),negative_60000=torch.full((128,32,8),-60000.),cancel_60000=torch.tensor([60000.,-60000.]*4).expand(128,32,8).clone())
 for case,c in cases.items():
  c.requires_grad_();m.zero_grad(set_to_none=True);out=m.geo(c);loss=out.square().mean();loss.backward()
  bad=[k for k,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
  row=dict(seed=seed,case=case,output_finite=bool(torch.isfinite(out).all()),input_gradient_finite=bool(torch.isfinite(c.grad).all()),nonfinite_parameter_gradients=bad,max_output=float(out.detach().abs().max()),max_input_gradient=float(c.grad.abs().max()),stats=m.diagnostics)
  rows.append(row)
  assert row['output_finite'] and row['input_gradient_finite'] and not bad,row
 print('STRESS_OK',seed,len(cases),flush=True)
(R/'cubic_results/stress_checks.json').write_text(json.dumps(rows,indent=2))
