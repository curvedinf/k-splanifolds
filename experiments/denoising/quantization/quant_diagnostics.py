import json
from pathlib import Path
import torch
import run as b
from rank_models import make_model
from quant_ablation import QuantKS2
R=Path(__file__).parent;torch.set_num_threads(8)
s=json.loads((R/'results/splits.json').read_text());v=b.patches([R/p for p in s['validation']],4096,456)
g=torch.Generator().manual_seed(789);x=(v+torch.randn(v.shape,generator=g)*b.SIGMA-.5)/.2;rows=[]
with torch.no_grad():
 for seed in [11,22,33]:
  m=make_model('R8S32',seed);m.load_state_dict(torch.load(R/f'rank_results/R8S32_seed{seed}.pt',weights_only=False)['state_dict']);q=QuantKS2(m,'fp16')
  c=m.down(x).reshape(-1,32,8);h=c.half();total=h.sum(-1,keepdim=True);near=total.abs()<1e-6;w=torch.where(near,torch.full_like(h,.125),h/torch.where(near,torch.ones_like(total),total));t=total/8
  den=1+q.dd.half()*(h-t).abs()[...,None]+q.dw.half()*w.square()[...,None]
  rows.append(dict(seed=seed,charts=total.numel(),near_zero_branch_changes=int(((c.sum(-1,keepdim=True).abs()<1e-6)!=near).sum()),half_weight_nonfinite=int((~torch.isfinite(w)).sum()),half_weight_square_nonfinite_fraction=float((~torch.isfinite(w.square())).float().mean()),half_denominator_nonfinite_fraction=float((~torch.isfinite(den)).float().mean()),max_abs_chart_coordinate=float(c.abs().max())))
(R/'quant_results/numerical_diagnostics.json').write_text(json.dumps(rows,indent=2));print(rows)
