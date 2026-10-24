"""Independent CPU checks for new KS2-only conditioning/rank interventions."""
import argparse,copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from study_models_v9 import build_model as old_build
from study_models_v10 import build_model,parameter_breakdown


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(4281)
 tokens=torch.randint(256,(2,8));records=[];configs={j['config']['name']:j['config'] for j in json.loads(a.manifest.read_text())['jobs']}
 for name,cfg in configs.items():
  model=build_model(11,cfg);baseline=old_build(11,dict(family='mlp',name='original_mlp'))
  torch.testing.assert_close(model(tokens),baseline(tokens),atol=0,rtol=0)
  count=parameter_breakdown(model)['total'];assert count<=615936,(name,count)
  for i,b in enumerate(model.blocks):
   for k,v in b.attention.state_dict().items():assert torch.equal(v,baseline.blocks[i].attention.state_dict()[k])
   cell=b.ffn
   assert cell.down.weight.shape[0]==cfg['charts']*cfg.get('subspace_rank',8)
   with torch.no_grad():cell.up.weight.normal_(0,.02)
  expected=None;error=None
  if cfg.get('geometry_basis'):
   original=old_build(11,{k:v for k,v in cfg.items() if k!='geometry_basis'})
   for b,o in zip(model.blocks,original.blocks):
    with torch.no_grad():o.ffn.up.weight.copy_(b.ffn.up.weight)
    x=torch.randn(19,b.ffn.num_subspaces,8)*2;x.requires_grad_(True)
    actual=b.ffn.geo_step(x);target=o.ffn.geo_step(x)
    torch.testing.assert_close(actual,target,atol=2e-5,rtol=3e-5)
    u=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*u).sum(),x,retain_graph=True)[0];gt=torch.autograd.grad((target*u).sum(),x)[0]
    torch.testing.assert_close(ga,gt,atol=4e-5,rtol=5e-5)
   with torch.no_grad():
    actual=model(tokens);target=original(tokens)
    torch.testing.assert_close(actual,target,atol=3e-5,rtol=3e-5);error=float((actual-target).abs().max())
  if cfg.get('subspace_rank'):
   origcfg=dict(cfg,charts=cfg['charts']*cfg['subspace_rank']//8);origcfg.pop('subspace_rank');original=old_build(11,origcfg)
   for b,o in zip(model.blocks,original.blocks):assert torch.equal(b.ffn.down.weight,o.ffn.down.weight)
  if cfg.get('geometry_placement'):
   c=model.blocks[0].ffn;x=torch.randn(2,5,96);coords=c.down(x).reshape(-1,c.num_subspaces,c.subspace_rank).clamp(-65504,65504);h=F.relu(coords).square()
   h=h+c.geo_step(coords if cfg['geometry_placement']=='parallel' else h)
   torch.testing.assert_close(c(x),c.up(h.reshape(2,5,-1)),atol=0,rtol=0)
  loss=F.cross_entropy(model(tokens).flatten(0,1),tokens.roll(-1,1).flatten());loss.backward()
  for k,v in model.named_parameters():
   if k=='blocks.0.attention.value_bias':continue
   assert v.grad is not None and torch.isfinite(v.grad).all(),k
  changed=tokens.clone();changed[:,5:]=(changed[:,5:]+71)%256
  with torch.no_grad():torch.testing.assert_close(model(tokens)[:,:5],model(changed)[:,:5],atol=0,rtol=0)
  row=dict(name=name,status='PASS',parameters=count,equivalent_basis_logit_max_error=error);records.append(row);print(json.dumps(row),flush=True)
 # Backward compatibility includes every exact initialized tensor.
 for cfg in [dict(name='original_mlp',family='mlp'),dict(name='spine',family='dense_ks2',charts=46,cell='ks2_relu2',geometry_terms='spine')]:
  old=old_build(22,cfg);new=build_model(22,cfg)
  assert old.state_dict().keys()==new.state_dict().keys()
  for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
 a.out.write_text(json.dumps(dict(status='PASS',rows=records,checks=['exact zero-readout initial logits and shared attention','all models within original615936 parameter budget','nonzero-readout finite gradients and causality','basis function and coordinate-gradient equivalence','rank projection initialization preserved','independent geometry-placement equations','v9 backward compatibility']),indent=2)+'\n')
if __name__=='__main__':main()
