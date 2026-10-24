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
from diagnose_general_fp64 import evaluate as evaluate_fp64


def precise_reference(cell,x,upstream,basis,terms):
 names=[n for n,p in cell.named_parameters(recurse=False) if p.requires_grad]
 params={n:p.detach().cpu().double().requires_grad_(n in names) for n,p in cell.named_parameters(recurse=False)}
 inputs=x.detach().cpu().double().requires_grad_()
 output=evaluate_fp64(inputs,params,basis,terms)
 gradients=torch.autograd.grad(output,(inputs,*(params[n] for n in names)),grad_outputs=upstream.detach().cpu().double())
 return output.detach(),gradients


def relative_error(value,reference):
 return (value.detach().cpu().double()-reference).norm().item()/max(reference.norm().item(),1e-30)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(8123);run.recurrence=recurrence;rows=[];stress_roundoff=[]
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
      precise=None
      try:torch.testing.assert_close(actual,target,atol=5e-5,rtol=4e-4)
      except AssertionError as error:
       if scenario!='clamp':raise
       precise=precise_reference(ref,x,u,basis,terms)
       eager_error=relative_error(target,precise[0]);fused_error=relative_error(actual,precise[0])
       assert max(eager_error,fused_error)<5e-5,(rank,basis,terms,scenario,eager_error,fused_error)
       stress_roundoff.append(dict(rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,quantity='forward',eager_fp64_relative_l2=eager_error,fused_fp64_relative_l2=fused_error,strict_elementwise_failure=str(error)))
      expected=torch.autograd.grad((target*u).sum(),(x,*pr));measured=torch.autograd.grad((actual*u).sum(),(y,*pf))
      errors=[]
      for i,(g,h) in enumerate(zip(expected,measured)):
       try:torch.testing.assert_close(h,g,atol=7e-5,rtol=5e-4)
       except AssertionError as error:
        if scenario!='tiny' or i!=0:raise
        if precise is None:precise=precise_reference(ref,x,u,basis,terms)
        eager_error=relative_error(g,precise[1][i]);fused_error=relative_error(h,precise[1][i])
        assert max(eager_error,fused_error)<1e-5,(rank,basis,terms,scenario,eager_error,fused_error)
        stress_roundoff.append(dict(rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,quantity='input_gradient',eager_fp64_relative_l2=eager_error,fused_fp64_relative_l2=fused_error,strict_elementwise_failure=str(error)))
       errors.append(dict(index=i,max_absolute=(h-g).abs().max().item(),relative_l2=(h-g).double().norm().item()/max(g.double().norm().item(),1e-30)))
      rows.append(dict(rank=rank,basis=basis,terms=terms,trainability=trainability,scenario=scenario,forward_max_absolute=(actual-target).abs().max().item(),gradient_errors=errors))
    print(json.dumps(dict(rank=rank,basis=basis,terms=terms,status='PASS')),flush=True)
 # Test the exact trained model functions that are eligible for longer runs.
 models=[]
 candidates=['spine_powers','spine_taylor','full_powers','full_taylor','spine_rank1','spine_rank2','spine_rank4','spine_rank16','full_rank2','full_rank4','spine_geo_rms','full_geo_rms','spine_parallel','full_parallel','spine_after_activation','full_after_activation']
 model_cases=[(name,'all') for name in candidates]+[(name,mode) for name in ('full_powers','spine_rank4') for mode in ('warps','frozen')]
 for name,trainability in model_cases:
  path=a.root/'wave11_conditioning'/f'{name}_11';r=json.loads((path/'results.json').read_text());cfg=dict(r['study_config'])
  if trainability=='frozen':cfg['frozen_geometry']=True
  elif trainability=='warps':cfg['geometry_trainable']='warps'
  fused_cfg=with_fused_geometry(cfg)
  state=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
  ref=build_model(11,cfg);ref.load_state_dict(state);ref=ref.cuda()
  fused=build_model(11,fused_cfg);fused.load_state_dict(state);fused=fused.cuda()
  _,val,_=run.prepare_data(Path(r['arguments']['data']),path)
  pos=torch.randint(len(val)-65,(2,),generator=torch.Generator().manual_seed(4918));tokens=val[pos[:,None]+torch.arange(65)].cuda()
  def calculate(model):
   logits=model(tokens[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),tokens[:,1:].flatten());loss.backward()
   return logits.detach(),loss.item(),{n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
  expected,loss,g=calculate(ref);actual,other,h=calculate(fused)
  torch.testing.assert_close(actual,expected,atol=7e-5,rtol=5e-5);assert g.keys()==h.keys()
  errors=[]
  for key in g:
   absolute=(g[key]-h[key]).abs().max().item();relative=(g[key]-h[key]).double().norm().item()/max(g[key].double().norm().item(),1e-30)
   assert relative<2e-4 or absolute<1e-7,(name,key,absolute,relative)
   errors.append(dict(name=key,max_absolute=absolute,relative_l2=relative))
  record=dict(model=name,trainability=trainability,fused_config=fused_cfg,status='PASS',logit_max_error=(actual-expected).abs().max().item(),eager_loss=loss,fused_loss=other,gradient_errors=errors);models.append(record)
  # Use a fresh model for graph capture, avoiding retained graphs across streams.
  capture=build_model(11,fused_cfg);capture.load_state_dict(state);capture=capture.cuda()
  optimizer=torch.optim.AdamW(capture.parameters(),lr=torch.tensor(.001,device='cuda'),capturable=True,foreach=True)
  def update():
   optimizer.zero_grad(set_to_none=False);z=F.cross_entropy(capture(tokens[:,:-1]).flatten(0,1),tokens[:,1:].flatten());z.backward();torch.nn.utils.clip_grad_norm_(capture.parameters(),1.,foreach=True);optimizer.step();return z
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):update()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured_loss=update()
  for _ in range(3):graph.replay()
  torch.cuda.synchronize();assert torch.isfinite(captured_loss)
  assert all(torch.isfinite(p).all() for p in capture.parameters());del graph,capture,optimizer,ref,fused
  print(json.dumps(dict(model=name,trainability=trainability,status='TRAINED_AND_CAPTURE_PASS')),flush=True)
 output=dict(status='PASS',component_cases=len(rows),rows=rows,trained_models=models,stress_roundoff=stress_roundoff,validation_tolerances='Ordinary inputs and all parameter gradients retain original elementwise tolerances. Only clamp-stress forward values and tiny-stress input gradients may use independent CPU float64 autograd, requiring both eager and fused relative L2 errors below5e-5 and1e-5 respectively. Every exception and original strict failure is retained. Trained model logit/gradient checks are unchanged.',source_sha256=hashlib.sha256(Path(__file__).with_name('ks2_general_gpu.py').read_bytes()).hexdigest())
 a.out.write_text(json.dumps(output,indent=2)+'\n')
 # Isolated forward+backward timings include discarded-gradient savings.
 cases=[]
 def add_case(name,cfg,backend):
  model=build_model(11,cfg).cuda();cell=model.blocks[0].ffn
  x=torch.randn(256,cell.num_subspaces,cell.subspace_rank,device='cuda',requires_grad=True)
  params=[p for p in cell.parameters(recurse=False) if p.requires_grad]
  def operation(cell=cell,x=x,params=params):return torch.autograd.grad(cell.geo_step(x).sum(),(x,*params))
  cases.append((name,backend,operation))
 for name in candidates:
  cfg=json.loads((a.root/'wave11_conditioning'/f'{name}_11'/'results.json').read_text())['study_config']
  add_case(name,cfg,'eager');add_case(name,with_fused_geometry(cfg),'fused')
 for terms in ('full','spine'):
  for mode in ('warps','frozen'):
   name=f'{terms}46_{mode}'
   cfg=dict(name=name,family='dense_ks2',charts=46,cell='ks2_relu2')
   if terms=='spine':cfg['geometry_terms']='spine'
   if mode=='warps':cfg['geometry_trainable']='warps'
   else:cfg['frozen_geometry']=True
   old=dict(cfg,**({'fused_partial':True} if terms=='spine' else {'fused_geometry':True}))
   add_case(name,cfg,'eager');add_case(name,old,'previous_fused');add_case(name,with_fused_geometry(cfg),'skip_unused_gradients')
 timings=[];rng=random.Random(8102)
 for repeat in range(3):
  order=cases.copy();rng.shuffle(order)
  for name,backend,operation in order:
   ms=triton.testing.do_bench_cudagraph(operation,rep=100);timings.append(dict(model=name,backend=backend,repeat=repeat,ms=ms))
 summary=[dict(model=n,backend=b,median_ms=statistics.median(r['ms'] for r in timings if r['model']==n and r['backend']==b)) for n,b,_ in cases]
 a.out.with_name('general_geometry_speed.json').write_text(json.dumps(dict(protocol='GPU0,256inputtokens,actualchartcounts; geometryforward+requiredgeometry/inputgradients, no projections/attention/optimizer.Three randomized captured-kernel timing rounds,100ms percase.',rows=timings,summary=summary),indent=2)+'\n')
if __name__=='__main__':main()
