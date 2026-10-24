"""Configurable study worker, retaining original paired sampling/optimizer protocol."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
import run,study_train_core
from study_models_v3 import build_model,make_optimizer,parameter_breakdown

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,required=True)
 args,remaining=p.parse_known_args();cfg=json.loads(args.config.read_text())
 out=Path(remaining[remaining.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
 if (out/'results.json').exists():raise RuntimeError('Use fresh directory')
 root=Path(__file__).resolve().parent;snapshot=out/'sources';snapshot.mkdir(exist_ok=True);sources={}
 for name in ('study_models.py','study_models_v2.py','study_models_v3.py','study_train.py','study_train_v3.py','study_train_core.py','study_full_eval.py','study_full_eval_v3.py','study_wave_v2.py','study_wave_v3.py','ks2_geometry_gpu.py','side_stream.py','side_history.py','spread_projection.py','run.py','gpu_recurrence.py','../common/ks2.py','../common/ks2_stabilized.py'):
  data=(root/name).read_bytes();dest=snapshot/name.replace('../','parent_');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
 run.Model=lambda seed,variant,**kw:build_model(seed,cfg);run.make_optimizer=make_optimizer
 sys.argv=[sys.argv[0],*remaining];study_train_core.main()
 path=out/'results.json'
 if not path.exists():return
 r=json.loads(path.read_text());a=r['arguments'];model=build_model(a['seed'],cfg).cuda().eval()
 checkpoint=torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=True)
 model.load_state_dict(checkpoint['model']);assert checkpoint['step']==a['steps']
 assert all(int(s['step'].item())==a['steps'] for s in checkpoint['optimizer']['state'].values())
 assert all(torch.isfinite(p).all() for p in model.parameters());del checkpoint
 train,val,_=run.prepare_data(Path(a['data']),out);length=a['sequence_length'];batch_size=a['batch_size']
 static=torch.zeros(batch_size,length+1,dtype=torch.long,device='cuda')
 with torch.no_grad():
  model(static[:,:-1]);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):loss=F.cross_entropy(model(static[:,:-1]).flatten(0,1),static[:,1:].flatten())
 r['evaluation']={}
 for name,source,seed,targets in [('original_val',val,12345,32*batch_size*length),('expanded_val',val,54321,65536),('sampled_train',train,98765,65536)]:
  count=max(1,targets//(length*batch_size));positions=torch.randint(len(source)-length-1,(count,batch_size),generator=torch.Generator().manual_seed(seed))
  batches=source[positions[:,:,None]+torch.arange(length+1)].cuda();losses=[]
  for batch in batches:static.copy_(batch);graph.replay();losses.append(loss.clone())
  r['evaluation'][name]=torch.stack(losses).mean().item()
 assert abs(r['evaluation']['original_val']-r['history'][-1]['val_loss'])<2e-6
 r['ffn_types']=[cfg.get('cell','ks2') if hasattr(b,'ffn') else 'mlp' for b in model.blocks]
 r.update(study_config=model.study_config,parameter_breakdown=parameter_breakdown(model),source_hashes=sources,
          architecture='Timed KS2 study: '+cfg['name'],coding='fixed_both')
 path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(evaluation=r['evaluation'])),flush=True)
if __name__=='__main__':main()
