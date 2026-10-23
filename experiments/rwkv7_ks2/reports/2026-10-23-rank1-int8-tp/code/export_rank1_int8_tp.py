"""Save/load the integrated INT8 TP export. No optimizer or FP32 matrix copies."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.distributed as dist
import run
from study_models_v19 import build_model
from rank1_int8_tp import convert,inference_recurrence


def load_export(directory,rank=0,device='cuda',nonlinear='fp16',reducer=None):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    world=meta['tensor_parallel_size']
    if world>1:
        assert dist.is_initialized() and dist.get_world_size()==world
        assert dist.get_rank()==rank
    model=build_model(meta['seed'],meta['study_config'])
    model=convert(model,rank,world,nonlinear,reducer)
    record=meta['shards'][rank];path=directory/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
    run.recurrence=inference_recurrence
    return model.to(device).eval()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tp',type=int,choices=[1,2,4],default=4);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2)
    metadata=json.loads((a.checkpoint_run/'completed.json').read_text());seed=metadata['arguments']['seed'];cfg=metadata['study_config']
    model=build_model(seed,cfg);checkpoint=a.checkpoint_run/'checkpoint.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model'])
    model=model.to(a.device)
    a.out.mkdir(parents=True,exist_ok=False);shards=[]
    for rank in range(a.tp):
        exported=convert(model,rank,a.tp);state={k:v.cpu() for k,v in exported.state_dict().items()};name=f'rank_{rank}.pt';torch.save(state,a.out/name)
        matrices=[(n,b) for n,b in state.items() if n.endswith('qweight')]
        assert matrices and all(v.dtype==torch.int8 for _,v in matrices)
        assert not any(n.endswith(('down.weight','up.weight','head.weight','embedding.weight')) for n in state)
        shards.append(dict(rank=rank,file=name,sha256=hashlib.sha256((a.out/name).read_bytes()).hexdigest(),bytes=(a.out/name).stat().st_size,tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),int8_elements=sum(v.numel() for v in state.values() if v.dtype==torch.int8),float32_elements=sum(v.numel() for v in state.values() if v.dtype==torch.float32),linear_matrices=len(matrices),ffn_tensor_bytes=sum(v.numel()*v.element_size() for n,v in state.items() if '.ffn.' in n)))
    result=dict(format_version=1,seed=seed,study_config=cfg,tensor_parallel_size=a.tp,attention_heads_by_rank=[3//a.tp+int(r<3%a.tp) for r in range(a.tp)],ffn_subspaces_by_rank=[368//a.tp]*a.tp,checkpoint_source=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),trained_parameters_per_ffn=73600,recurrent_state_dtype='float32',default_nonlinear='fp16',nonlinear_options=['fp32','fp16','int8'],quantization='All learned linear maps and embeddings INT8; KS2 coefficients INT8; per-token/group activation quantization; INT32 matrix dots. Scales, warp scalars, normalization, time mixing, residuals and RWKV state remain FP32.',nonlinear_scope='KS2 geometry and ReLU-squared. RWKV controller activation fusion uses FP32 nonlinearities.',code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['rank1_int8_tp.py','rank1_int8_attention_tp.py','export_rank1_int8_tp.py']},shards=shards)
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',shards=len(shards),tensor_bytes=sum(s['tensor_bytes'] for s in shards))))
if __name__=='__main__':main()
