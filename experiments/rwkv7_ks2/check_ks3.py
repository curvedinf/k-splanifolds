"""Check KS3 against the three local trained checkpoints and measured GPU kernels.

Run from the repository with its ROCm environment; historical checkpoint runs
and benchmark_sources are required. Writes /tmp/ks3_checkpoint_checks.json.
"""
import sys,json,copy,importlib.util
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo),str(repo/'experiments/rwkv7_ks2')]
import torch
import run
from study_models_v19 import build_model
from ks3 import KS3
from ks3_gpu import KS3Int8
import rank1_int8_tp,ks3_gpu
assert rank1_int8_tp is ks3_gpu
p=repo/'experiments/rwkv7_ks2/gpu_runs/rank1_int8_tp/benchmark_sources/rank1_int8_tp.py'
spec=importlib.util.spec_from_file_location('measured_ks2',p);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
records=[]
for seed in [11,22,33]:
    directory=repo/f'experiments/rwkv7_ks2/gpu_runs/goal_20261022_232213/wave13_selected_long/larger_spine_rank1_fused_repeat_{seed}'
    cfg=json.loads((directory/'completed.json').read_text())['study_config'];cfg['fused_general']=False
    model=build_model(seed,cfg).eval()
    model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
    for layer,block in enumerate(model.blocks):
        legacy=block.ffn
        cell=KS3(96)
        with torch.no_grad():
            cell.down.weight.copy_(legacy.down.weight);cell.up.weight.copy_(legacy.up.weight)
            cell.coefficients.copy_(legacy.geo_spine.reshape(368,6));cell.raw_scale.copy_(legacy.geo_raw_scale.reshape(368));cell.raw_shift.copy_(legacy.geo_raw_shift.reshape(368))
        torch.manual_seed(456+layer)
        x=torch.randn(2,7,96,requires_grad=True)
        actual,expected=cell(x),legacy(x)
        torch.testing.assert_close(actual,expected,atol=3e-4,rtol=4e-5)
        upstream=torch.randn_like(actual)/actual.numel()
        ga=torch.autograd.grad(actual,(x,cell.down.weight,cell.up.weight,cell.coefficients,cell.raw_scale,cell.raw_shift),upstream)
        ge=torch.autograd.grad(expected,(x,legacy.down.weight,legacy.up.weight,legacy.geo_spine,legacy.geo_raw_scale,legacy.geo_raw_shift),upstream)
        for a,e in zip(ga,ge):torch.testing.assert_close(a,e.reshape_as(a),atol=2e-6,rtol=3e-4)
        record=dict(seed=seed,layer=layer,forward_error=float((actual-expected).abs().max().detach()),gradient_error=max(float((a-e.reshape_as(a)).abs().max()) for a,e in zip(ga,ge)))
        cell=cell.cuda();legacy=legacy.cuda();x=x.detach().cuda()
        with torch.inference_mode():
            for precision in ['fp32','fp16','int8']:
                new=KS3Int8(cell,nonlinear=precision);previous=old.Int8Rank1FFN(legacy,nonlinear=precision)
                assert new.state_dict().keys()==previous.state_dict().keys()
                for k,v in new.state_dict().items():assert torch.equal(v,previous.state_dict()[k]),k
                torch.testing.assert_close(new(x),previous(x),atol=0,rtol=0)
            assert KS3Int8(cell).nonlinear==1
            for rank in range(4):
                new=KS3Int8(cell,rank=rank,world=4);previous=old.Int8Rank1FFN(legacy,rank=rank,world=4,nonlinear='fp16')
                for k,v in new.state_dict().items():assert torch.equal(v,previous.state_dict()[k]),(rank,k)
        records.append(record)
        print(record,flush=True)
Path('/tmp/ks3_checkpoint_checks.json').write_text(json.dumps(dict(status='PASS',checks=records,quantized_outputs='exact in all three modes',tp4_weights='exact on all four ranks'),indent=2)+'\n')
