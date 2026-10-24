"""Compare trained logits and all surviving gradients before/after dead-output pruning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import run
from history_final_prune import prune_final_side
from study_models_v9 import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    torch.set_num_threads(2)
    result=json.loads((a.run/'completed.json').read_text())
    cfg=dict(result['study_config'],fused_geometry=a.device=='cuda')
    original=build_model(result['arguments']['seed'],cfg)
    original.load_state_dict(torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
    pruned=prune_final_side(copy.deepcopy(original))
    original=original.to(a.device);pruned=pruned.to(a.device)
    if a.device=='cuda':
        from gpu_recurrence import recurrence
        run.recurrence=recurrence
    _,val,_=run.prepare_data(Path(result['arguments']['data']),a.run)
    batch=val[torch.tensor([1000,2000])[:,None]+torch.arange(17)].to(a.device)
    outputs=[];losses=[]
    for model in (original,pruned):
        output=model(batch[:,:-1]);loss=F.cross_entropy(output.flatten(0,1),batch[:,1:].flatten())
        loss.backward();outputs.append(output.detach());losses.append(loss.item())
    torch.testing.assert_close(outputs[0],outputs[1],atol=2e-5,rtol=2e-5)
    assert torch.count_nonzero(original.blocks[-1].ffn.up.weight.grad[96:])==0
    old=dict(original.named_parameters());errors=[];relative=[]
    for name,param in pruned.named_parameters():
        before=old[name].grad;after=param.grad
        if before is None or after is None:
            assert before is None and after is None,name
            continue
        if name=='blocks.3.ffn.up.weight':before=before[:96]
        torch.testing.assert_close(before,after,atol=8e-6,rtol=8e-5,msg=name)
        errors.append((before-after).abs().max().item())
        relative.append(((before-after).double().norm()/before.double().norm().clamp_min(1e-20)).item())
    output=dict(status='PASS',device=a.device,checkpoint_step=result['arguments']['steps'],
                max_logit_error=(outputs[0]-outputs[1]).abs().max().item(),losses=losses,
                max_gradient_absolute_error=max(errors),max_gradient_relative_l2=max(relative),
                original_parameters=sum(p.numel() for p in original.parameters()),pruned_parameters=sum(p.numel() for p in pruned.parameters()),
                original_buffer_bytes=sum(b.numel()*b.element_size() for b in original.buffers()),pruned_buffer_bytes=sum(b.numel()*b.element_size() for b in pruned.buffers()),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                implementation_sha256=hashlib.sha256(Path(__file__).with_name('history_final_prune.py').read_bytes()).hexdigest())
    a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
