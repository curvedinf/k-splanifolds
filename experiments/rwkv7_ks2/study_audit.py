"""Final artifact audit: steps, finite state, optimizer counters and recorded source versions."""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);rows=[]
    for path in sorted(a.root.rglob('completed.json')):
        d=json.loads(path.read_text());steps=d['arguments']['steps']
        assert d['history'][-1]['step']==steps and d['invalid_steps']==0
        assert math.isfinite(d['tiled_evaluation']['loss'])
        source_count=0
        for name,h in d['source_hashes'].items():
            src=path.parent/'sources'/name.replace('../','parent_')
            assert hashlib.sha256(src.read_bytes()).hexdigest()==h,(path,name)
            source_count+=1
        evaluation_sources=d['tiled_evaluation'].get('evaluation_source_sha256',{})
        for name,h in evaluation_sources.items():
            assert hashlib.sha256((path.parent/'evaluation_sources'/name).read_bytes()).hexdigest()==h,(path,'evaluation',name)
        cp=torch.load(path.parent/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert cp['step']==steps
        for name,tensor in cp['model'].items():
            if tensor.is_floating_point():assert torch.isfinite(tensor).all(),(path,name)
        optsteps=[]
        for item in cp['optimizer']['state'].values():
            optsteps.append(int(item['step'].item()))
            for name,tensor in item.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():assert torch.isfinite(tensor).all(),(path,'optimizer',name)
        assert optsteps and set(optsteps)=={steps},(path,set(optsteps))
        row=dict(run=str(path.parent.relative_to(a.root)),status='PASS',steps=steps,verified_source_files=source_count,
                 verified_evaluation_source_files=len(evaluation_sources),
                 optimizer_parameter_states=len(optsteps),finite_model_and_optimizer=True,
                 checkpoint_sha256=hashlib.sha256((path.parent/'checkpoint.pt').read_bytes()).hexdigest())
        del cp
        intermediate=[]
        for checkpoint in sorted(path.parent.glob('checkpoint_*.pt')):
            expected=int(checkpoint.stem.split('_')[-1])
            cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert cp['step']==expected<steps,(checkpoint,cp['step'])
            assert any(ev['evaluation_step']==expected and math.isfinite(ev['loss']) for ev in d.get('intermediate_evaluations',[])),checkpoint
            for name,tensor in cp['model'].items():
                if tensor.is_floating_point():assert torch.isfinite(tensor).all(),(checkpoint,name)
            for item in cp['optimizer']['state'].values():
                assert int(item['step'].item())==expected,checkpoint
                for name,tensor in item.items():
                    if torch.is_tensor(tensor) and tensor.is_floating_point():assert torch.isfinite(tensor).all(),(checkpoint,name)
            intermediate.append(dict(file=checkpoint.name,step=expected,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),status='PASS'))
            del cp
        row['intermediate_checkpoints']=intermediate
        rows.append(row)
    failures=[dict(path=str(p.relative_to(a.root)),data=json.loads(p.read_text())) for p in a.root.rglob('failure.json')]
    incomplete=[]
    for manifest in a.root.glob('*_manifest.json'):
        spec=json.loads(manifest.read_text())
        if 'jobs' not in spec or 'out' not in spec:continue
        for job in spec['jobs']:
            dest=Path(spec['out'])/job['id']
            legacy=dest/f"fixed_both_{job.get('seed',11)}"/'completed.json'
            if not (dest/'completed.json').exists() and not legacy.exists():
                incomplete.append(dict(manifest=manifest.name,job=job['id'],started=dest.exists()))
    result=dict(epoch=time.time(),status='PASS' if not failures and not incomplete else 'CHECK_INCOMPLETE',
                completed=len(rows),runs=rows,failures=failures,incomplete_jobs=incomplete,
                source_snapshot_recovery='See recovered_source_snapshots.json; recovered files matched hashes recorded during training.')
    (a.root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],completed=len(rows),failures=len(failures),incomplete=len(incomplete))))


if __name__=='__main__':main()
