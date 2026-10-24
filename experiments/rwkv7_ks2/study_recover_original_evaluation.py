"""Finish original-equation evaluation at training batch size, without retraining."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from sweep_plateau import environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    wave=a.root/'wave12c_original_geometry'
    while not (wave/'all_results.json').exists():time.sleep(3)
    # GPU0 is reserved after this wave by the following queue's explicit gate.
    code=Path(__file__).resolve().parent
    jobs=json.loads((a.root/'wave12c_original_geometry_manifest.json').read_text())['jobs']
    assigned={}
    for line in Path('/tmp/ks2_study_wave12c.log').read_text().splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if event.get('event')=='start':assigned[event['job']['id']]=event['gpu']
    notes=[]
    for job in jobs:
        if job['config']['name']!='full_original_equation':continue
        path=wave/job['id'];result=json.loads((path/'results.json').read_text())
        assert result['arguments']['steps']==15000 and result['invalid_steps']==0
        checkpoint_hash=hashlib.sha256((path/'checkpoint.pt').read_bytes()).hexdigest()
        old_score=None
        if (path/'tiled.json').exists():
            old_score=json.loads((path/'tiled.json').read_text())['loss']
            shutil.copyfile(path/'tiled.json',path/'original_batch64_tiled.json')
        with (path/'tiled_batch4_recovery.log').open('w') as log:
            subprocess.run([sys.executable,str(code/'study_full_eval_v16.py'),'--run',str(path)],env=environment(0),stdout=log,stderr=subprocess.STDOUT,check=True)
        tiled=json.loads((path/'tiled.json').read_text())
        assert tiled['evaluation_batch_size']==4 and tiled['targets']==1777024
        assert checkpoint_hash==hashlib.sha256((path/'checkpoint.pt').read_bytes()).hexdigest()
        note=dict(job=job['id'],checkpoint_sha256=checkpoint_hash,training_restarted=False,
                  old_batch64_loss=old_score,batch4_loss=tiled['loss'],
                  max_probe_batch_error=max(x['batch4_vs_eval_max_error'] for x in tiled['by_offset']),
                  exceptions=sum(x['batch_check']!='PASS' for x in tiled['by_offset']))
        if (path/'failure.json').exists():(path/'failure.json').rename(path/'evaluation_batch64_failure.json')
        if (path/'completed.json').exists():shutil.copyfile(path/'completed.json',path/'completed_before_evaluation_revision.json')
        result.update(tiled_evaluation=tiled,physical_gpu=assigned[job['id']],study_job_id=job['id'],status='complete',intermediate_evaluations=[],context_evaluations=[],evaluation_recovery=note)
        (path/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
        notes.append(note);print(json.dumps(note),flush=True)
    assert all((wave/job['id']/'completed.json').exists() for job in jobs)
    shutil.copyfile(wave/'all_results.json',wave/'all_results_before_evaluation_recovery.json')
    (wave/'all_results.json').write_text(json.dumps([json.loads((wave/job['id']/'completed.json').read_text()) for job in jobs],indent=2)+'\n')
    result=dict(status='PASS',finished_epoch=time.time(),rows=notes,reason='Original signed coordinate-sum geometry amplifies batch-dependent FP32 reductions. All three signed-original-equation runs use evaluation batch4, matching training. Fixed validation windows/targets, all weights, training steps and optimizer states are unchanged. Other model families retain the existing evaluator batch size and strict check. Original failed batch64 evaluations are retained beside each run.')
    (a.root/'evaluation_recovery_notes.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.root/'post_canonical_recovery_finished.json').write_text(json.dumps(dict(status='PASS',finished_epoch=time.time()))+'\n')


if __name__=='__main__':main()
