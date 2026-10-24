"""Replicate selected Hadamard combinations at a common longer horizon."""
import argparse
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).resolve().parent
    deadline=json.loads((root/'study_state.json').read_text())['deadline_epoch']
    manifest=root/'wave2_scaling_manifest.json'
    print(json.dumps(dict(event='waiting_for_complete_scaling_cohorts',epoch=time.time())),flush=True)
    while True:
        if time.time()>=deadline:raise RuntimeError('No remaining exploration time')
        if manifest.exists():
            spec=json.loads(manifest.read_text());paths=[Path(spec['out'])/j['id'] for j in spec['jobs']]
            assert not any((path/'failure.json').exists() for path in paths),'Resolve scaling failure before selection'
            if all((path/'completed.json').exists() for path in paths):break
        time.sleep(5)
    shutil.copyfile(root/'hadmux_cpu_check_v2.json',root/'wave2_cpu_check.json')
    records=[json.loads((path/'completed.json').read_text()) for path in paths]
    groups=[]
    for name in sorted({r['study_config']['name'] for r in records}):
        cohort=[r for r in records if r['study_config']['name']==name]
        assert sorted(r['arguments']['seed'] for r in cohort)==[11,22,33]
        groups.append(dict(name=name,config=cohort[0]['study_config'],
                           mean=st.mean(r['tiled_evaluation']['loss'] for r in cohort),
                           mean_ms=st.mean(r['training_seconds']*1000/15000 for r in cohort),
                           values=[r['tiled_evaluation']['loss'] for r in cohort]))
    groups.sort(key=lambda g:g['mean'])
    scaled=min((g for g in groups if g['config']['family']=='hadmux'),key=lambda g:g['mean'])
    matched=next(g for g in groups if g['name']=='scale_r1_matched_s2048')
    if scaled['name']==matched['name']:
        scaled=min((g for g in groups if g['config']['family']=='hadmux' and g['config']['geometry_rank']==8),key=lambda g:g['mean'])
    baseline_records=[json.loads(p.read_text()) for p in (root/'wave1_transfer').glob('had_full_r8_*/completed.json')]
    baseline_records=[r for r in baseline_records if r['study_config']['name']=='had_full_r8']
    assert len(baseline_records)==3
    baseline=dict(name='had_full_r8',config=baseline_records[0]['study_config'],mean=st.mean(r['tiled_evaluation']['loss'] for r in baseline_records),
                  mean_ms=st.mean(r['training_seconds']*1000/15000 for r in baseline_records))
    mlp=next(g for g in groups if g['config']['family']=='mlp')
    candidates=[baseline,scaled,matched,mlp]
    # User requested a combined learned-size × step-time cost axis. Include
    # its lowest-cost Hadamard frontier point and the best small learned
    # Hadamard spine, selected only from complete three-seed transfer results.
    transfer=[json.loads(p.read_text()) for p in (root/'wave1_transfer').glob('*/completed.json')]
    for name in ('had_spine_r8_frozen','had_spine_r4_relu2'):
        cohort=[r for r in transfer if r['study_config']['name']==name]
        assert sorted(r['arguments']['seed'] for r in cohort)==[11,22,33]
        candidates.append(dict(name=name,config=cohort[0]['study_config'],
            mean=st.mean(r['tiled_evaluation']['loss'] for r in cohort),
            mean_ms=st.mean(r['training_seconds']*1000/15000 for r in cohort)))
    remaining=deadline-time.time()
    forecasts=[]
    for steps in (100000,75000,50000,30000,25000):
        clocks=[45.]*4
        for seed in (11,22,33):
            # Put the largest measured case first within each paired cohort.
            for g in sorted(candidates,key=lambda g:g['mean_ms'],reverse=True):
                gpu=min(range(4),key=lambda i:clocks[i])
                clocks[gpu]+=1.15*g['mean_ms']*steps/1000+60
        forecasts.append(dict(steps=steps,predicted_seconds=max(clocks),available_seconds=remaining))
    feasible=[f for f in forecasts if f['predicted_seconds']<=remaining-60]
    assert feasible,'Insufficient time for all six arms at the revised 25k minimum; do not silently drop cohorts'
    chosen=feasible[0];steps=chosen['steps']
    selection=dict(epoch=time.time(),criterion='Original Hadamard baseline, best complete scaled mean, matched-budget scaled R1, and unchanged original MLP. If matched R1 is itself best, include the best scaled R8 candidate as the other family. Also include frozen R8 and learned R4 spine to probe the user-requested combined size-times-speed frontier. Three seeds for every arm.',
                   all_scaling_groups=groups,candidates=candidates,steps=steps,forecasts=forecasts,
                   control='Original MLP architecture/initialization/optimizer, no tuning. All new long arms share the chosen cosine horizon. Historical 15k/100k references remain separately reported.',
                   timing='Horizon chosen before long training, from complete short-run times and remaining user window, with15%training-time margin+60sperjob+45svalidation. Final actual completion and utilization are audited separately.')
    (root/'long_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    jobs=[dict(id=f'long_{g["name"]}_{seed}',config=dict(g['config'],name='long_'+g['name']),seed=seed,steps=steps)
          for seed in (11,22,33) for g in sorted(candidates,key=lambda g:g['mean_ms'],reverse=True)]
    next_manifest=root/'wave3_long_manifest.json'
    next_manifest.write_text(json.dumps(dict(out=str(root/'wave3_long'),deadline_epoch=deadline,jobs=jobs),indent=2)+'\n')
    with (root/'wave3_cpu_check.log').open('w') as log:
        subprocess.run([sys.executable,str(code/'check_hadmux_v2.py'),'--manifest',str(next_manifest),'--cpu'],stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(root/'hadmux_cpu_check_v2.json',root/'wave3_cpu_check.json')
    print(json.dumps(dict(event='long_selected',epoch=time.time(),steps=steps,candidates=[g['name'] for g in candidates],predicted_seconds=chosen['predicted_seconds'])),flush=True)
    with (root/'wave3_dispatch.jsonl').open('w') as log:
        subprocess.run([sys.executable,str(code/'hadmux_wave_v3.py'),'--manifest',str(next_manifest),
                        '--requires-check',str(root.parent/'goal_20261022_232213/compact_code_checks/general_geometry_check.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(event='long_wave_finished',epoch=time.time())),flush=True)


if __name__=='__main__':main()
