"""Verify full cohort coverage, launch bounds, controls and reserved-score gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import time


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;state=read(root/'study_state.json');plan=read(root/'final_holdout_plan.json')
    assert read(root/'mlp_optimizer_identity.json')['status']=='PASS'
    expected=[];intervals=[];waves=[]
    for index,name in enumerate(plan['manifests'],1):
        spec=read(root/name);out=Path(spec['out'])
        cpu=read(root/('hadmux_cpu_check.json' if index==1 else f'wave{index}_cpu_check.json'))
        assert cpu['status']=='PASS' and cpu['original_mlp_all_three_seeds_unchanged']
        for job in spec['jobs']:
            d=read(out/job['id']/'completed.json');args=d['arguments']
            assert args['steps']==job['steps'] and args['seed']==job['seed']
            assert d['study_config']==job['config']
            assert args['width']==96 and args['layers']==4
            assert args['batch_size']==4 and args['sequence_length']==64
            assert d['training_targets']==job['steps']*256
            assert d['tiled_evaluation']['targets']==1777024
            if job['config']['family']=='mlp':
                assert set(job['config'])=={'name','family'}
                assert d['ffn_parameters_by_layer']==[73728]*4
            else:
                assert job['config']['family']=='hadmux'
                assert job['config'].get('output_lr')==5.5
                assert job['config']['side_width'] in (2048,8192)
            expected.append(dict(label=out.name+'/'+job['id'],record=d))
        for cfgname in {j['config']['name'] for j in spec['jobs']}:
            assert sorted(j['seed'] for j in spec['jobs'] if j['config']['name']==cfgname)==[11,22,33]
        for gpu in range(4):
            log=out/f'validation_gpu{gpu}/check.log'
            records=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
            assert {r['config'] for r in records if r.get('status')=='PASS'}=={j['config']['name'] for j in spec['jobs']}
        dispatch=root/f'wave{index}_dispatch.jsonl'
        if index==1:dispatch=root/'wave1_dispatch.jsonl'
        lines=[json.loads(s) for s in dispatch.read_text().splitlines() if s.startswith('{')]
        starts={e['job']['id']:e for e in lines if e.get('event')=='start'}
        ends={e['job']:e for e in lines if e.get('event')=='complete'}
        assert set(starts)==set(ends)=={j['id'] for j in spec['jobs']}
        for name,start in starts.items():
            end=ends[name];assert start['gpu']==end['gpu']
            assert state['start_epoch']<=start['time']<state['deadline_epoch']
            assert start['time']<end['time']
            intervals.append(dict(wave=out.name,job=name,gpu=start['gpu'],start=start['time'],end=end['time']))
        waves.append(dict(wave=out.name,runs=len(spec['jobs']),cpu_check='PASS',gpu_checks=[0,1,2,3]))
    for gpu in range(4):
        ordered=sorted([v for v in intervals if v['gpu']==gpu],key=lambda v:v['start'])
        assert ordered
        assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:]))
    assert len({v['record']['data']['train']['sha256'] for v in expected})==1
    assert len({v['record']['data']['val']['sha256'] for v in expected})==1
    holdout=read(root/'final_checks/holdout_results.json');coverage=read(root/'final_checks/coverage.json')
    assert coverage['started_epoch']>=max(state['deadline_epoch'],read(root/'training_complete.json')['finished_epoch'])
    assert coverage['plan_sha256']==hashlib.sha256((root/'final_holdout_plan.json').read_bytes()).hexdigest()
    labels={v['label'] for v in expected}|{v['label'] for v in plan['prior_references']}
    assert {v['label'] for v in holdout['rows']}==labels and len(holdout['rows'])==len(labels)
    metadata=read(root/'reserved_holdout.json')
    assert holdout['holdout_sha256']==metadata['sha256']
    for v in holdout['rows']:
        assert v['evaluation']['validation_sha256']==metadata['sha256']
        assert all(offset['batch_check']=='PASS' for offset in v['evaluation']['by_offset'])
    speed=read(root/'final_checks/speed_results.json')
    assert all(len(v['values'])==5 for v in speed['summary'])
    assert read(root/'audit.json')['status']=='PASS'
    telemetry=[json.loads(s) for s in (root/'gpu_telemetry.jsonl').read_text().splitlines()]
    cards=[]
    for gpu in range(4):
        active=[v for v in intervals if v['gpu']==gpu]
        observed=[t['stats'][f'card{gpu}'] for t in telemetry if t['returncode']==0 and any(v['start']<=t['epoch']<=v['end'] for v in active)]
        samples=[float(t['GPU use (%)']) for t in observed]
        clocks=[int(t['sclk clock speed:'].strip('()').replace('Mhz','')) for t in observed]
        temperatures=[float(t['Temperature (Sensor junction) (C)']) for t in observed]
        cards.append(dict(gpu=gpu,job_count=len(active),job_wall_seconds=sum(v['end']-v['start'] for v in active),sample_count=len(samples),mean_gpu_use=st.mean(samples),median_gpu_use=st.median(samples),clock_mhz_min=min(clocks),clock_mhz_max=max(clocks),clock_mhz_median=st.median(clocks),junction_temperature_max=max(temperatures)))
    result=dict(status='PASS',epoch=time.time(),waves=waves,training_runs=len(expected),reserved_scores=len(labels),all_launches_before_deadline=True,no_overlapping_jobs_per_gpu=True,all_four_gpus_verified=True,unchanged_original_mlp=True,training_intervals=intervals,gpus=cards,
                utilization_note='Utilization samples cover dispatched jobs including setup/evaluation. These are independent runs on four GPUs, not tensor parallel training. Queue tails and cohort-selection waits are included in the raw telemetry.')
    (root/'protocol_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_intervals','gpus')}))


if __name__=='__main__':main()
