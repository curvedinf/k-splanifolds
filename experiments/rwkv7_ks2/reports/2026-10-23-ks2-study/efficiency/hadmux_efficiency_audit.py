"""Recompute combined-metric points from recorded runs and timing evidence."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'efficiency';data=json.loads((out/'metrics.json').read_text());points=data['points'];measurements={}
    speed_path=root/'final_checks/speed_results.json'
    speeds=json.loads(speed_path.read_text()) if speed_path.exists() else None
    reserved_path=root/'final_checks/holdout_results.json'
    reserved=json.loads(reserved_path.read_text()) if reserved_path.exists() else None
    reserved_scores={str(Path(v['run']).resolve()):v['evaluation']['loss'] for v in reserved['rows']} if reserved else {}
    def read(item):
        path=root.parent/item['study']/item['record'];d=json.loads(path.read_text());args=d['arguments']
        value=dict(seed=args['seed'],steps=args['steps'],ffn_parameters_by_layer=d['ffn_parameters_by_layer'],training_seconds=d['training_seconds'],ms=1000*d['training_seconds']/args['steps'],loss=d['tiled_evaluation']['loss'],record_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        measurements[item['study']+'/'+item['record']]=value
        return value
    for point in points:
        raw=[dict(read(item)) for item in point['sources']];refs=[dict(read(item)) for item in point['references']]
        if point['quality_mode']=='reserved':
            assert point['validation_sha']==reserved['holdout_sha256']
            for items,values in ((point['sources'],raw),(point['references'],refs)):
                for item,value in zip(items,values):
                    key=str((root.parent/item['study']/item['record']).parent.resolve())
                    value['loss']=reserved_scores[key]
        seeds=sorted({r['seed'] for r in raw});assert seeds==point['seeds'] and len(seeds)==point['n']
        losses=[];times=[];parameters=[]
        for seed in seeds:
            cohort=[r for r in raw if r['seed']==seed]
            losses.append(st.mean(r['loss'] for r in cohort));times.append(st.mean(r['ms'] for r in cohort));parameters.append(st.mean(st.mean(r['ffn_parameters_by_layer']) for r in cohort))
        close(point['loss'],st.mean(losses));close(point['mlp_loss'],st.mean(r['loss'] for r in refs))
        close(point['mlp_parameters'],st.mean(st.mean(r['ffn_parameters_by_layer']) for r in refs))
        expected_params=st.mean(parameters)
        if point['derived_pruning']:
            checks=[json.loads((root/path).read_text()) for path in point['prune_evidence']]
            assert len(checks)==3 and all(v['status']=='PASS' for v in checks)
            expected_params-=st.mean(v['original_parameters']-v['pruned_parameters'] for v in checks)/4
            parent=points[point['parent_id']];close(parent['loss'],point['loss'])
        close(point['parameters'],expected_params)
        if point['timing_mode']=='worker':
            close(point['ms'],st.mean(times));close(point['mlp_ms'],st.mean(r['ms'] for r in refs))
        else:
            assert speeds is not None
            timing=next(v for v in speeds['summary'] if v['model']==point['speed_model'])
            mlp=next(v for v in speeds['summary'] if v['model']=='original_mlp_repeat')
            assert len(timing['values'])==5
            close(point['ms'],st.median(timing['values']));close(point['mlp_ms'],st.median(mlp['values']))
        close(point['cost'],point['parameters']/point['mlp_parameters']*point['ms']/point['mlp_ms'])
    for view in data['views']:
        cohort=[points[i] for i in view['points'] if points[i]['n']>=3]
        assert len({p['timing_mode'] for p in cohort})<=1
        assert len({p['quality_mode'] for p in cohort})<=1
        expected={p['id'] for p in cohort if not any(q['cost']<=p['cost'] and q['loss']<=p['loss'] and (q['cost']<p['cost'] or q['loss']<p['loss']) for q in cohort)}
        assert expected==set(view['frontier'])
        for p in cohort:
            if p['name']=='Original MLP':close(p['cost'],1.)
        for threshold in view['quality_tolerance_winners']:
            eligible=[p for p in cohort if p['loss_gap']<=threshold['max_excess_loss']]
            close(points[threshold['best_id']]['cost'],min(p['cost'] for p in eligible))
    csvrows=list(csv.DictReader((out/'metrics.csv').open()));assert len(csvrows)==len(points)
    for row,point in zip(csvrows,points):
        assert int(row['id'])==point['id'];close(float(row['cost']),point['cost']);close(float(row['loss']),point['loss'])
    (out/'run_measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
    result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),points=len(points),unique_runs=len(measurements),views=len(data['views']),costs_recomputed_from_source_records=True,frontiers_recomputed=True,primary_mlp_cost_exactly_one=True,csv_matches_json=True,derived_pruning_points=sum(p['derived_pruning'] for p in points),training_still_running=not(root/'training_complete.json').exists(),generator_source_sha256=hashlib.sha256(Path(__file__).with_name('hadmux_efficiency.py').read_bytes()).hexdigest(),auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'snapshot_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
