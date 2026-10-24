"""Rank complete combination cohorts and optionally prepare matched long runs."""
import argparse
import json
from pathlib import Path
import statistics as st
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--models', nargs='+')
    p.add_argument('--rationale', type=Path)
    a = p.parse_args()
    root = a.root
    spec = read(root / 'wave15_combinations_manifest.json')
    configs = {job['config']['name']: job['config'] for job in spec['jobs']}
    refs = {seed: read(root / (f'initial/mlp/fixed_both_{seed}/completed.json' if seed != 33 else 'wave3/mlp_seed33/completed.json')) for seed in (11, 22, 33)}
    candidates = []
    for name, cfg in configs.items():
        rows = []
        for seed in (11, 22, 33):
            path = Path(spec['out']) / f'{name}_{seed}' / 'completed.json'
            if not path.exists():
                continue
            result = read(path)
            parent_path = Path(spec['out']) / f'combo_rank1_reference_{seed}' / 'completed.json'
            parent = read(parent_path)['tiled_evaluation']['loss'] if parent_path.exists() else None
            loss = result['tiled_evaluation']['loss']
            rank=cfg.get('subspace_rank',8);charts=cfg['charts']
            fixed=charts*(6*rank+2) if cfg.get('frozen_geometry') else charts*6*rank if cfg.get('geometry_trainable')=='warps' else 0
            rows.append(dict(seed=seed, loss=loss, mlp_loss=refs[seed]['tiled_evaluation']['loss'],
                             rank1_reference_loss=parent, parameters=result['parameters'],
                             learned_per_ffn=result['ffn_parameters_by_layer'][0],fixed_geometry_per_ffn=fixed,
                             total_coefficients_per_ffn=result['ffn_parameters_by_layer'][0]+fixed,
                             observed_ms=result['training_seconds'] * 1000 / 15000,
                             source=str(path)))
        complete = len(rows) == 3
        candidates.append(dict(model=name, config=cfg, seeds=len(rows), eligible=complete and name != 'combo_rank1_reference',
                               mean_loss=st.mean(r['loss'] for r in rows) if complete else None,
                               mean_mlp_gap=st.mean(r['loss'] - r['mlp_loss'] for r in rows) if complete else None,
                               mean_rank1_gap=st.mean(r['loss'] - r['rank1_reference_loss'] for r in rows) if complete and all(r['rank1_reference_loss'] is not None for r in rows) else None,
                               wins_vs_mlp=sum(r['loss'] < r['mlp_loss'] for r in rows), individual=rows))
    candidates.sort(key=lambda c: (not c['eligible'], c['mean_loss'] if c['mean_loss'] is not None else float('inf')))
    write(root / 'combination_ranking.json', dict(recorded_epoch=time.time(), candidates=candidates,
          policy='Three-seed exploratory losses only; original MLP remains primary. Same-wave rank1 replication measures incremental change, not a replacement MLP control. No old or fresh reserved scores are read.'))
    print(json.dumps([dict(model=c['model'], seeds=c['seeds'], loss=c['mean_loss'], gap_vs_mlp=c['mean_mlp_gap'], gap_vs_rank1=c['mean_rank1_gap']) for c in candidates], indent=2))
    if not a.models:
        return
    assert a.rationale and 2 <= len(a.models) <= 4 and len(set(a.models)) == len(a.models)
    assert all(c['seeds'] == 3 for c in candidates), 'Wait for every three-seed screen cohort'
    by_name = {c['model']: c for c in candidates}
    selected = [by_name[name] for name in a.models]
    assert all(c['eligible'] for c in selected)
    reasons = read(a.rationale)
    assert set(reasons) == set(a.models)
    state = read(root / 'study_state.json')
    assert time.time() < state['deadline_epoch']
    assert not (root / 'training_complete.json').exists() and not (root / 'final_checks').exists()
    assert any(c['config'].get('geometry_basis') for c in selected)
    assert any(c['config'].get('frozen_geometry') or c['config'].get('geometry_trainable') for c in selected)
    wave = 'wave16_combination_long'
    manifest = root / (wave + '_manifest.json')
    assert not manifest.exists() and not (root / wave).exists()
    jobs, schedule = [], []
    finish = [time.time()] * 4
    for seed in (11, 22, 33):
        for pool in ('original', 'larger'):
            for name in a.models:
                job = dict(id=f'{pool}_{name}_{seed}', seed=seed, steps=100000, config=configs[name])
                if pool == 'larger':
                    job['data'] = str(root / 'data10m')
                jobs.append(job)
                ms = float(reasons[name]['estimated_ms'])
                assert reasons[name]['reason'] and ms > 0
                gpu = min(range(4), key=lambda i: finish[i])
                start = finish[gpu]
                finish[gpu] += ms * 100 + 90
                schedule.append(dict(job=job['id'], gpu_estimate=gpu, start_estimate=start, finish_estimate=finish[gpu]))
    assert max(item['start_estimate'] for item in schedule) < state['deadline_epoch'], 'Reduce families to fit dispatch deadline'
    write(root / 'combination_long_selection.json', dict(recorded_epoch=time.time(), selected=a.models,
          rationale=reasons, all_candidates=candidates, estimated_schedule=schedule,
          protocol='Fresh100k schedule, original/larger pools, seeds11/22/33, fixed original MLP checkpoints. Comparisons to prior single-intervention KS2 show whether wins accumulate. No reserved-based selection.'))
    write(manifest, dict(out=str(root / wave), deadline_epoch=state['deadline_epoch'], jobs=jobs))
    holdout = read(root / 'extended_holdout_plan.json')
    holdout['manifests'].append(manifest.name)
    write(root / 'extended_holdout_plan.json', holdout)
    speed = read(root / 'additional_speed_configs.json')
    speed['configs'].extend(dict(configs[name], name='selected_' + name) for name in a.models)
    assert len({c['name'] for c in speed['configs']}) == len(speed['configs'])
    write(root / 'additional_speed_configs.json', speed)
    state['waves'].append(dict(name=wave, status='prepared', manifest=manifest.name,
                               reason='User-requested accumulation test; recorded complete-three-seed selection.'))
    state['updated_epoch'] = time.time()
    write(root / 'study_state.json', state)
    print(json.dumps(dict(manifest=str(manifest), jobs=len(jobs), latest_estimated_finish=max(finish))))


if __name__ == '__main__':
    main()
