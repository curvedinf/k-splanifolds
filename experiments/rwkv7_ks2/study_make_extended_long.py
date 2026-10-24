"""Record an explicit three-seed KS2 selection and prepare paired 100k runs.

This writes reviewable manifests; it never launches training or reads holdout data.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from study_fusion import with_fused_geometry


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--rationale', type=Path, required=True,
                   help='JSON mapping every selected model to its reason and estimated fused ms/step')
    a = p.parse_args()
    assert 2 <= len(a.models) <= 4 and len(set(a.models)) == len(a.models)
    state = json.loads((a.root / 'study_state.json').read_text())
    ranking = json.loads((a.root / 'extended_candidate_ranking.json').read_text())
    rationale = json.loads(a.rationale.read_text())
    assert set(rationale) == set(a.models)
    assert not (a.root / 'training_complete.json').exists()
    assert not (a.root / 'final_checks').exists()
    assert time.time() < state['deadline_epoch']
    assert all(c['seeds_complete'] == 3 for c in ranking['candidates']), 'Refresh ranking after all screens finish'
    by_name = {c['model']: c for c in ranking['candidates']}
    selected = [by_name[name] for name in a.models]
    assert len({c['equivalent_architecture'] for c in selected})==len(selected), 'Do not select equivalent implementations as separate architectures'
    assert all(c['eligible'] and c['seeds_complete'] == 3 for c in selected)
    assert {c['mechanism'] for c in selected} == {'conditioning_rank_placement', 'fixed_geometry_adaptation'}
    for wave in ('wave11_conditioning', 'wave12_fixed_geometry', 'wave12b_rank_freezing', 'wave12c_original_geometry'):
        spec = json.loads((a.root / (wave + '_manifest.json')).read_text())
        assert all((Path(spec['out']) / job['id'] / 'completed.json').exists() for job in spec['jobs']), wave
    configs = {c['model']: with_fused_geometry(c['config']) for c in selected}
    if any(c.get('fused_general') for c in configs.values()):
        check = json.loads((a.root / 'compact_code_checks/general_geometry_check.json').read_text())
        assert check['status'] == 'PASS'
        assert check['source_sha256'] == hashlib.sha256(Path(__file__).with_name('ks2_general_gpu.py').read_bytes()).hexdigest()
    wave = 'wave13_selected_long'
    manifest = a.root / (wave + '_manifest.json')
    assert not manifest.exists() and not (a.root / wave).exists()
    jobs = []
    finish = [time.time()] * 4
    schedule = []
    for seed in (11, 22, 33):
        for pool in ('original', 'larger'):
            for name in a.models:
                job = dict(id=f'{pool}_{name}_{seed}', seed=seed, steps=100000, config=configs[name])
                if pool == 'larger':
                    job['data'] = str(a.root / 'data10m')
                jobs.append(job)
                ms = float(rationale[name]['estimated_fused_step_ms'])
                assert ms > 0 and rationale[name]['reason']
                gpu = min(range(4), key=lambda i: finish[i])
                start = finish[gpu]
                finish[gpu] += ms * 100 + 90
                schedule.append(dict(job=job['id'], estimated_gpu=gpu, estimated_start=start, estimated_finish=finish[gpu]))
    assert max(x['estimated_start'] for x in schedule) < state['deadline_epoch'], 'Reduce selected families: estimated last dispatch exceeds deadline'
    selection = dict(recorded_epoch=time.time(), selected=a.models, rationale=rationale,
                     all_candidates=ranking['candidates'], fused_configs=configs,
                     protocol='100k fresh cosine schedule, original and larger training pools, seeds11/22/33. Compare existing exact original MLP100k checkpoints. No MLP changes. Selection used exploratory three-seed means and mechanistic value; reserved interval remains sealed.',
                     runtime_estimate=dict(note='Greedy four-card estimate from declared fused step times plus90seconds per job for setup and evaluations; approximate, not a dispatch guarantee.', schedule=schedule))
    write(a.root / 'extended_long_selection.json', selection)
    write(manifest, dict(out=str(a.root / wave), deadline_epoch=state['deadline_epoch'], jobs=jobs))
    holdout_path = a.root / 'extended_holdout_plan.json'
    holdout = json.loads(holdout_path.read_text())
    assert manifest.name not in holdout['manifests']
    holdout['manifests'].append(manifest.name)
    write(holdout_path, holdout)
    speed_path = a.root / 'additional_speed_configs.json'
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else dict(configs=[])
    speed['configs'] += [dict(cfg, name='selected_' + name) for name, cfg in configs.items()]
    speed['configs'].append(dict(name='frozen_spine46_skip_gradients', family='dense_ks2', charts=46,
                                  cell='ks2_relu2', geometry_terms='spine', frozen_geometry=True, fused_general=True))
    assert len({c['name'] for c in speed['configs']}) == len(speed['configs'])
    write(speed_path, speed)
    state['waves'].append(dict(name=wave, status='prepared', manifest=manifest.name,
                               reason='Explicit selection recorded in extended_long_selection.json; all three seeds on both pools, unchanged original MLP reference.'))
    state['updated_epoch'] = time.time()
    write(a.root / 'study_state.json', state)
    print(json.dumps(dict(manifest=str(manifest), jobs=len(jobs), selected=a.models,
                          estimated_last_start=max(x['estimated_start'] for x in schedule),
                          estimated_finish=max(finish)), indent=2))


if __name__ == '__main__':
    main()
