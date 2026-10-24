"""Audit comparison and final evaluation coverage independently of score ranking."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    expected = {}
    def register(path, phase, label):
        key = str(path.relative_to(root))
        assert key not in expected, key
        expected[key] = dict(phase=phase, label=label)
    spec = read(root / 'wave8_long_manifest.json')
    for job in spec['jobs']:
        register(Path(spec['out']) / job['id'], 'long', job['id'])
    spec = read(root / 'wave9_geometry_learning_manifest.json')
    for job in spec['jobs']:
        register(Path(spec['out']) / job['id'], 'short', job['id'])
    for seed in (11, 22, 33):
        register(root / (f'initial/mlp/fixed_both_{seed}' if seed != 33 else 'wave3/mlp_seed33'),
                 'short', f'original_mlp_{seed}')
        register(root / ('wave4_activation/spine_frozen' if seed == 11 else f'wave6_budget/spine_frozen_{seed}'),
                 'short', f'frozen_spine_{seed}')
        register(root / f'wave4_activation/spine_fused_{seed}', 'short', f'learned_spine_{seed}')
    for name in read(root / 'extended_holdout_plan.json')['manifests']:
        spec = read(root / name)
        for job in spec['jobs']:
            path = Path(spec['out']) / job['id']
            phase = 'extended_long' if job.get('steps', 15000) >= 100000 else 'extended_short'
            register(path, phase, path.parent.name + '/' + job['id'])
    pending = [key for key in expected if not (root / key / 'completed.json').exists()]
    pairs = []
    for path in sorted(path for wave in ('wave13_selected_long','wave16_combination_long') for path in (root / wave).glob('*/completed.json')):
        run = read(path)
        pool = 'larger' if run['data']['train']['bytes'] > 10000000 else 'original'
        seed = run['arguments']['seed']
        control = read(root / f'wave8_long/{pool}_original_mlp_{seed}/completed.json')
        for field in ('steps', 'batch_size', 'sequence_length', 'width', 'layers', 'seed'):
            assert run['arguments'][field] == control['arguments'][field], (path, field)
        for field in ('precision', 'training_targets'):
            assert run[field] == control[field], (path, field)
        for split in ('train', 'val'):
            assert run['data'][split]['sha256'] == control['data'][split]['sha256'], (path, split)
        for field in ('targets', 'sequence_length', 'validation_sha256', 'evaluation_step'):
            assert run['tiled_evaluation'][field] == control['tiled_evaluation'][field], (path, field)
        assert control['study_config']['family'] == 'mlp'
        assert control['parameters'] == 615936
        assert {r['evaluation_step'] for r in run['intermediate_evaluations']} == {15000, 50000}
        pairs.append(dict(run=str(path.parent.relative_to(root)), control=control['study_job_id'], status='PASS'))
    final = root / 'final_checks'
    final_result = None
    if (final / 'complete.json').exists():
        assert not pending
        assert (root / 'training_complete.json').exists()
        deadline = read(root / 'study_state.json')['deadline_epoch']
        assert final.stat().st_ctime >= deadline, 'Final evaluation directory predates exploration deadline'
        result = read(final / 'holdout_results.json')
        assert not result['missing_training_jobs']
        reserved = read(root / result.get('holdout_manifest', 'reserved_holdout.json'))
        assert result['holdout_sha256'] == reserved['sha256']
        actual = {r['run']: r for r in result['rows']}
        assert len(actual) == len(result['rows']), 'Duplicate final evaluations'
        assert set(actual) == set(expected), (set(expected) - set(actual), set(actual) - set(expected))
        target_counts = {}
        for key, value in actual.items():
            assert value['phase'] == expected[key]['phase']
            assert value['job'] == expected[key]['label']
            evaluation = value['evaluation']
            assert evaluation['validation_sha256'] == reserved['sha256']
            assert math.isfinite(evaluation['loss'])
            trained = read(root / key / 'completed.json')
            assert evaluation['evaluation_step'] == trained['arguments']['steps']
            length = evaluation['sequence_length']
            offsets = [0, length // 4, length // 2, 3 * length // 4]
            targets = sum(((reserved['bytes'] - offset - 1) // length) * length for offset in offsets)
            assert evaluation['targets'] == targets, (key, targets, evaluation['targets'])
            assert [row['offset'] for row in evaluation['by_offset']] == offsets
            for name, digest in evaluation.get('evaluation_source_sha256', {}).items():
                assert hashlib.sha256((root / key / 'evaluation_sources' / name).read_bytes()).hexdigest() == digest
            target_counts[length] = targets
        speed = read(final / 'speed_results.json')
        base = {'original_mlp', 'frozen_spine46', 'history_pruned_fused'}
        base |= {f'{model}_{backend}' for model in ('history', 'full31', 'spine46', 'undamped33') for backend in ('eager', 'fused')}
        extra = root / 'additional_speed_configs.json'
        if extra.exists():
            base |= {cfg['name'] for cfg in read(extra)['configs']}
        assert {r['model'] for r in speed['summary']} == base
        assert len(speed['rows']) == 5 * len(base)
        for model in base:
            values = [r for r in speed['rows'] if r['model'] == model]
            assert {r['repeat'] for r in values} == set(range(5))
            assert len(values) == 5 and all(r['gpu'] == 0 and math.isfinite(r['ms']) and r['ms'] > 0 for r in values)
        final_result = dict(status='PASS', heldout_runs=len(actual), target_counts=target_counts,
                            speed_configurations=len(base), speed_rounds=5)
    expected_long_pairs=sum(len(read(root/(wave+'_manifest.json'))['jobs']) for wave in ('wave13_selected_long','wave16_combination_long') if (root/(wave+'_manifest.json')).exists())
    result = dict(status='PASS' if final_result and len(pairs) == expected_long_pairs else 'PENDING',
                  epoch=time.time(), expected_holdout_runs=len(expected), pending_training=pending,
                  matched_long_pairs=pairs, final_evaluation=final_result,
                  limits='This checks artifact coverage, data/target matching and final timing repetitions. '
                         'Source/checkpoint contents are checked separately by study_audit.py; '
                         'initial control invariance and optimizer settings by original_reference_check.json. '
                         'It does not infer model quality from pass/fail.')
    (root / 'protocol_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], expected_holdout=len(expected),
                          pending_training=len(pending), matched_long_pairs=len(pairs))))


if __name__ == '__main__':
    main()
