"""Rank complete replicated KS2 screens; never dispatch or read reserved scores."""
import argparse
import json
import statistics
import time
from pathlib import Path

from study_collect import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    refs = {r['seed']: r for r in rows if (r['wave'], r['job']) in (
        ('initial', 'mlp'), ('wave3', 'mlp_seed33'))}
    assert set(refs) == {11, 22, 33}
    waves = ('wave11_conditioning', 'wave12_fixed_geometry', 'wave12b_rank_freezing', 'wave12c_original_geometry')
    candidates = []
    for wave in waves:
        spec = json.loads((args.root / (wave + '_manifest.json')).read_text())
        names = list(dict.fromkeys(j['config']['name'] for j in spec['jobs']))
        for name in names:
            matched = [r for r in rows if r['wave'] == wave and r['config']['name'] == name]
            assert len({r['seed'] for r in matched}) == len(matched)
            cfg = next(j['config'] for j in spec['jobs'] if j['config']['name'] == name)
            complete = {r['seed'] for r in matched} == {11, 22, 33}
            reference_repeat = name in ('spine_eager_reference', 'full_eager_reference')
            gaps = [r['full_validation_loss'] - refs[r['seed']]['full_validation_loss'] for r in matched]
            candidates.append(dict(
                wave=wave, model=name, config=cfg,
                mechanism='fixed_geometry_adaptation' if cfg.get('frozen_geometry') or cfg.get('geometry_trainable') else 'conditioning_rank_placement',
                eligible=complete and not reference_repeat,
                exclusion='reference repeat' if reference_repeat else None if complete else 'three-seed cohort incomplete',
                seeds_complete=len(matched),
                mean_loss=statistics.mean(r['full_validation_loss'] for r in matched) if complete else None,
                mean_paired_gap=statistics.mean(gaps) if complete else None,
                paired_gap_sd=statistics.stdev(gaps) if complete else None,
                seed_wins=sum(x < 0 for x in gaps) if complete else None,
                learned_parameters=matched[0]['parameters'] if matched else None,
                fixed_buffer_bytes=matched[0]['buffer_bytes'] if matched else None,
                observed_step_ms=statistics.median(r['ms_per_step'] for r in matched) if matched else None,
                individual=[dict(seed=r['seed'], loss=r['full_validation_loss'], reference=refs[r['seed']]['full_validation_loss'], run=r['source']) for r in sorted(matched, key=lambda r: r['seed'])],
            ))
    by_name={c['model']:c for c in candidates}
    for c in candidates:
        c['selection_score']=c['mean_loss']
        c['equivalent_architecture']=c['model'].removesuffix('_fused_repeat')
    for name in ('spine_rank1','spine_rank4'):
        eager=by_name[name];fused=by_name[name+'_fused_repeat']
        if eager['seeds_complete']==fused['seeds_complete']==3:
            pairs=[dict(seed=seed,eager=next(r['loss'] for r in eager['individual'] if r['seed']==seed),fused=next(r['loss'] for r in fused['individual'] if r['seed']==seed)) for seed in (11,22,33)]
            score=statistics.mean((r['eager']+r['fused'])/2 for r in pairs)
            eager.update(eligible=False,exclusion='Equivalent fused backend used for long runs; both cohorts contribute to selection score',selection_score=score)
            fused.update(selection_score=score,equivalent_cohort_pairs=pairs,selection_note='Three seeds with two floating-point implementations per seed, not six independent seeds. Average both implementations; do not select the more favorable trajectory. Fused backend is chosen for runtime after numerical validation.')
    candidates.sort(key=lambda r: (not r['eligible'], r['selection_score'] if r['selection_score'] is not None else float('inf'), r['model']))
    result = dict(recorded_epoch=time.time(), policy='Exploratory full-shard mean over all three seeds. Incomplete cohorts cannot be selected. Reference repeats are excluded. For equivalent eager/fused rank1 andrank4 cohorts, average both per seed and select the validated fused backend for speed. No reserved evaluation is read. Preserve at least one conditioning/rank/placement and one fixed-geometry candidate where useful; use mechanistic value and runtime to choose up to four, documenting every selection and rejection.', candidates=candidates)
    (args.root / 'extended_candidate_ranking.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=sum(r['eligible'] for r in candidates), total=len(candidates), ranked_complete=[dict(model=r['model'], mean=r['mean_loss'], selection_score=r['selection_score'], gap=r['mean_paired_gap'], wins=r['seed_wins']) for r in candidates if r['eligible']]), indent=2))


if __name__ == '__main__':
    main()
