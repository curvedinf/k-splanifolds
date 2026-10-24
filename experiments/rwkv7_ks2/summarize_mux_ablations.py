"""Summarize four independent fixes and paired training/coding-seed comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).parent
    fresh=json.loads((args.runs/'all_results.json').read_text())
    assert len(fresh)==12
    reference={}
    for mode,file in [('baseline','raw_mux'),('demuxed_history','side_history')]:
        row=json.loads((root/(file+'_results.json')).read_text())
        summary=json.loads((root/(file+'_summary.json')).read_text())
        row['arguments'].update(mode=mode,code_seed=103415,attention_skip=False)
        row['tiled_evaluation']=summary['tiled_evaluation']
        row['origin']='prior completed seed11/code103415 run'
        reference[mode]=row
    manifest=reference['baseline']['data']
    for row in fresh:
        cfg=row['arguments']
        expected=567552 if cfg['mode']=='demuxed_history' else 545184 if cfg['mode']=='output_gains' else 543136
        assert row['parameters']==expected and row['data']==manifest
        assert cfg['steps']==row['history'][-1]['step']==15000 and row['invalid_steps']==0
        assert row['training_targets']==3840000 and row['ffn_types']==['ks2']*4
        assert not cfg['attention_skip'] and row['output_lr_multiplier']==5.5
        assert row['tiled_evaluation']['targets']==1777024
        assert row['tiled_evaluation']['batch4_vs_batch64_max_loss_error']<2e-6
        for name,expected_hash in row['source_hashes'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected_hash,name
    lookup={(r['arguments']['mode'],r['arguments']['seed'],r['arguments']['code_seed']):r
            for r in [*reference.values(),*fresh]}
    baseline=reference['baseline']['tiled_evaluation']['loss']
    def short(row):
        return dict(mode=row['arguments']['mode'],seed=row['arguments']['seed'],code_seed=row['arguments']['code_seed'],
                    parameters=row['parameters'],full_shard_loss=row['tiled_evaluation']['loss'],
                    expanded_val=row['evaluation']['expanded_val'],
                    delta_vs_seed11_raw_baseline=row['tiled_evaluation']['loss']-baseline,
                    benchmark_ms=row['benchmark_ms_per_step'],sustained_ms=row['training_seconds']/15)
    arms=[short(reference['baseline'])]+[short(lookup[(m,11,103415)]) for m in
          ('head_skip','output_gains','persistent_state','source_norm')]+[short(reference['demuxed_history'])]
    robustness={}
    for axis,points in [('training_seeds',[(s,103415) for s in (11,22,33)]),
                        ('coding_seeds',[(11,c) for c in (103415,103416,103417)])]:
        paired=[]
        for seed,code in points:
            raw=lookup[('baseline',seed,code)]['tiled_evaluation']['loss']
            old=lookup[('demuxed_history',seed,code)]['tiled_evaluation']['loss']
            paired.append(dict(seed=seed,code_seed=code,raw_loss=raw,demuxed_history_loss=old,raw_minus_demuxed=raw-old))
        summary={}
        for key in ('raw_loss','demuxed_history_loss','raw_minus_demuxed'):
            values=[r[key] for r in paired]
            summary[key]=dict(mean=stats.mean(values),sample_sd=stats.stdev(values),min=min(values),max=max(values))
        robustness[axis]=dict(paired=paired,summary=summary)
    summary=dict(steps=15000,new_runs=12,reference_runs=2,attention_skip=False,
        independent_arms=arms,robustness=robustness,
        source_probe=json.loads((root/'mux_source_probe.json').read_text()),
        limitations=['Four architectural changes have one seed each; robustness runs replicate the two baselines only.',
                     'Training-seed and coding-seed checks vary one axis at a time, not a full factorial; they share seed11/code103415.',
                     'Alternate code seeds redraw all fixed attention/input/final-output codes for raw and all input/output codes for demuxed history.',
                     'Fresh runs use all four GPUs concurrently; baseline timings came from earlier isolated runs.',
                     'Full-shard losses reuse the same held-out shard at four overlapping offsets, not independent test examples.'])
    (root/'mux_hypotheses_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'mux_hypotheses_results.json').write_text(json.dumps(dict(fresh=fresh,references=reference),indent=2)+'\n')
    (args.runs/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(independent_arms=arms,robustness=robustness),indent=2))


if __name__=='__main__':main()
