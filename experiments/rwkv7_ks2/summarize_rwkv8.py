"""Record paired ROSA hybrid results and the controller-rank tradeoff."""
import argparse
import json
from pathlib import Path
from rwkv8_rosa import MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    results=json.loads((args.runs/'all_results.json').read_text())
    lookup={r['arguments']['rwkv8_mode']:r for r in results}
    assert set(lookup)==set(MODES)
    base=lookup['baseline'];rows={}
    for mode in MODES:
        r=lookup[mode]
        assert r['data']==base['data'] and r['training_targets']==3840000
        assert r['arguments']['steps']==r['history'][-1]['step']==15000 and r['invalid_steps']==0
        assert abs(r['history'][0]['val_loss']-base['history'][0]['val_loss'])<2e-6
        assert r['tiled_evaluation']['targets']==1777024
        rows[mode]=dict(parameters=r['parameters'],parameter_breakdown=r['parameter_breakdown'],
            full_shard_loss=r['tiled_evaluation']['loss'],
            loss_delta_vs_baseline=r['tiled_evaluation']['loss']-base['tiled_evaluation']['loss'],
            benchmark_ms_per_step=r['benchmark_ms_per_step'],sustained_ms_per_step=r['training_seconds']/15,
            evaluation=r['evaluation'],tiled_evaluation=r['tiled_evaluation'],physical_gpu=r['physical_gpu'])
    summary=dict(steps=15000,seed=11,batch_size=4,sequence_length=64,arms=rows,
        rosa_delta_reduced_ranks=rows['hybrid_matched']['full_shard_loss']-rows['reduced_control']['full_shard_loss'],
        rosa_delta_original_ranks=rows['hybrid_full']['full_shard_loss']-rows['baseline']['full_shard_loss'],
        parameter_match_delta=rows['hybrid_matched']['parameters']-rows['baseline']['parameters'],
        architecture='Local width/budget adaptation of BlinkDL RWKV-v8 RWKV7+ROSA parallel hybrid. 96D main, 32 independent binary ROSA streams, original 64x8 KS2 decoders and 2048D side/history. Matched version reduces decay/rate/value ranks to16 and gate rank to40. Full version retains original ranks32/32/32/64.',
        provenance=lookup['hybrid_matched']['rwkv8_protocol'],
        validation='ROSA hard forward matches pinned official suffix-automaton reference, GPU exact bit-flip credits agree with brute force, GPU graph capture passes. Paired unchanged model parameters, matched initial logits, finite training/nonzero ROSA QKV gradients, causality, source hashes, data, optimizer counters and checkpoint reevaluation checked.',
        limits='One training seed per configuration. The official hybrid demo has no ROSA backward; this experiment uses an explicitly documented discrete counterfactual surrogate with Bernoulli credit mapping. This is an adaptation, not an official full-scale RWKV-8 training recipe. Sequence length64 and state resets match earlier runs; results do not measure long-context benefits.')
    for p in (args.runs/'summary.json',root/'rwkv8_summary.json'):
        p.write_text(json.dumps(summary,indent=2)+'\n')
    (root/'rwkv8_results.json').write_text(json.dumps(results,indent=2)+'\n')
    current=json.loads((root/'current_baseline.json').read_text())
    current['latest_completed_repeat']=str((args.runs/'baseline/fixed_both_11/completed.json').resolve().relative_to(root))
    (root/'current_baseline.json').write_text(json.dumps(current,indent=2)+'\n')
    for mode,row in rows.items():print(mode,row['parameters'],row['full_shard_loss'],row['loss_delta_vs_baseline'],row['sustained_ms_per_step'])


if __name__=='__main__':main()
