"""Summarize the demuxed-history RWKV projection ablation and validation."""
import argparse
import json
from pathlib import Path
from rwkv_mux import MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    results=json.loads((args.runs/'all_results.json').read_text())
    lookup={r['arguments']['rwkv_mux_mode']:r for r in results}
    assert set(lookup)==set(MODES)
    base=lookup['baseline']
    rows={}
    for mode in MODES:
        r=lookup[mode]
        assert r['data']==base['data']
        assert r['arguments']['steps']==r['history'][-1]['step']==15000
        assert r['training_targets']==3840000 and r['invalid_steps']==0
        assert abs(r['history'][0]['val_loss']-base['history'][0]['val_loss'])<2e-6
        assert r['tiled_evaluation']['targets']==1777024
        removed=sum(v['removed_dense_parameters'] for v in r['rwkv_replacements'])
        added=sum(v['added_gains'] for v in r['rwkv_replacements'])
        assert r['parameters']==567552-removed+added
        rows[mode]=dict(parameters=r['parameters'],parameter_reduction=567552-r['parameters'],
            rwkv_parameters=r['parameter_breakdown']['rwkv'],replaced_linears=len(r['rwkv_replacements']),
            full_shard_loss=r['tiled_evaluation']['loss'],
            loss_delta_vs_baseline=r['tiled_evaluation']['loss']-base['tiled_evaluation']['loss'],
            benchmark_ms_per_step=r['benchmark_ms_per_step'],sustained_ms_per_step=r['training_seconds']/15,
            evaluation=r['evaluation'],tiled_evaluation=r['tiled_evaluation'],physical_gpu=r['physical_gpu'])
    previous=json.loads((root/'side_history_summary.json').read_text())
    repeat=json.loads((root/'single_mux_gains_summary.json').read_text())['arms']['demuxed_history']
    summary=dict(baseline='demuxed_history_64_rank8_side2048',steps=15000,seed=11,
        batch_size=4,sequence_length=64,arms=rows,
        historical_baseline_seed11_losses=[previous['tiled_evaluation']['loss'],repeat['full_shard_loss']],
        replacements='All means all 46 nn.Linear modules inside RWKV. Each isolated type is replaced in all four layers, except value_control down/up which exist only in layers 2-4 (three occurrences).',
        mux='Fixed signed/permuted rank-8 Hadamard views with every input in every coordinate. Unit row norms, full attainable rank. Independent fixed seeds by layer and projection type, identical for a given slot in all-replaced and isolated arms. Learned scalar per output, initialized to the original dense row L2 norm.',
        unchanged='Embedding/head, all four 64-chart KS2 decoders, 2048D side residual, both 96D history states per previous layer, decoder mux/demux, RWKV recurrence/nonlinearities/norms/biases/time mix; unreplaced parameters have identical initialization.',
        optimizer='RWKV gain LR/WD match the replaced dense weights (base cosine 0.001 to 0.0002; WD 0.01). Existing decoder gains retain 5.5x LR and WD 0.01/5.5.',
        validation='All arms pass exact CPU initial logits and paired unchanged parameters; all-input code coverage/full rank; gain gradients and updates after zero paths open; causality. All-replaced GPU recurrence forward/backward matches reference. Training source hashes, identical data, finite steps, saved optimizer step counters, checkpoint reevaluation and full-shard batch-size parity checked.',
        limits='One training seed per arm. Fixed parameter budget is intentionally reduced when linears are replaced. Whole-model initial function is preserved by zero-initialized residual outputs, but internal projection orientations change. Hadamard matrices are precomputed and applied with dense F.linear, not an optimized butterfly kernel. Four-offset evaluation repeats the held-out shard with overlapping contexts; these are not independent datasets.')
    for p in (args.runs/'summary.json',root/'rwkv_mux_summary.json'):
        p.write_text(json.dumps(summary,indent=2)+'\n')
    (root/'rwkv_mux_results.json').write_text(json.dumps(results,indent=2)+'\n')
    current=json.loads((root/'current_baseline.json').read_text())
    current['latest_completed_repeat']=str((args.runs/'baseline/fixed_both_11/completed.json').resolve().relative_to(root))
    (root/'current_baseline.json').write_text(json.dumps(current,indent=2)+'\n')
    for mode,row in rows.items():
        print(f"{mode:22s} {row['parameters']:7d} {row['full_shard_loss']:.6f} {row['loss_delta_vs_baseline']:+.6f} {row['sustained_ms_per_step']:.3f} ms")


if __name__=='__main__':main()
