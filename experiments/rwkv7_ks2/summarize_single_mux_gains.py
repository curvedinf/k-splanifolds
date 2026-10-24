"""Validate and summarize the five-arm shared-mux gain experiment."""
import argparse
import json
from pathlib import Path
import torch


def stats(value):
    value=value.float()
    return dict(mean=value.mean().item(),std=value.std(unbiased=False).item(),
                minimum=value.min().item(),maximum=value.max().item(),
                rms=value.square().mean().sqrt().item())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    results=json.loads((args.runs/'all_results.json').read_text())
    lookup={r['gain_ablation']:r for r in results}
    assert set(lookup)=={'baseline','input','output','dynamic','demuxed_history'}
    base=lookup['baseline']
    rows={}
    for mode in ('baseline','output','input','dynamic','demuxed_history'):
        r=lookup[mode]
        assert r['data']==base['data']
        assert r['training_targets']==3840000 and r['invalid_steps']==0
        assert r['arguments']['steps']==r['history'][-1]['step']==15000
        if mode in ('input','output'):
            assert abs(r['history'][0]['val_loss']-base['history'][0]['val_loss'])<2e-6
        row=dict(parameters=r['parameters'],full_shard_loss=r['tiled_evaluation']['loss'],
                 loss_delta_vs_single_mux=r['tiled_evaluation']['loss']-base['tiled_evaluation']['loss'],
                 benchmark_ms_per_step=r['benchmark_ms_per_step'],
                 sustained_ms_per_step=r['training_seconds']/15,
                 initial_validation_loss=r['history'][0]['val_loss'],
                 evaluation=r['evaluation'],tiled_evaluation=r['tiled_evaluation'],physical_gpu=r['physical_gpu'])
        if mode in ('input','output','dynamic'):
            checkpoint=torch.load(args.runs/mode/'fixed_both_11/checkpoint.pt',map_location='cpu',weights_only=False)
            assert checkpoint['step']==15000
            sd=checkpoint['model']
            assert all(torch.isfinite(v).all() for v in sd.values())
            row['learned_gain_statistics']=[]
            for i in range(4):
                suffix='gain_vectors' if mode=='dynamic' else 'gain'
                value=sd[f'blocks.{i}.mux.{suffix}']
                record=stats(value)
                if mode=='dynamic':record['per_vector_rms']=value.square().mean(-1).sqrt().tolist()
                else:record['rms_change_from_one']=(value-1).square().mean().sqrt().item()
                row['learned_gain_statistics'].append(record)
        rows[mode]=row
    historical={}
    for mode,file in [('baseline','single_mux_summary.json'),('demuxed_history','side_history_summary.json')]:
        previous=json.loads((root/file).read_text())
        historical[mode]=dict(previous_loss=previous['tiled_evaluation']['loss'],
            fresh_loss=rows[mode]['full_shard_loss'],
            fresh_minus_previous=rows[mode]['full_shard_loss']-previous['tiled_evaluation']['loss'])
    summary=dict(historical_same_seed_repeats=historical,steps=15000,seed=11,batch_size=4,sequence_length=64,arms=rows,
        protocol='Three separate gain arms on the corrected single-mux architecture, plus fresh single-mux and demuxed-history controls. One GPU per independent job across four MI100s. Same byte data, samples, FP32 optimizer schedule, 64 rank-8 main KS2 subspaces, four layers, and 96D RWKV. Extra dynamic control view is local to each layer.',
        static_initialization='Input and output gains start at one, preserving the CPU initial function exactly; GPU parity holds within FP32 rounding tolerance. Original learned weights and fixed main mux rows are paired in all three arms.',
        dynamic_formula='z = KS2(extra_rank8_mux_view); g = sum_i(z_i * learned_vector_i); g = g / sqrt(max(mean(g*g),1e-8)); main_mux_output *= g. Normalize once across the 608 summed coordinates per token. No additive offset or separate static gains.',
        dynamic_initialization='Eight nonzero random 608D vectors per layer, Normal(0,1/sqrt(8)); changes the initial function and adds 22,736 parameters. Vectors use 5.5x LR; extra geometry uses base LR.',
        evaluation='Same held-out byte shard, four overlapping offsets, 1,777,024 target visits. Single seed; these visits are not independent data. GPU training is not bit deterministic, including the unchanged reference.',
        validation='Routing, normalization-after-weighted-sum, finite nonzero gradients, optimizer groups, actual gain updates, and causality passed. Every training step finite; source hashes, checkpoint optimizer counters, restoration and full-shard batch-size parity checked.')
    for path in (args.runs/'summary.json',root/'single_mux_gains_summary.json'):
        path.write_text(json.dumps(summary,indent=2)+'\n')
    (root/'single_mux_gains_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({mode:{k:v for k,v in row.items() if k in ('parameters','full_shard_loss','loss_delta_vs_single_mux','benchmark_ms_per_step','sustained_ms_per_step')} for mode,row in rows.items()},indent=2))


if __name__=='__main__':main()
