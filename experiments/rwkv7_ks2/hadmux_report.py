"""Build the Hadamard follow-up ledger, comparisons and GitHub-readable report."""
import argparse
import datetime
import json
from pathlib import Path
import statistics as st
from study_collect import collect


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| '+' | '.join(map(cell, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(cell, r))+' |' for r in rows])+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = collect(root)
    state = json.loads((root/'study_state.json').read_text())
    prior = root.parent/state['prior_study']
    oldrows = json.loads((prior/'all_runs.json').read_text())['rows']
    refs = {}
    for seed in (11, 22, 33):
        refs[(15000, seed)] = next(r for r in oldrows if r['family']=='mlp' and r['seed']==seed and
            ((r['wave']=='initial' and r['job']=='mlp') or (r['wave']=='wave3' and r['job']=='mlp_seed33')))
        refs[(100000, seed)] = next(r for r in oldrows if r['wave']=='wave8_long' and r['job']==f'original_original_mlp_{seed}')
    # Historical references remain primary at their original horizons. New
    # horizons use the contemporaneous, unchanged MLP cohort.
    for row in rows:
        if row['family']=='mlp' and (row['steps'],row['seed']) not in refs:
            refs[(row['steps'],row['seed'])]=row
    groups = []
    for wave, name in sorted({(r['wave'], r['config']['name']) for r in rows}):
        cohort = [r for r in rows if r['wave']==wave and r['config']['name']==name]
        cfg = cohort[0]['config']
        losses = [r['full_validation_loss'] for r in cohort]
        controls = [refs.get((r['steps'], r['seed'])) for r in cohort]
        paired = [r['full_validation_loss']-m['full_validation_loss'] for r, m in zip(cohort, controls) if m]
        raw = json.loads((root/cohort[0]['source']).read_text())
        ffn = raw['ffn_parameters_by_layer']
        group = dict(wave=wave, name=name, config=cfg, n=len(cohort), seeds=[r['seed'] for r in cohort],
                     steps=cohort[0]['steps'], mean=st.mean(losses), sd=st.stdev(losses) if len(losses)>1 else None,
                     learned_per_ffn=ffn, learned_all_ffns=sum(ffn),
                     mean_ms=st.mean(r['ms_per_step'] for r in cohort),
                     mean_training_seconds=st.mean(r['training_seconds'] for r in cohort),
                     fixed_buffer_mib=st.mean(r['buffer_bytes'] for r in cohort)/2**20,
                     peak_allocated_mib=st.mean(r['peak_allocated_bytes'] for r in cohort)/2**20,
                     mean_sampled_train=st.mean(r['sampled_training_loss'] for r in cohort),
                     mlp_gap=st.mean(paired) if len(paired)==len(cohort) else None,
                     mlp_seed_wins=sum(v<0 for v in paired) if len(paired)==len(cohort) else None,
                     individual=[dict(seed=r['seed'],loss=r['full_validation_loss'],ms=r['ms_per_step'],source=r['source']) for r in cohort])
        groups.append(group)
    summary = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   completed_runs=len(rows), status=state['status'], groups=groups)
    (root/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    sections = ['# Returning to Hadamard multiplexing: KS2 combinations and scaling\n',
                f"Status: **{state['status']}**. {len(rows)} completed runs.\n",
                '## Question and controls\n',
                'The main pursuit, following the user’s metric request, is the lower-left frontier of **(learned FFN parameter ratio × step-time ratio) versus validation loss**. Lowest loss alone is not the selection objective. The targeted final screen widens frozen Hadamard features at a constant learned FFN budget.\n',
                'Can the KS2 improvements from the preceding study transfer to the fixed Hadamard architecture, and can its decoder capacity scale efficiently? The primary control remains the original four-layer, 96D MLP model. No MLP architecture or optimizer search is performed.\n',
                'The first screen keeps the prior 96D attention stream, 2,048D decoder-only side residual, and differentiable history of each previous layer’s post-attention and post-decoder 96D states. Both decoder input and output projections remain fixed Hadamard-derived codes. Learned output gains remain.\n',
                '## Architecture\n',
                '```text\n96D main → LayerNorm → RWKV-7 → add to main\nconcat(main, side, previous post-attention and post-decoder states)\n  → LayerNorm + causal time mixing\n  → fixed Hadamard mux → F coordinates\n  → coordinate bypass + KS2 geometry → optional ReLU²\n  → fixed Hadamard demux × learned output gains\n  → add to main and side residuals\nfinal main → LayerNorm → learned byte logits\n```\n',
                'At F=512, the mux uses the original 64 rank-8 projection codes. Geometry can regroup those exact coordinates into 64 R8 charts, 128 R4 charts, or 512 R1 charts. Every coordinate mixes the full decoder input. Changing geometry rank in this screen does not change the projection matrices. The Hadamard construction rank and the KS2 geometry rank are separate choices.\n',
                'The current implementation materializes the fixed codes as dense matrices. They have no weight gradients; this does not make their forward/input-gradient matrix multiplications free. Fixed coefficient storage is separate from learned parameter count.\n',
                'With the 2,048D side residual, the four mux input widths are 2,144, 2,336, 2,528 and 2,720; each demux outputs 2,144 coordinates. At F=512 that averages 2,342,912 projection multiply-accumulates per token per FFN forward pass, versus 73,728 for the original 96→384→96 MLP projections. This arithmetic count excludes geometry, attention and kernel overhead and is not a measured speed ratio. Fixed-map savings should be compared at the same matrix dimensions; the narrow learned-projection R1 architecture has different dimensions and no side/history stream.\n',
                '## Protocol\n',
                f"The two-hour exploration window starts {state['start_utc']} and ends {state['deadline_utc']}. New training launches stop at that cutoff; in-flight runs finish before final scoring. Four MI100 GPUs run independent jobs.\n",
                'Training uses FP32, captured HIP graphs, batch 4 × 64 bytes, AdamW with cosine learning rate 0.001→0.0002, clipping at 1, and weight decay 0.01. Hadamard output gains retain the prior 5.5× learning rate with inverse weight-decay adjustment. Three seeds are used for replicated claims.\n',
                'Loss is nats per byte on the exploratory validation shard, averaged over four overlapping context offsets. Those offsets are not independent datasets. Fresh reserved text at source validation tokens 3,000,000–3,099,999 is withheld until selection and training finish.\n',
                'Parameter tables count the FFN module only; attention, embedding/head, and external LayerNorm/time-mixing parameters are separate. The original MLP has 73,728 learned parameters per FFN. Frozen geometry is still stored and counted separately from learned parameters.\n']
    sections += ['The 2k side/history architecture additionally learns an average 7,296 external FFN LayerNorm/time-mixing parameters per layer, versus 288 for the original MLP. Thus the frozen branch’s 2,144 learned FFN-module parameters are not its entire decoder-path budget. Those external parameters remain constant across the targeted frozen-feature sweep ([scope check](ffn_metric_scope.json)).\n']
    for wave in sorted({g['wave'] for g in groups}):
        selected = sorted([g for g in groups if g['wave']==wave], key=lambda g:g['mean'])
        sections += [f'## {wave.replace("_", " ")}\n', table(
            ['Configuration','Seeds','Steps','Learned / FFN','Mean loss','Seed SD','Gap vs MLP','Mean ms/step','Training seconds','Buffers MiB','Peak MiB'],
            [[g['name'],g['n'],g['steps'],','.join(str(x) for x in sorted(set(g['learned_per_ffn']))),f"{g['mean']:.6f}",f"{g['sd']:.6f}" if g['sd'] is not None else '—',
              f"{g['mlp_gap']:+.6f}" if g['mlp_gap'] is not None else 'awaiting paired control',f"{g['mean_ms']:.3f}",f"{g['mean_training_seconds']:.1f}",f"{g['fixed_buffer_mib']:.1f}",f"{g['peak_allocated_mib']:.1f}"] for g in selected]),
            'Partial cohorts are provisional. Training-worker timing includes actual training segments and excludes setup/evaluation; final controlled timings are reported separately when available.\n']
    sections += ['MLP gaps use the historical original MLP at 15k/100k and the contemporaneous original MLP at other horizons. Buffers include all fixed model tensors, while peak memory includes training allocations. Neither is a learned-parameter count.\n']
    by_name = {g['name']:g for g in groups}
    parents = {
        'had_full_r8_relu2':'had_full_r8',
        'scale_full_f1024_s2048':'had_full_r8_relu2',
        'scale_full_f512_s8192':'had_full_r8_relu2',
        'scale_r1_f2048_s2048':'had_spine_r1_powers',
        'scale_r1_matched_s2048':'had_spine_r1_powers',
        'scale_r8spine_f2048_s2048':'had_spine_r8_relu2',
        'scale_r8spine_f2048_s8192':'scale_r8spine_f2048_s2048',
        'pareto_frozen_r8_f1024_s2048':'had_spine_r8_frozen',
        'pareto_frozen_r8_f2048_s2048':'had_spine_r8_frozen',
        'pareto_r8_warps_only_f512_s2048':'had_spine_r8_frozen',
    }
    parent_rows=[]
    for child,parent in parents.items():
        if child not in by_name or parent not in by_name:continue
        c,b=by_name[child],by_name[parent]
        if c['n']!=3 or b['n']!=3:continue
        paired=[v['loss']-next(x['loss'] for x in b['individual'] if x['seed']==v['seed']) for v in c['individual']]
        parent_rows.append([child,parent,f'{st.mean(paired):+.6f}',sum(v<0 for v in paired),f"{c['mean_ms']/b['mean_ms']:.2f}×",f"{sum(c['learned_per_ffn'])/sum(b['learned_per_ffn']):.2f}×"])
    if parent_rows:
        sections += ['## Improvements relative to the appropriate Hadamard parent\n',table(['Child','Parent','Paired loss change','Seed wins / 3','Step-time ratio','Learned FFN ratio'],parent_rows)]
    repeat=[r for r in rows if r['config']['name']=='original_mlp_repeat']
    if repeat:
        sections += ['## Original MLP reproduction\n',table(['Seed','Historical loss','Repeat loss','Difference'],[[r['seed'],f"{refs[(15000,r['seed'])]['full_validation_loss']:.6f}",f"{r['full_validation_loss']:.6f}",f"{r['full_validation_loss']-refs[(15000,r['seed'])]['full_validation_loss']:+.6f}"] for r in repeat]),
            'All three initial model states and optimizer groups/defaults exactly match the original MLP ([identity check](mlp_optimizer_identity.json)). FP32 training trajectories are not bitwise reproducible across these executions. Differences of this size limit interpretation of tiny historical gaps. No repeat was selected or discarded to favor KS2.\n']
    speed=root/'final_checks/speed_results.json'
    if speed.exists():
        measured=json.loads(speed.read_text())
        sections += ['## Controlled whole-step speed\n',measured['protocol']+'\n',table(['Model','Median ms','Min ms','Max ms'],[[v['model'],f"{v['median_ms']:.3f}",f"{v['min_ms']:.3f}",f"{v['max_ms']:.3f}"] for v in measured['summary']]),'[All five timing rounds](final_checks/speed_results.json).\n']
    reserved=root/'final_checks/holdout_results.json'
    if reserved.exists():
        evidence=json.loads(reserved.read_text());reserved_rows=[]
        for item in evidence['rows']:
            source=json.loads((Path(item['run'])/'completed.json').read_text())
            cfg=source.get('study_config') or dict(name='original_mlp',family='mlp')
            name=cfg['name'] if not item['prior'] else item['label'].rsplit('_',1)[0]
            reserved_rows.append(dict(name=name,seed=source['arguments']['seed'],steps=source['arguments']['steps'],family=cfg['family'],loss=item['evaluation']['loss'],prior=item['prior']))
        controls={(v['steps'],v['seed']):v for v in reserved_rows if v['name']=='original_mlp'}
        for v in reserved_rows:
            if v['family']=='mlp' and (v['steps'],v['seed']) not in controls:controls[(v['steps'],v['seed'])]=v
        reserved_groups=[]
        for name in sorted({v['name'] for v in reserved_rows}):
            cohort=[v for v in reserved_rows if v['name']==name];values=[v['loss'] for v in cohort]
            paired=[v['loss']-controls[(v['steps'],v['seed'])]['loss'] for v in cohort]
            reserved_groups.append(dict(name=name,n=len(cohort),steps=cohort[0]['steps'],mean=st.mean(values),sd=st.stdev(values),mlp_gap=st.mean(paired),seed_wins=sum(v<0 for v in paired),values=cohort))
        summary['reserved_groups']=reserved_groups
        sections += ['## Fresh reserved text: scored after selection and training\n',
            'All new configurations and the predeclared prior reference cohorts are included. Prior learned-projection R1 models have no decoder side/history stream; their differences cannot be attributed to projection type alone.\n',
            table(['Model','Steps','Seeds','Mean loss','Seed SD','Gap vs original MLP','Seed wins / 3'],[[v['name'],v['steps'],v['n'],f"{v['mean']:.6f}",f"{v['sd']:.6f}",f"{v['mlp_gap']:+.6f}",v['seed_wins']] for v in sorted(reserved_groups,key=lambda v:(v['steps'],v['mean']))]),
            '[Predeclared coverage](final_holdout_plan.json) · [Scores and dataset hashes](final_checks/holdout_results.json).\n']
    if (root/'efficiency/size_speed_loss.png').exists():
        sections += ['## Combined size × speed cost versus loss\n',
            '**X = (learned FFN parameters / original MLP FFN parameters) × (step milliseconds / original MLP step milliseconds). Y = loss.** Lower-left is better. This uses the combined metric requested by the user, with equal proportional weight on size and step duration.\n',
            '![Combined size × speed cost versus loss](efficiency/size_speed_loss.png)\n',
            'Stars show the observed frontier among complete three-seed cohorts; open points are provisional. The default compares the original training dataset at 15k updates and, once final checks finish, uses fresh reserved loss with controlled same-GPU timings. Exploratory loss, worker timings and other matched data/horizon cohorts remain selectable in the same chart. Fixed buffers are excluded from learned size and remain available in hover/data.\n',
            '[Definitions and frontier tables](efficiency/REPORT.md) · [Interactive chart](efficiency/size_speed_loss.html) · [Metrics CSV](efficiency/metrics.csv).\n']
    for name,caption in [('long_curves','Long-run validation loss versus updates and measured training time')]:
        if (root/'figures'/f'{name}.svg').exists():sections += [f'## {caption}\n',f'![{caption}](figures/{name}.svg)\n']
    factorial_names = ['had_spine_r8','had_spine_r1','had_spine_r8_relu2','had_spine_r1_relu2']
    by_name = {g['name']:g for g in groups if g['wave']=='wave1_transfer'}
    if all(name in by_name and by_name[name]['n']==3 for name in factorial_names):
        interaction=[]
        for seed in (11,22,33):
            values=[next(v['loss'] for v in by_name[name]['individual'] if v['seed']==seed) for name in factorial_names]
            interaction.append(dict(seed=seed,rank1_gap=values[1]-values[0],relu2_gap=values[2]-values[0],
                                    combined_gap=values[3]-values[0],interaction=values[3]-values[1]-values[2]+values[0]))
        summary['rank_activation_interaction']=interaction
        sections += ['## Do rank-1 and ReLU² accumulate under fixed Hadamard routing?\n',table(
            ['Seed','R1 change from R8','ReLU² change from plain R8','Combined change','Interaction'],
            [[v['seed'],f"{v['rank1_gap']:+.6f}",f"{v['relu2_gap']:+.6f}",f"{v['combined_gap']:+.6f}",f"{v['interaction']:+.6f}"] for v in interaction]),
            'All four arms use spine-only geometry, the same 512 fixed projected coordinates, 2k side residual and history. Negative loss changes favor the intervention. Positive interaction means the combined loss is worse than the sum of the two separate improvements would suggest. Rank changes also change warp count and scalar-input statistics; this is an interaction between complete interventions, not an isolated proof about summary count.\n']
    probe=root/'wave1_transfer_activation_probe.json'
    if probe.exists():
        observed=[v for v in json.loads(probe.read_text())['rows'] if v['state']=='trained']
        sections += ['## Is geometry or the side channel inactive?\n',table(
            ['Model','Mean coordinate RMS','Mean geometry/coordinate RMS','Side RMS entering layer 4'],
            [[v['model'],f"{st.mean(x['coordinate_rms'] for x in v['layers']):.3f}",f"{st.mean(x['geometry_to_coordinate_rms'] for x in v['layers']):.3f}",f"{v['layers'][-1]['side_input_rms']:.3f}"] for v in observed]),
            'These descriptive measurements use the same 4×64 training bytes and seed 11 checkpoints. They are not held-out scores or causal ablations. The observed R1 geometry is active and the side residual carries nonzero signal; large activation magnitude alone does not establish useful information. [Full activation probe](wave1_transfer_activation_probe.json).\n']
    projection=root/'projection_speed.json'
    if projection.exists():
        probe=json.loads(projection.read_text())
        sections += ['## Fixed versus learned projection cost at the same dimensions\n',table(
            ['Features','Direction','Projection kind','Median ms'],
            [[v['features'],v['direction'],v['kind'],f"{v['median_ms']:.4f}"] for v in probe['summary']]),
            probe['protocol']+'\n\n[Raw five-round measurements](projection_speed.json).\n']
    audit=root/'protocol_audit.json'
    if audit.exists():
        verified=json.loads(audit.read_text())
        sections += ['## Execution and verification\n',
            f"Protocol audit: **{verified['status']}**. {verified['training_runs']} complete training runs; {verified['reserved_scores']} fresh reserved scores. All training launches occurred before the exploration cutoff. Each physical GPU ran independent jobs, with no overlapping study jobs on a card.\n",
            table(['GPU','Jobs','Utilization samples','Mean GPU use %','Median clock MHz','Clock min–max MHz','Max junction °C'],[[v['gpu'],v['job_count'],v['sample_count'],f"{v['mean_gpu_use']:.1f}",v['clock_mhz_median'],f"{v['clock_mhz_min']}–{v['clock_mhz_max']}",v['junction_temperature_max']] for v in verified['gpus']]),
            'Telemetry covers job setup, training and evaluation. Clocks were not locked; worker timings therefore reflect the operating conditions during those jobs. The final randomized same-GPU timings help separate architecture cost from card/order effects, but do not establish performance at a universal clock.\n',
            '[Checkpoint, optimizer and executed-source audit](audit.json) · [Protocol and GPU timeline](protocol_audit.json) · [Exact source archive](source_bundle.tar.gz) · [Archive verification](source_bundle_check.json).\n']
    sections += ['## Interpretation limits\n',
                 'A configuration must improve the appropriate Hadamard parent, not merely approach MLP. Rank, geometry simplification, feature width and side width have different effects. The first transfer screen holds the projection matrices fixed; later width changes can change codes and initialization and must be interpreted as complete architecture changes.\n',
                 'Equal-step loss and loss at measured training time answer different questions. Comparisons across different cosine horizons are labeled; intermediate checkpoints from a 100k schedule are not equivalent to independently annealed 15k runs. Small seed-level gains are not universal advantages.\n',
                 '## Complete run ledger\n', table(
                     ['Wave','Run','Seed','Steps','Loss','ms/step','Record'],
                     [[r['wave'],r['job'],r['seed'],r['steps'],f"{r['full_validation_loss']:.6f}",f"{r['ms_per_step']:.3f}",f"[JSON]({r['source']})"] for r in rows]),
                 '## Artifacts\n',
                 '- [Experiment plan](experiment_plan.json)\n- [Comparison summary](comparison_summary.json)\n- [CPU structural checks](hadmux_cpu_check.json)\n- [Reserved interval provenance](reserved_holdout.json)\n- [Study state](study_state.json)\n']
    # Lead with the user's joint cost/loss objective and the completed findings.
    start=next((i for i,s in enumerate(sections) if s.startswith('## Combined size')),None)
    if start is not None:
        end=next((i for i in range(start+1,len(sections)) if sections[i].startswith('## ')),len(sections))
        metric=sections[start:end];del sections[start:end]
        sections[2:2]=metric
    findings=root/'FINDINGS.md'
    if findings.exists():
        position=2+len(metric) if start is not None else 2
        sections[position:position]=[findings.read_text()]
    (root/'REPORT.md').write_text('\n'.join(sections))
    (root/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows), groups=len(groups), report=str(root/'REPORT.md'))))


if __name__ == '__main__':
    main()
