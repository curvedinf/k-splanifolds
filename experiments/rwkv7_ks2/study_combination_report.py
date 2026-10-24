"""Report incremental gains from the explicitly requested combination phase."""
import datetime
import json
import statistics as st


def add_sections(root, rows, section, table, esc, lookup, fig):
    if not (root / 'extension2_start.json').exists():
        return {}
    extension = json.loads((root / 'extension2_start.json').read_text())
    stamp = lambda value: datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    section('Two-hour extension to combine improvements', f'''<p>The user requested a new two-hour window specifically to combine successful KS2 changes: {stamp(extension['start_epoch'])} through {stamp(extension['deadline_epoch'])}. The previous 359 training runs had already completed. The original MLP remains fixed. The new screen has twelve KS2 configurations and three seeds each, followed by selected 100k replications on both training pools.</p><p>The earlier reserved scoring had already evaluated 109 checkpoints when this request arrived. Those outputs and 34 completed timing cases are preserved in <a href="interrupted_final_checks_before_extension2/partial_events.json">the interrupted evaluation record</a>, and are not used for combination selection. A fresh, disjoint interval at source validation tokens 2,000,000–2,099,999 is reserved for final confirmation. Its decoded size is 437,621 bytes. Old evaluation files remain unchanged, and new scores use separate filenames.</p><p>The screen combines small charts with power-basis optimization, slow geometry learning, warp-only or frozen geometry, and bias/gain calibration. Polynomial reparameterization combined with frozen coefficients is deliberately excluded as a new architecture: after coefficient conversion it describes the same fixed polynomial, so that combination would largely test floating-point implementation differences. <a href="combination_plan.json">Recorded plan</a> · <a href="combination_cpu_check.json">Structural, gradient and unchanged-control checks</a> · <a href="reserved_holdout_extension2.json">Fresh interval provenance</a>.</p>''')
    configs = json.loads((root / 'wave15_combinations_manifest.json').read_text())['jobs']
    parent_specs = {
        'combo_rank1_reference': ('wave12b_rank_freezing', 'spine_rank1_fused_repeat'),
        'combo_rank1_powers': ('wave15_combinations', 'combo_rank1_reference'),
        'combo_rank1_warps': ('wave15_combinations', 'combo_rank1_reference'),
        'combo_rank1_geo01': ('wave15_combinations', 'combo_rank1_reference'),
        'combo_rank1_frozen': ('wave15_combinations', 'combo_rank1_reference'),
        'combo_rank1_frozen_bias': ('wave15_combinations', 'combo_rank1_frozen'),
        'combo_rank1_frozen_gain': ('wave15_combinations', 'combo_rank1_frozen'),
        'combo_rank1_powers_geo01': ('wave15_combinations', 'combo_rank1_powers'),
        'combo_rank1_warps_bias': ('wave15_combinations', 'combo_rank1_warps'),
        'combo_rank4_frozen_bias': ('wave12b_rank_freezing', 'spine_rank4_frozen'),
        'combo_rank4_frozen_gain': ('wave12b_rank_freezing', 'spine_rank4_frozen'),
        'combo_rank4_powers': ('wave12b_rank_freezing', 'spine_rank4_fused_repeat'),
    }
    refs = {seed: lookup('initial', 'mlp', seed) if seed != 33 else lookup('wave3', 'mlp_seed33', seed) for seed in (11, 22, 33)}
    groups = []
    for name in dict.fromkeys(job['config']['name'] for job in configs):
        observed = []
        for seed in (11, 22, 33):
            record = lookup('wave15_combinations', f'{name}_{seed}')
            if record is None:
                continue
            wave, parent = parent_specs[name]
            previous = lookup(wave, f'{parent}_{seed}')
            raw = json.loads((root / 'wave15_combinations' / f'{name}_{seed}' / 'completed.json').read_text())
            cfg=raw['study_config'];rank=cfg.get('subspace_rank',8);charts=cfg['charts']
            fixed=charts*(6*rank+2) if cfg.get('frozen_geometry') else charts*6*rank if cfg.get('geometry_trainable')=='warps' else 0
            observed.append(dict(seed=seed, loss=record['full_validation_loss'],
                                 mlp_gap=record['full_validation_loss'] - refs[seed]['full_validation_loss'],
                                 parent_gap=record['full_validation_loss'] - previous['full_validation_loss'] if previous else None,
                                 ffn_parameters=raw['ffn_parameters_by_layer'][0],fixed_geometry_per_ffn=fixed))
        if not observed:
            continue
        parent_gaps = [x['parent_gap'] for x in observed if x['parent_gap'] is not None]
        groups.append(dict(model=name, parent=parent_specs[name][1], n=len(observed),
                           mean_loss=st.mean(x['loss'] for x in observed),
                           mean_mlp_gap=st.mean(x['mlp_gap'] for x in observed),
                           mean_parent_gap=st.mean(parent_gaps) if parent_gaps else None,
                           parent_wins=sum(g < 0 for g in parent_gaps), parent_pairs=len(parent_gaps),
                           ffn_parameters=observed[0]['ffn_parameters'],fixed_geometry_per_ffn=observed[0]['fixed_geometry_per_ffn'], individual=observed))
    interpretation=''
    by_name={g['model']:g for g in groups}
    if len(groups)==12 and all(g['n']==3 for g in groups):
        reference=by_name['combo_rank1_reference'];best=min((g for g in groups if g['model']!='combo_rank1_reference'),key=lambda g:g['mean_loss'])
        bias=by_name['combo_rank1_frozen_bias']
        interpretation=f'<p><strong>Completed screen:</strong> the fully learned rank-1 repeat has mean {reference["mean_loss"]:.6f}; the lowest-loss combination is {esc(best["model"])} at {best["mean_loss"]:.6f}. No combination improves that reference mean. Frozen rank-1 plus bias does improve its frozen parent by {bias["mean_parent_gap"]:+.6f}, with {bias["parent_wins"]}/{bias["parent_pairs"]} seed wins. This is a useful incremental gain within frozen geometry, rather than an accumulated lead over fully learned rank-1 KS2.</p>'
    section('Do the short-run improvements accumulate?', interpretation+table(
        ['Combination', 'Learned per FFN', 'Fixed geometry per FFN', 'Seeds', 'Mean loss', 'Gap vs fixed MLP', 'Simpler KS2 parent', 'Gap vs parent', 'Parent seed wins'],
        [[esc(g['model']), f"{g['ffn_parameters']:,}",f"{g['fixed_geometry_per_ffn']:,}", g['n'], f"{g['mean_loss']:.6f}", f"{g['mean_mlp_gap']:+.6f}", esc(g['parent']), f"{g['mean_parent_gap']:+.6f}" if g['mean_parent_gap'] is not None else 'pending', f"{g['parent_wins']}/{g['parent_pairs']}"] for g in groups]) + '''<p>Negative gaps favor the combination. The parent column identifies the specific incremental intervention; it does not replace the original MLP benchmark. Rank-1 reference is an exact architecture repeat and measures trajectory variation. Most rank-1 parents are rerun in the same wave; rank-4 parents reuse the earlier fused three-seed cohorts. All completed seeds are retained, and incomplete cohorts remain provisional. The fixed MLP has 73,728 learned parameters per FFN. Rank-1 frozen/warp calibration variants use 73,968 learned plus fixed coefficients per FFN, 240 more than MLP, despite using fewer learned parameters. Rank-4 frozen calibration uses 73,416 total coefficients per FFN, below MLP. Any advantage must be interpreted with those storage counts.</p>''')
    interaction = []
    for seed in (11, 22, 33):
        names = ('combo_rank1_reference', 'combo_rank1_powers', 'combo_rank1_geo01', 'combo_rank1_powers_geo01')
        rs = [lookup('wave15_combinations', f'{name}_{seed}') for name in names]
        if not all(rs):
            continue
        base, power, slow, combined = [r['full_validation_loss'] for r in rs]
        predicted = power + slow - base
        interaction.append(dict(seed=seed, base=base, powers=power, slow=slow, combined=combined,
                                additive_prediction=predicted, interaction=combined - predicted))
    if interaction:
        section('Power basis and geometry learning-rate interaction', table(
            ['Seed', 'Rank-1 parent', '+ powers', '+ slow geometry', '+ both', 'Additive loss prediction', 'Observed minus additive'],
            [[r['seed']] + [f"{r[k]:.6f}" for k in ('base', 'powers', 'slow', 'combined', 'additive_prediction')] + [f"{r['interaction']:+.6f}"] for r in interaction]) + '''<p>The additive prediction is loss(power) + loss(slow) − loss(parent), using the same seed and horizon. A negative residual means a better result than adding the individual observed effects; a positive residual means worse. A negative interaction can simply mean that two individually harmful changes do less harm together than their summed penalties: it does not establish that the combined model beats its parent. This small factorial contrast describes these training runs, not a statistical proof of synergy. Nonlinear optimization and floating-point trajectory variation can produce interactions.</p>''')
    repeat_path=root/'rank1_repeat_variation.json'
    if repeat_path.exists():
        repeats=json.loads(repeat_path.read_text())['rows']
        section('Variation in the unchanged KS2 reference',table(
            ['Seed','Earlier/new physical GPU','Shared source files changed','Earlier full-shard loss','Repeat full-shard loss','Repeat minus earlier'],
            [[r['seed'],f"{r['old_gpu']} / {r['new_gpu']}",len(r['changed_shared_sources']),f"{r['old_final']:.6f}",f"{r['new_final']:.6f}",f"{r['new_final']-r['old_final']:+.6f}"] for r in repeats])+'''<p>The equivalent fused rank-1 architecture was repeated without changing its settings. Data hashes, PyTorch/ROCm versions and all 51 shared source files match. Initial sampled losses match exactly; 100-update sampled losses differ by about 10⁻⁶ or less. Later trajectories diverge, including seed 11 on the same physical GPU. This is consistent with amplified floating-point variation, but the specific runtime source has not been localized. The custom geometry and recurrence kernels use no explicit atomics; that alone does not establish determinism for every operation in the complete model.</p><p>The new three-seed short mean is about 0.002 lower than the earlier fused repeat. Both cohorts remain in the record. The combination screen uses its same-wave reference; the long comparisons reuse the pre-existing rank-1 100k checkpoints. They do not have a new contemporaneous long reference repeat, so small incremental gains require caution. Neither cohort changes the original MLP control. <a href="rank1_repeat_variation.json">Source, software, data and checkpoint-loss comparison</a>.</p>''')
    long_groups=[]
    for pool in ('original','larger'):
        candidates=[r for r in rows if r['wave']=='wave16_combination_long' and r['job'].startswith(pool+'_')]
        for name in sorted({r['config']['name'] for r in candidates}):
            records=[]
            for record in candidates:
                if record['config']['name']!=name:continue
                seed=record['seed'];mlp=lookup('wave8_long',f'{pool}_original_mlp_{seed}')
                rank1=lookup('wave13_selected_long',f'{pool}_spine_rank1_fused_repeat_{seed}')
                records.append(dict(seed=seed,loss=record['full_validation_loss'],mlp_loss=mlp['full_validation_loss'],rank1_loss=rank1['full_validation_loss'],train_gap=record['sampled_training_loss']-mlp['sampled_training_loss']))
            long_groups.append(dict(pool=pool,model=name,n=len(records),mean=st.mean(r['loss'] for r in records),
                                    mlp_gap=st.mean(r['loss']-r['mlp_loss'] for r in records),
                                    rank1_gap=st.mean(r['loss']-r['rank1_loss'] for r in records),
                                    train_gap=st.mean(r['train_gap'] for r in records),
                                    mlp_wins=sum(r['loss']<r['mlp_loss'] for r in records),
                                    rank1_wins=sum(r['loss']<r['rank1_loss'] for r in records),individual=records))
    if long_groups:
        section('Do combined gains survive 100k updates?',table(
            ['Pool','Combination','Seeds','Mean loss','Gap vs fixed MLP','MLP seed wins','Sampled train gap vs MLP','Gap vs existing rank-1 KS2','Rank-1 seed wins'],
            [[g['pool'],esc(g['model']),g['n'],f"{g['mean']:.6f}",f"{g['mlp_gap']:+.6f}",f"{g['mlp_wins']}/{g['n']}",f"{g['train_gap']:+.6f}",f"{g['rank1_gap']:+.6f}",f"{g['rank1_wins']}/{g['n']}"] for g in long_groups])+fig('combination_long_trajectories','Combined KS2 changes compared with the fixed MLP and the prior rank-1 KS2 at matching 100k-schedule checkpoints.')+'''<p>The original MLP remains the primary reference. The existing rank-1 KS2 is the strongest earlier larger-pool long-run result; comparing against it tests whether a combination improves on that practical starting point. For rank-4 combinations, rank-1 is a benchmark, not an otherwise-identical single-intervention parent. The short-run parent table provides the more direct incremental ablations. These long comparisons reuse earlier reference checkpoints, retain all three seeds, and use the same data pool, update count and fresh 100k cosine schedule. The training-loss column uses the same 65,536 sampled targets within each pool; it is a paired model difference, not an absolute train-versus-validation generalization estimate.</p>''')
    selection=root/'combination_long_selection.json'
    if selection.exists():
        chosen=json.loads(selection.read_text())
        section('Selection of long combination tests',table(['Combination','Reason'],[[esc(name),esc(chosen['rationale'][name]['reason'])] for name in chosen['selected']])+'''<p>Selection was recorded after every three-seed screen completed and before reading the fresh reserved interval. All rejected candidates remain in the short-run tables and final reserved scoring. <a href="combination_long_selection.json">Complete selection record, all candidates and estimated schedule</a>.</p>''')
    held_groups=[]
    holdout=root/'final_checks/holdout_results.json'
    if holdout.exists():
        measured={r['run']:r['evaluation']['loss'] for r in json.loads(holdout.read_text())['rows']}
        for group in long_groups:
            pool,name=group['pool'],group['model']
            values=[]
            for seed in (11,22,33):
                paths=(f'wave16_combination_long/{pool}_{name}_{seed}',
                       f'wave8_long/{pool}_original_mlp_{seed}',
                       f'wave13_selected_long/{pool}_spine_rank1_fused_repeat_{seed}')
                assert all(path in measured for path in paths),paths
                loss,mlp,rank1=[measured[path] for path in paths]
                values.append(dict(seed=seed,loss=loss,mlp_loss=mlp,rank1_loss=rank1,
                                   mlp_gap=loss-mlp,rank1_gap=loss-rank1))
            held_groups.append(dict(pool=pool,model=name,n=len(values),
                                    mean=st.mean(v['loss'] for v in values),
                                    mlp_gap=st.mean(v['mlp_gap'] for v in values),
                                    rank1_gap=st.mean(v['rank1_gap'] for v in values),
                                    mlp_wins=sum(v['mlp_gap']<0 for v in values),
                                    rank1_wins=sum(v['rank1_gap']<0 for v in values),individual=values))
        section('Do combinations improve on the reserved interval?',table(
            ['Pool','Combination','Seeds','Reserved loss','Gap vs unchanged MLP','MLP seed wins','Gap vs existing rank-1 KS2','Rank-1 seed wins'],
            [[g['pool'],esc(g['model']),g['n'],f"{g['mean']:.6f}",f"{g['mlp_gap']:+.6f}",f"{g['mlp_wins']}/{g['n']}",f"{g['rank1_gap']:+.6f}",f"{g['rank1_wins']}/{g['n']}"] for g in held_groups])+'''<p>These are the same completed 100k checkpoints scored on the fresh 437,621-byte interval reserved before combination screening. The original MLP is the primary control; the prior rank-1 KS2 tests whether a combination adds to the strongest earlier larger-pool result. Both references use the identical reserved text, seed and training pool. This final interval was not used to choose or tune these models. All four selected combinations and all three seeds are retained. A win against MLP alone is not evidence that improvements accumulated beyond the simpler KS2 model.</p>''')
    return dict(short_groups=groups, power_lr_interaction=interaction,long_groups=long_groups,reserved_long_groups=held_groups)
