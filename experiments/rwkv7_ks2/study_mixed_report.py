"""Report the explicitly requested mixed-rank and MLP-branch experiments."""
import json
import statistics as st


def add_sections(root, rows, section, table, esc, lookup):
    plan_path = root / 'mixed_rank_plan.json'
    if not plan_path.exists():
        return {}
    configs = json.loads(plan_path.read_text())['configs']
    designs = []
    for cfg in configs:
        features = sum(rank * charts for rank, charts in cfg['mixed_ranks'])
        geometry = 6 * features + 2 * sum(charts for rank, charts in cfg['mixed_ranks'])
        designs.append([cfg['name'], ', '.join(f'{charts} × rank {rank}' for rank, charts in cfg['mixed_ranks']),
                        sum(charts for rank,charts in cfg['mixed_ranks']),cfg.get('mlp_features', 0), '368', f'{70656 + geometry:,}'])
    section('Mixed KS2 ranks and an MLP branch', '''<p>During the two-hour extension, the user requested mixed KS2 ranks, two rank-16 charts, and an MLP in its own subspace. These are mixtures within every FFN, not different ranks assigned to different layers. Each of the four FFNs keeps one learned 96→368 input projection and one learned 368→96 output projection. Disjoint coordinate groups use their own KS2 spine charts; every projected coordinate can depend on all 96 main-stream inputs.</p><p>The hybrid replaces 32 rank-1 coordinates in the balanced mixture with a 32-feature ReLU² MLP branch. Its hidden coordinates receive no KS2 geometry. The branch outputs concatenate with the KS2 outputs before the shared learned readout. Equivalently, the MLP branch has a learned 96→32 input projection and a learned 32→96 output contribution. It does not increase the 368-feature allocation or add a separate residual path.</p>'''+table(
        ['Variant', 'KS2 chart allocation per FFN', 'Independent spine summaries', 'MLP hidden features', 'Total projected features', 'Learned per FFN'], designs)+'''<p>All geometry is learned and all three FFN budgets are below the original MLP's 73,728 parameters. The mixtures retain the spine-only cell, with coordinate residual followed by ReLU²; they do not reintroduce full edge/damping geometry. All variants use three seeds and the same 15k schedule on the original pool. The original MLP remains the primary reference, and the earlier same-extension learned rank-1 cohort supplies the simpler KS2 comparison. <a href="mixed_rank_plan.json">Request and plan</a> · <a href="mixed_rank_cpu_check.json">CPU structural and gradient checks</a> · <a href="wave17_mixed_ranks_manifest.json">Exact manifest</a>.</p>''')
    scheduling=root/'mixed_rank_interleave.json'
    if scheduling.exists():
        record=json.loads(scheduling.read_text())
        finished=record.get('resume_finished_epoch')
        description=f"All workers had resumed by {(finished-record['started_epoch'])/60:.2f} minutes after suspension." if finished else 'The short screen is running. As its queue drains, each released GPU resumes its suspended long worker; the dispatcher resumes with the first released GPU.'
        section('Scheduling the late mixed-rank request',f'''<p>The end-of-queue forecast could not start every new three-seed test before the cutoff. The long-run dispatcher and its four active workers were therefore suspended in memory while the short mixed-rank screen uses the four GPUs. No checkpoint was reloaded, no optimizer state was reset and no training steps were removed. {esc(description)}</p><p>The suspended runs' raw elapsed training times can include the suspension and are excluded from speed comparisons. Their original measurements remain in the raw artifacts, with suspension flags in the flat ledger. Controlled final timing uses separate fresh processes and is unaffected. <a href="mixed_rank_interleave.json">Exact process identities and suspension/resumption timestamps</a> · <a href="mixed_rank_schedule_forecast.json">Scheduling forecast</a>.</p>''')
    groups = []
    for cfg in configs:
        values = []
        for seed in (11, 22, 33):
            run = lookup('wave17_mixed_ranks', f'{cfg["name"]}_{seed}')
            if run is None:
                continue
            mlp = lookup('initial', 'mlp', seed) if seed != 33 else lookup('wave3', 'mlp_seed33', seed)
            rank1 = lookup('wave15_combinations', f'combo_rank1_reference_{seed}')
            parent = lookup('wave17_mixed_ranks', f'mixed_balanced_{seed}') if cfg.get('mlp_features') else None
            values.append(dict(seed=seed, loss=run['full_validation_loss'],
                               mlp_gap=run['full_validation_loss']-mlp['full_validation_loss'],
                               rank1_gap=run['full_validation_loss']-rank1['full_validation_loss'],
                               parent_gap=run['full_validation_loss']-parent['full_validation_loss'] if parent else None))
        if not values:
            continue
        parents = [v['parent_gap'] for v in values if v['parent_gap'] is not None]
        groups.append(dict(model=cfg['name'], n=len(values), mean=st.mean(v['loss'] for v in values),
                           mlp_gap=st.mean(v['mlp_gap'] for v in values), rank1_gap=st.mean(v['rank1_gap'] for v in values),
                           mlp_wins=sum(v['mlp_gap']<0 for v in values), rank1_wins=sum(v['rank1_gap']<0 for v in values),
                           parent_gap=st.mean(parents) if parents else None, individual=values))
    if groups:
        section('Mixed-rank results', table(
            ['Variant','Seeds','Mean loss','Gap vs unchanged MLP','MLP seed wins','Gap vs uniform rank-1','Rank-1 seed wins','MLP branch gap vs balanced KS2'],
            [[esc(g['model']),g['n'],f"{g['mean']:.6f}",f"{g['mlp_gap']:+.6f}",f"{g['mlp_wins']}/{g['n']}",f"{g['rank1_gap']:+.6f}",f"{g['rank1_wins']}/{g['n']}",f"{g['parent_gap']:+.6f}" if g['parent_gap'] is not None else '—'] for g in groups])+'''<p>Negative gaps favor the mixture. The final column compares the MLP-branch model directly with the balanced all-KS2 mixture; both have 368 features, while the MLP branch removes the geometry parameters belonging to its 32 coordinates. Partial cohorts are provisional. These 15k tests do not establish the mixed architectures' 100k ordering.</p>''')
    if len(groups)==3 and all(g['n']==3 for g in groups):
        section('What the mixed-rank experiment suggests', '''<p>Among the all-KS2 spine variants, the mean ordering is uniform rank-1 (368 independent spine summaries), rank-1-heavy mixture (258), then balanced mixture (154). This is consistent with the hypothesis that more independently evaluated scalar geometry is useful at this small decoder width. It does not isolate summary count: chart grouping, mean-input scale and the number of learned warp parameters change together. The earlier scale-only interventions did not reproduce the rank benefit.</p><p>The hybrid has 122 KS2 summaries plus 32 ordinary ReLU² features and performs close to the balanced all-KS2 mixture. Its small mean gain does not show that MLP and KS2 provide strongly complementary features. Every model still has 368 raw coordinate bypasses and 368 output activations, so the count of spine summaries is not a rank bound on the complete FFN. Two rank-16 charts are present in every requested mixture, but the experiment does not establish that rank-16 charts help: there is no otherwise-identical mixture without them. These are findings about the tested spine-only cell, not a test of mixed-rank full edge/damping geometry.</p>''')
    reserved=[]
    holdout=root/'final_checks/holdout_results.json'
    if holdout.exists():
        measured={r['run']:r['evaluation']['loss'] for r in json.loads(holdout.read_text())['rows']}
        for cfg in configs:
            values=[]
            for seed in (11,22,33):
                run=f'wave17_mixed_ranks/{cfg["name"]}_{seed}'
                mlp=f'initial/mlp/fixed_both_{seed}' if seed!=33 else 'wave3/mlp_seed33'
                rank1=f'wave15_combinations/combo_rank1_reference_{seed}'
                balanced=f'wave17_mixed_ranks/mixed_balanced_{seed}'
                assert all(path in measured for path in (run,mlp,rank1,balanced))
                values.append(dict(seed=seed,loss=measured[run],mlp_gap=measured[run]-measured[mlp],
                                   rank1_gap=measured[run]-measured[rank1],parent_gap=measured[run]-measured[balanced]))
            reserved.append(dict(model=cfg['name'],n=len(values),mean=st.mean(v['loss'] for v in values),
                                 mlp_gap=st.mean(v['mlp_gap'] for v in values),rank1_gap=st.mean(v['rank1_gap'] for v in values),
                                 mlp_wins=sum(v['mlp_gap']<0 for v in values),rank1_wins=sum(v['rank1_gap']<0 for v in values),
                                 parent_gap=st.mean(v['parent_gap'] for v in values) if cfg.get('mlp_features') else None,individual=values))
        section('Mixed ranks on the reserved interval',table(
            ['Variant','Seeds','Reserved loss','Gap vs unchanged MLP','MLP seed wins','Gap vs uniform rank-1','Rank-1 seed wins','MLP branch gap vs balanced KS2'],
            [[esc(g['model']),g['n'],f"{g['mean']:.6f}",f"{g['mlp_gap']:+.6f}",f"{g['mlp_wins']}/{g['n']}",f"{g['rank1_gap']:+.6f}",f"{g['rank1_wins']}/{g['n']}",f"{g['parent_gap']:+.6f}" if g['parent_gap'] is not None else '—'] for g in reserved])+'''<p>All three requested architectures and all three seeds are scored on the same fresh reserved text interval after training ends. References use the same 15k horizon and original training pool. The mixture was specified before these scores existed; no rank allocation or MLP-branch size was selected using the reserved interval. A narrow gain on this interval remains specific to the tested data, model scale and training horizon.</p>''')
    return dict(configurations=configs, short_groups=groups,reserved_groups=reserved)
