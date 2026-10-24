"""Concise final overview derived from the completed, reserved comparisons."""
import json
import statistics as st


def final_overview(root, rows, table):
    held = root / 'final_checks/holdout_results.json'
    speed = root / 'final_checks/speed_results.json'
    if not held.exists() or not speed.exists():
        return None
    measured = {r['run']: r['evaluation']['loss']
                for r in json.loads(held.read_text())['rows']}
    exploratory = {}
    for r in rows:
        path = r['wave'] + '/' + r['job']
        if r['wave'] == 'initial':
            path += '/fixed_both_' + str(r['seed'])
        exploratory[path] = r['full_validation_loss']
    timings = {r['model']: r['median_ms']
               for r in json.loads(speed.read_text())['summary']}
    results = []
    for pool in ('original', 'larger'):
        for name, wave, model in (
                ('Rank-1, Hermite basis', 'wave13_selected_long', 'spine_rank1_fused_repeat'),
                ('Rank-1, power basis', 'wave16_combination_long', 'combo_rank1_powers')):
            paths = [f'{wave}/{pool}_{model}_{seed}' for seed in (11, 22, 33)]
            controls = [f'wave8_long/{pool}_original_mlp_{seed}' for seed in (11, 22, 33)]
            evgap = st.mean(exploratory[k] - exploratory[m] for k, m in zip(paths, controls))
            hgap = st.mean(measured[k] - measured[m] for k, m in zip(paths, controls))
            wins = sum(measured[k] < measured[m] for k, m in zip(paths, controls))
            results.append([pool.capitalize(), name, '73,600', f'{evgap:+.6f}',
                            f'{hgap:+.6f}', f'{wins}/3'])
    hybrid = [f'wave17_mixed_ranks/mixed_balanced_mlp32_{s}' for s in (11, 22, 33)]
    parent = [f'wave17_mixed_ranks/mixed_balanced_{s}' for s in (11, 22, 33)]
    hybrid_gap = st.mean(measured[k] - measured[m] for k, m in zip(hybrid, parent))
    hybrid_wins = sum(measured[k] < measured[m] for k, m in zip(hybrid, parent))
    timing_rows = [[name, f'{timings[key]:.3f}'] for name, key in (
        ('Original MLP', 'original_mlp'),
        ('Uniform rank-1 KS2', 'selected_spine_rank1_fused_repeat'),
        ('Uniform rank-1 KS2, power basis', 'selected_combo_rank1_powers'),
        ('Mixed KS2 ranks', 'mixed_balanced'),
        ('Mixed KS2 + MLP subspace', 'mixed_balanced_mlp32'))]
    return '''<p><strong>KS2 can modestly beat the unchanged MLP in this harness, but the gains depend on the training pool and geometry basis.</strong>
    Learned projections, ReLU² output features and smaller spine charts close the earlier demuxed-history architecture's loss gap.
    The table compares two useful rank-1 candidates after 100,000 updates. Negative gaps favor KS2; losses are nats per byte.
    Every model has four FFNs. The original MLP has <strong>73,728 learned parameters per FFN (294,912 across four)</strong>;
    uniform rank-1 KS2 has <strong>73,600 per FFN (294,400 across four)</strong>.</p>''' + table(
        ['Training pool', 'KS2 candidate', 'Learned per FFN', 'Exploratory gap vs MLP',
         'Fresh reserved gap vs MLP', 'Reserved seed wins'], results) + f'''
    <p><strong>Combining wins is conditional.</strong> None of the eleven new combinations beats the same-extension uniform rank-1 mean at 15k.
    At 100k, rank-1 plus the power basis improves all three original-pool seeds over both MLP and the earlier rank-1 parent on exploratory validation.
    On the larger pool, all four combinations trail the earlier Hermite-basis rank-1 mean. The reserved comparisons above provide a separate confirmation check;
    they were not used to choose configurations. Same-architecture repeat variation and only three seeds limit claims about small differences.</p>
    <p><strong>The requested MLP subspace gives no clear incremental gain.</strong> The mixed cell includes two rank-16 charts in every FFN.
    Replacing 32 rank-1 features with a separate ReLU² MLP branch leaves 368 projected features and uses <strong>72,916 learned parameters per FFN</strong>
    versus 73,172 for its mixed KS2-only parent. The exploratory mean change is −0.000217; on fresh reserved text it is {hybrid_gap:+.6f}
    ({hybrid_wins}/3 seed wins). The mixed variants do not improve the uniform rank-1 exploratory mean. These are 15k tests of spine-only KS2;
    they do not isolate the benefit of rank-16 charts or test mixed-rank full edge/damping cells.</p>
    <p><strong>Efficiency is measured separately from loss.</strong> The following are median whole-step times from five randomized rounds on one MI100,
    B4 × 64 bytes, FP32. The MLP branch uses the same input/readout projections as the KS2 coordinates.</p>''' + table(
        ['Model', 'Milliseconds per training step'], timing_rows) + f'''
    <p>Fused geometry reduces cost, while shared RWKV blocks dominate the isolated block timings. This identifies a compute target, not an attention-capacity explanation for loss.
    Compact Hadamard storage is about 12× smaller but its tested training step is 2.45× slower. Rank-1 provides more independent scalar geometry inputs;
    scale-only interventions do not explain its advantage. Rank grouping, warp count and basis conditioning remain partly confounded.</p>
    <p><strong>{len(rows)} completed training runs; 259 fresh reserved evaluations.</strong> The study uses the original MLP as its sole primary control.
    All final training starts preceded the two-hour extension's cutoff; in-flight runs then finished. Detailed comparisons, prior RHAC/RWKV evidence,
    numerical checks and all outcomes follow. No claim of a universal KS2 advantage is warranted.</p>
    <p><a href="all_runs.csv">All run metrics (CSV)</a> · <a href="final_checks/holdout_results.json">Fresh reserved scores</a> ·
    <a href="final_checks/speed_results.json">Controlled timings</a> · <a href="goal_completion_review.json">Completion review</a></p>'''
