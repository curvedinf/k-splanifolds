"""Build a standalone, auditable HTML report from completed study artifacts."""
import argparse
import base64
import datetime
import html
import json
from pathlib import Path
import statistics as st
from study_collect import collect


def esc(x):
    return html.escape(str(x))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
    rows=collect(a.root)
    state=json.loads((a.root/'study_state.json').read_text())
    parts=[]
    def section(title, body):
        ident=title.lower().replace(' ','-').replace('/','-')
        parts.append((title,ident,body))
    def artifact_link(path,label):
        if (a.root/path).exists():return f'<a href="{esc(path)}">{esc(label)}</a>'
        return f'{esc(label)} (pending)'
    def lookup(wave,job,seed=None):
        found=[r for r in rows if r['wave']==wave and r['job']==job and (seed is None or r['seed']==seed)]
        return found[0] if found else None
    def score(wave,job,seed=None):
        r=lookup(wave,job,seed)
        return f"{r['full_validation_loss']:.6f}" if r else 'pending'
    def run_table(items):
        out=[]
        for wave,job,label in items:
            r=lookup(wave,job)
            if r:
                out.append([esc(label),f"{r['parameters']:,}",r['seed'],f"{r['full_validation_loss']:.6f}",f"{r['ms_per_step']:.2f}",f'<a href="{esc(r["source"])}">result</a>'])
        return table(['Configuration','Learned parameters','Seed','Loss ↓','ms/step ↓','Evidence'],out)
    def cohort(label, members):
        rs=[lookup(*m) for m in members];rs=[r for r in rs if r]
        vals=[r['full_validation_loss'] for r in rs]
        return [esc(label),len(vals),f'{st.mean(vals):.6f}' if vals else 'pending',f'{st.stdev(vals):.6f}' if len(vals)>1 else '—',', '.join(f"{r['seed']}: {r['full_validation_loss']:.6f}" for r in rs)]
    def fig(name,caption):
        if not (a.root/'figures'/f'{name}.svg').exists():return ''
        encoded=base64.b64encode((a.root/'figures'/f'{name}.svg').read_bytes()).decode('ascii')
        return f'<figure><img src="data:image/svg+xml;base64,{encoded}" alt="{esc(caption)}"><figcaption>{esc(caption)} <a href="figures/{name}.png">PNG</a> · <a href="figures/{name}.svg">SVG</a></figcaption></figure>'

    long=[r for r in rows if r['steps']>=100000]
    done=(a.root/'final_checks/complete.json').exists()
    for audit_name in ('audit.json','protocol_audit.json'):
        audit_path=a.root/audit_name
        done=done and audit_path.exists() and json.loads(audit_path.read_text())['status']=='PASS'
    status='Final' if done else 'Working report — long-run and reserved-holdout results still being collected'
    long_overview=[]
    for pool in ('original','larger'):
        pairs=[]
        for seed in (11,22,33):
            k=lookup('wave8_long',f'{pool}_spine46_{seed}');m=lookup('wave8_long',f'{pool}_original_mlp_{seed}')
            if k and m:pairs.append((k['full_validation_loss'],m['full_validation_loss']))
        if pairs:
            long_overview.append(f'On the {pool} pool, learned-spine KS2 averages {st.mean(k for k,m in pairs):.6f} '
                                 f'versus {st.mean(m for k,m in pairs):.6f} for the original MLP '
                                 f'({len(pairs)} paired seeds; KS2 minus MLP {st.mean(k-m for k,m in pairs):+.6f}).')
    frozen_large=[lookup('wave10_selective_long',f'larger_frozen46_{seed}') for seed in (11,22,33)]
    if all(frozen_large):
        references=[lookup('wave8_long',f'larger_original_mlp_{seed}') for seed in (11,22,33)]
        ksmean=st.mean(r['full_validation_loss'] for r in frozen_large);mlmean=st.mean(r['full_validation_loss'] for r in references)
        wins=sum(k['full_validation_loss']<m['full_validation_loss'] for k,m in zip(frozen_large,references))
        long_overview.append(f'The extension finds frozen-spine KS2 at {ksmean:.6f} on the larger pool versus {mlmean:.6f} for the fixed MLP, with {wins}/3 seed wins (gap {ksmean-mlmean:+.6f}). This modest exploratory advantage is checked on the reserved interval below.')
    long_overview='<p><strong>100k results:</strong> '+' '.join(long_overview)+'</p>' if long_overview else ''
    selected_overview=[]
    for pool in ('original','larger'):
        paired=[(lookup('wave13_selected_long',f'{pool}_spine_rank1_fused_repeat_{seed}'),lookup('wave8_long',f'{pool}_original_mlp_{seed}')) for seed in (11,22,33)]
        if all(k and m for k,m in paired):
            kmean=st.mean(k['full_validation_loss'] for k,m in paired);mmean=st.mean(m['full_validation_loss'] for k,m in paired)
            selected_overview.append(f'{pool.capitalize()} pool: rank-1 KS2 {kmean:.6f}, unchanged MLP {mmean:.6f}; paired gap {kmean-mmean:+.6f}, {sum(k["full_validation_loss"]<m["full_validation_loss"] for k,m in paired)}/3 seed wins.')
    selected_overview='<p><strong>Uniform rank-1 with the original Hermite basis, 100k updates:</strong> '+' '.join(selected_overview)+' The rank-1 FFN has 73,600 learned parameters per layer versus 73,728 for the original MLP. This original-basis variant has a larger-pool advantage without a smaller-pool lead; the power-basis combination is reported separately below.</p>' if selected_overview else ''
    section('Findings at a glance',f'''
    <p><strong>The objective is to improve KS2 against the original, unchanged MLP reference.</strong>
    The strongest quality directions are learned input/output routing, quadratic output features and smaller spine charts.
    A rank-1 spine with the original basis leads the original MLP consistently across three seeds on the larger training pool at 100k updates.
    The ranking depends on training pool, horizon and geometry parameterization; the combination comparisons below test whether improvements accumulate. These results do not establish a universal KS2 advantage.
    At the user's correction, queued MLP variants were canceled and the long-run comparisons returned to
    the original MLP initialization and optimizer. Completed alternate-MLP experiments are retained as diagnostics only,
    not substituted for the reference or used to manufacture a favorable comparison.</p>
    <p>The original demuxed-history KS2 architecture scores about 1.667 at 15,000 steps. Learned projections bring
    a similar-budget KS2 model to about 1.633, and a ReLU-squared feature transform brings it to about 1.624.
    The original MLP control averages about 1.621. Learned spine-only KS2 is approximately tied at this short horizon;
    frozen spine-only KS2 averages about 1.617 across three seeds. The longer-horizon comparisons below show that the ranking depends on the training regime and data pool.
    These are full-shard losses in <strong>nats per byte</strong>; lower is better.</p>
    <p>KS2 geometry is active and affects predictions. Frozen geometry remains input dependent through the learned
    projections, and competes closely with fully learned geometry. A rank-8 spine evaluates its six basis functions at
    one scalar summary shared by eight outputs. Rank-1 charts provide more distinct scalar responses and different
    basis conditioning at the same projected width. However, simply expanding the rank-8 geometry-input scale worsens
    the replicated short-run loss; compressing rank-1's scale retains its short-run performance. The averaging-induced
    scale reduction alone therefore does not explain the rank result.</p>
    <p>Fused geometry kernels pass numerical forward/backward checks and reduce training cost. They do not change
    the mathematical architecture, although small floating-point differences can change long training trajectories.
    Raw speed, learned parameter count and fixed projection storage are all reported separately. Compact Hadamard codes
    reduce fixed-buffer storage about twelvefold but make the tested whole training step 2.45 times slower. After simplifying
    and fusing KS2 geometry, the shared RWKV blocks dominate the measured block computation; that is evidence about compute
    cost, not proof that attention capacity limits loss.</p>
    <p><strong>{len(rows)} completed training runs</strong>, including {len(long)} long-horizon runs, are indexed below.
    This count excludes smoke tests, kernel probes and timing-only repetitions. Status: {esc(status)}.</p>
    <p><a href="all_runs.csv">Download all run metrics (CSV)</a> · <a href="all_runs.json">JSON with configurations</a> ·
    <a href="study_state.json">Study timeline</a> · {artifact_link('final_checks/holdout_results.json','Reserved holdout')} ·
    {artifact_link('final_checks/speed_results.json','Controlled speed measurements')}</p>'''+selected_overview+long_overview)

    section('Protocol and interpretation','''
    <p>The exploration starts at 2026-10-22 23:22:13 UTC. The original four-hour window ended at 03:22:13 UTC; the user explicitly extended work by another four hours, through 2026-10-23 07:22:13 UTC.
    New training jobs stop launching at the deadline; already running jobs finish. Final scoring, auditing and reporting follow.
    Four AMD MI100 GPUs run independent experiments concurrently, one model per GPU. This is not tensor parallelism.</p>
    <p>Unless explicitly labeled otherwise, models have four layers, a 96-dimensional main stream, a 256-byte vocabulary,
    learned embeddings and an untied logit head. Training uses FP32, captured HIP graphs, AdamW, global gradient clipping
    at 1, weight decay 0.01 and a cosine learning rate from 0.001 to 0.0002. Batch size is 4 and sequence length is 64:
    256 target bytes per update, 3.84 million targets in a 15,000-step screen and 25.6 million in a 100,000-step run. Sequence length 64 means 64 bytes here, not 64 source GPT-2 tokens.</p>
    <p>Original training data: 4,468,985 decoded bytes from the first million source tokens. The larger training pool has
    44,766,641 bytes from the first ten million tokens and contains the original pool as an exact prefix. Validation uses
    a separate 444,316-byte shard. Full-shard scoring averages four overlapping 64-byte tilings (offsets 0, 16, 32, 48),
    totaling 1,777,024 scored target visits. These are overlapping visits, not 1.78 million independent examples.</p>
    <p>A second, disjoint 451,714-byte validation interval was reserved before final model comparison. After partial final scoring,
    the user requested a further two-hour combination study. That phase reserves a fresh 437,621-byte interval, documented below,
    and retains the earlier partial scores as diagnostics. No architecture tuning uses either reserved interval's scores.
    All exploratory validation comparisons remain subject to adaptive selection. Three seeds give useful replication,
    not a precise population confidence interval; tables show sample standard deviation, not standard error.</p>
    <p>Initialization and sampling are paired within comparisons where shapes permit. Changing architecture changes random-number
    consumption and the optimization path; equal seed does not make every tensor identical. Shared attention, embedding and
    head initialization is checked explicitly. Warmup and timing updates are reset before training. Source snapshots, data hashes,
    commands, optimizer counters and restored-checkpoint evaluation accompany each run.</p>
    <p>The earliest 20 runs recorded source hashes without per-run source copies. Their snapshots were subsequently recovered
    only after verifying exact equality with those recorded hashes; the recovery ledger is
    <a href="recovered_source_snapshots.json">recovered_source_snapshots.json</a>. Later workers snapshot sources directly.
    The final <a href="audit.json">artifact audit</a> checks source hashes, checkpoint steps, finite tensors and optimizer counters.</p>
    <p>Fresh 100,000-step runs use a cosine schedule over the full 100,000 steps. Their 15k/50k intermediate checkpoints therefore
    do not match independently annealed 15k runs. Earlier 105k experiments used a different schedule: a 15k cosine followed by
    constant 0.0002. Historical results provide context, not a controlled head-to-head with this session.</p>''')

    section('Architectures and the route information takes','''
    <h3>Selected demuxed-history baseline</h3>
    <pre>main[96] → LayerNorm → RWKV-7 → add to main
                                  ↓ post-attention main[96]
    concat(post-attention main, side[2048], prior post-attention and post-decoder main states)
      → normalization / causal time mixing → fixed Hadamard mux → 64 × 8 coordinates
      → coordinates + KS2 geometry → fixed Hadamard demux → learned output gains
      → add to (main[96], side[2048])
    keep post-attention and post-decoder main states as next layers' history
    final main[96] → normalization → learned logits[256]</pre>
    <p>The side residual bypasses attention, embeddings and logits; it is a decoder-to-decoder state.
    Decoder inputs have widths 2144, 2336, 2528 and 2720. Attention continues to receive a direct 96D residual state.
    Only decoder updates are demuxed into that state. The final side output does not reach the task loss; its 2048 output
    gains are counted in the parameter budget even though they receive no task gradient.</p>
    <h3>Learned-projection KS2</h3>
    <pre>main → RWKV residual update
         → normalization / causal FFN time mixing
         → learned 96-to-(charts × 8) input projection
         → coordinates + KS2 geometry
         → optional feature activation (usually ReLU²)
         → learned (charts × 8)-to-96 output projection → residual add</pre>
    <p>The canonical full model has 31 rank-8 charts and 613,168 learned parameters. Undamped KS2 removes the denominator
    calculation and uses 33 charts at 615,120 parameters. Spine-only KS2 retains the spine and its two warps, removes edge
    terms, and uses 46 charts at 612,848 parameters. The original MLP uses learned projections and ReLU² at
    width 384, totaling 615,936 parameters. An archived width-368 diagnostic has 603,648; it is not the primary control.</p>
    <p>These learned-projection variants intentionally change the selected baseline's decoder routing and omit its 2048D side/history
    path. They isolate whether the KS2 cell can work with conventional adaptive routing. Their improvements cannot all be
    attributed to a single projection change in the original baseline.</p>
    <p>The original MLP input weights initialize uniformly within ±0.5/√width; the KS2 projection uses ±1/√width. The
    alternate MLP diagnostics match the latter range. Output projections start at zero, as in the original residual architecture.</p>
    <h3>Rank-1 spine candidate used in the combination study</h3>
    <pre>normalized / time-mixed main[96]
      → learned projection → 368 scalar coordinates c
      → each coordinate gets its own spine g(c): six coefficients + two warp parameters
      → ReLU(c + g(c))²
      → learned projection → residual update[96]</pre>
    <p>For each scalar chart, the basis argument is 4 tanh(c/4). The implementation also clamps coordinates to the FP16
    finite range while evaluating geometry in FP32. This candidate keeps the KS2 spine but omits the full cell's edge
    and damping terms. It acts as a bank of learned scalar nonlinearities between two learned projections, rather than
    using cross-coordinate geometry within each chart. All 96 main-stream inputs can influence every scalar through
    the learned projection. The two projections cost 70,656 parameters per FFN, and the 368 × 8 spine parameters add
    2,944, for 73,600 total. Freezing those 2,944 coefficients retains their nonlinear function; learned bias calibration
    adds 368 parameters before the ReLU² transform.</p>''')

    section('FFN-only parameter comparison',table(['FFN architecture','Learned per FFN','Learned across four FFNs','Fixed geometry per FFN'],[
        ['Original MLP','73,728','294,912','0'],
        ['Full KS2, rank 8 × 31','73,036','292,144','0'],
        ['Undamped KS2, rank 8 × 33','73,524','294,096','0'],
        ['KS2 rank-1 spine','73,600','294,400','0'],
        ['KS2 rank-4 spine / power basis','73,048','292,192','0'],
        ['KS2 rank-8 spine / power basis','72,956','291,824','0'],
        ['Frozen rank-1 spine + bias','71,024','284,096','2,944'],
        ['Frozen rank-4 spine + bias','71,024','284,096','2,392'],
        ['Frozen rank-4 spine','70,656','282,624','2,392'],
        ['Frozen rank-8 spine','70,656','282,624','2,300'],
        ['Frozen full geometry, rank 8 × 46','70,656','282,624','37,720'],
    ])+'''<p>These counts include the FFN input projection, geometry and output projection only. RWKV, embeddings, the logits head and surrounding normalization/time-mixing parameters are excluded. Frozen geometry values occupy storage and participate in computation even though they are not optimized. Frozen full geometry therefore uses 108,376 learned plus fixed values per FFN, exceeding the MLP's 73,728. The rank-1 comparison is nearly equal in both learned FFN budget and total model budget.</p>''')

    section('Where the parameter budget goes',table(['Architecture','RWKV','Embedding + head','Learned FFN inputs','Learned FFN outputs / gains','Learned geometry','Norm / mixing / other','Total'],[
        ['Original MLP','269,568','49,152','147,456','147,456','0','2,304','615,936'],
        ['Demuxed history, 64 charts','269,568','49,152','0','8,576','209,920','30,336','567,552'],
        ['Dense full KS2, 31 charts','269,568','49,152','95,232','95,232','101,680','2,304','613,168'],
        ['Dense undamped KS2, 33 charts','269,568','49,152','101,376','101,376','91,344','2,304','615,120'],
        ['Dense spine KS2, 46 charts','269,568','49,152','141,312','141,312','9,200','2,304','612,848'],
        ['Frozen spine KS2, 46 charts','269,568','49,152','141,312','141,312','0','2,304','603,648'],
        ['Learned spine, rank 1 × 368','269,568','49,152','141,312','141,312','11,776','2,304','615,424'],
        ['Frozen spine, rank 4 × 92','269,568','49,152','141,312','141,312','0','2,304','603,648'],
        ['Frozen full geometry, rank 8 × 46','269,568','49,152','141,312','141,312','0','2,304','603,648'],
    ])+'''
    <p>Counts sum all four layers. For a dense rank-8 KS2 chart at model width 96, the two projections cost 1,536 parameters.
    Full geometry adds 820, undamped geometry 692, and spine geometry 50. With four FFNs, the dense family has
    321,024 common parameters plus four times the chart count times its per-chart cost.</p>
    <p>Full geometry uses about 35% of the dense 31-chart decoder budget; spine geometry uses about 3% of the 46-chart budget.
    Thus the matched-total-budget spine configuration also has 368 projected features instead of 248. Its gain over the full
    cell cannot be attributed solely to the removed terms. Learned-versus-frozen spine preserves the 368-feature width and
    initialized tensors, providing a direct KS2 comparison of geometry training. Frozen geometry retains 9,200 fixed values
    even though they no longer count as learned parameters. Frozen rank-4 spine retains 9,568 fixed geometry values;
    frozen full geometry at 46 charts retains 150,880. The latter has 754,528 learned plus fixed coefficients and therefore
    exceeds the original MLP's total coefficient storage despite its lower learned parameter count. Power-basis spine has
    the same counts as learned rank-8 spine; its coefficient coordinates change, not its function-class dimension.</p>''')

    short_groups=[cohort('Original MLP', [('initial','mlp',11),('initial','mlp',22),('wave3','mlp_seed33',33)]),
      cohort('MLP384, matched input initialization',[('wave3','matched_init_mlp384',11),('wave4_activation','matched_mlp384_22',22),('wave4_activation','matched_mlp384_33',33)]),
      cohort('MLP368, matched input initialization',[('wave4_activation','matched_mlp368',11),('wave6_budget','matched_mlp368_22',22),('wave6_budget','matched_mlp368_33',33)]),
      cohort('Full KS2 31 + ReLU², eager',[('wave2','dense31_relu2',11),('wave3','dense_relu_seed22',22),('wave3','dense_relu_seed33',33)]),
      cohort('Undamped KS2 33 + ReLU², fused', [('wave4_activation',f'undamped_fused_{s}',s) for s in (11,22,33)]),
      cohort('Spine KS2 46 + ReLU², fused', [('wave4_activation',f'spine_fused_{s}',s) for s in (11,22,33)]),
      cohort('Frozen spine geometry, 368 features',[('wave4_activation','spine_frozen',11),('wave6_budget','spine_frozen_22',22),('wave6_budget','spine_frozen_33',33)])]
    primary_groups=[short_groups[i] for i in (0,3,4,5,6)]
    section('Replicated short-run comparisons',table(['Model','Seeds','Mean loss ↓','Seed SD','Individual scores'],primary_groups)+'''
    <p>The table uses named, preselected replications rather than choosing the lowest repeat for each seed.
    Same-seed repeats are retained in the complete ledger. For example, undamped seed 11 scored 1.618906 in the first eager
    screen, but 1.623443 in an eager repeat and 1.625468/1.627479 in two fused runs. The most favorable initial value is
    not sufficient evidence of a lead. These variations limit claims about differences of a few thousandths.</p>
    <p>The full 31-chart KS2 + ReLU² model remains slightly behind the original MLP across its initial three seeds.
    Learned spine KS2 is close to the original MLP after replication. Frozen spine KS2 beats that fixed reference on all
    three short-run seeds by roughly 0.0031–0.0044 nats/byte. The long-horizon and reserved-validation comparisons below
    assess whether that ordering persists; a short-run result is not a universal superiority claim.</p>'''+fig('ks2_progress','KS2 configurations compared with the unchanged original MLP. Error bars show seed SD; counts are explicit. Configurations differ in routing and activation as well as geometry.')+fig('short_learning_curves','Learning curves use the fixed sampled validation metric, not full-shard scores.'))

    section('Routing, activation and optimizer interventions',run_table([
      ('wave1','baseline_repeat','Original demuxed-history baseline'),
      ('wave1','geometry_lr4','History baseline, geometry LR ×4'),
      ('wave1','geometry_only','History baseline, remove coordinate bypass'),
      ('wave2','dense31','Learned input and output projections'),
      ('wave2','dense31_relu2','Learned projections + ReLU²'),
      ('wave4_activation','ks2_then_square','Learned projections + square'),
      ('wave4_activation','ks2_then_relu','Learned projections + ReLU'),
      ('wave4_activation','ks2_then_silu','Learned projections + SiLU'),
      ('wave2','dense31_gelu','Learned projections + GELU'),
      ('wave2','dense31_output55','Carry history output LR ×5.5 into dense KS2'),
      ('wave3','history_direct40_relu','History + direct learned main readout + ReLU²'),
      ('wave3','history_direct80_random_relu','History + direct learned readout + random ReLU² features'),
    ])+'''
    <p><strong>Quadratic features are a useful intervention.</strong> ReLU² and an unrectified square both reach about 1.624
    in the full KS2 decoder, whereas ReLU and SiLU are about 1.642. This points toward feature magnitude/order rather than
    rectification alone. It does not show a KS2-specific advantage: pure square and ReLU² MLP controls are also included.</p>
    <p><strong>Projection learning rate does not transfer automatically.</strong> The original history scalar gains use an LR
    multiplier of 5.5. Carrying that multiplier into a dense learned output projection worsens loss to about 1.653.
    Geometry LR ×4 helps a plain KS2 cell, but stacking it with ReLU² worsens the stronger cell; larger multipliers are worse.
    LR multipliers adjust weight decay inversely so the per-step decay amount is held fixed.</p>
    <p><strong>Output factorization interacts with activation.</strong> In the history model, replacing gain-times-matrix
    main readout with a direct matrix helps ReLU² cases substantially, but does not rescue the plain KS2 cell by itself.
    A history model with random ReLU² features and a learned main readout also reaches about 1.624. Recovering adaptive
    readout is useful, but that benefit is not unique to learned KS2 geometry.</p>
    <p>The factorized output also changes optimization: in y = gain × (Wf), a zero initial gain makes the initial gradient
    of a trainable W zero. A directly parameterized, zero-initialized W receives its readout gradient immediately.
    This is a mechanism implied by the parameterization, not a separate proof that it explains the entire observed loss
    difference. The strong random-feature/readout result also means these experiments do not establish that every input
    mux must become learned; adaptive output routing can recover much of the gap while retaining a fixed input mix.</p>
    <p>Cheap learned rotations, mux affine transforms, coordinate RMS normalization, changing coordinate scale, removing the
    bypass and broad geometry-LR sweeps all appear in the full ledger. Most do not close the gap. Moving more of the fixed
    total budget into width-128 attention yields about 1.642 for both KS2 and MLP, worse than the width-96 allocation.</p>'''+fig('geometry_learning_rate','Geometry learning rate has architecture-dependent effects.'))

    section('Is geometry inactive or wasted?', '''
    <p>It is not inactive. In the trained history baseline, coordinate RMS grows from roughly 0.80 to 1.06 across layers,
    while geometry-update RMS grows from 0.60 to 1.17. Edge terms contribute more than spine terms in this trained model.
    Removing geometry after training raises sampled loss; removing the coordinate bypass can hurt even more. These are
    interventions on a trained solution, not comparisons to retrained alternatives.</p>
    <p>In the learned-projection ReLU² model, geometry changes the sign of about 12–16% of coordinates before ReLU² and
    changes the positive feature vector by a relative norm of roughly 0.28–0.41. Zeroing all geometry increases the probe
    loss from about 1.699 to 1.833. The model relies on geometry even if a separately trained wider MLP can do better.</p>
    <p>The damping contribution is weak, around 0.024 in the original trained history probe. Removing it at 31 charts
    preserves short-run quality while saving 15,872 parameters. Reinvesting that budget in 33 charts is therefore a sensible
    candidate. The full geometry has 820 parameters per chart, undamped geometry 692, and spine-only geometry 50.</p>
    <p>The spine-only model allocates almost all its decoder budget to a wide learned projection, so its improvement over
    the full cell may partly come from width. Within KS2, freezing spine geometry preserves the 368-feature projection and
    initialized tensors: its three-seed short-run mean is 1.617121 versus 1.620510 for learned spine. The long-run comparison
    tests whether that advantage survives the longer schedule. Geometry dependence alone does not settle parameter efficiency.</p>
    <p>Probes are stored beside the relevant runs as <code>mechanism_probe.json</code> and <code>geometry_terms_probe.json</code>.
    Probe losses use a smaller sample and must not be compared numerically with full-shard tables.</p>''')

    gp=a.root/'geometry_trajectory.json'
    if gp.exists():
        geometry_rows=[]
        for row in json.loads(gp.read_text())['rows']:
            if row['step']!=100000:continue
            def series(key,percent=False):
                return ', '.join(f"{v[key]*(100 if percent else 1):.1f}" if percent else f"{v[key]:.3f}" for v in row['layers'])
            geometry_rows.append([esc(row['job']),series('geometry_relative_rms'),series('geometry_positive_fraction',True),
                                  series('coordinate_positive_fraction',True),series('updated_positive_fraction',True),
                                  series('spine_warp_saturated_fraction',True)])
        section('What learned geometry does over training',table(['100k checkpoint, seed 11','Geometry / coordinate RMS','Positive geometry %','Positive coordinates before geometry %','Positive after geometry %','Saturated spine warp %'],geometry_rows)+'''
        <p>Each cell contains layers 1–4 in order. This CPU probe uses 1,024 fixed targets from the exploratory validation
        shard. It also records initialization, 15k and 50k checkpoints in <a href="geometry_trajectory.json">the trajectory
        artifact</a>. It does not inspect the reserved validation interval.</p>
        <p>On the original-pool seed-11 runs, learned spine and undamped geometry become predominantly negative before the
        ReLU² transform. For learned spine, only about 2% of geometry entries are positive at 100k; geometry reduces the
        positive-coordinate fraction from about 20–32% to 10–17%, depending on layer. Frozen spine has roughly balanced
        geometry signs and barely changes that fraction; its learned input projection supplies much of the sparsity instead.</p>
        <p>This suggests a concrete role for learned KS2 in these solutions: input-dependent thresholding before quadratic
        features. Geometry includes both a substantial average offset and sample-dependent variation, so it is not merely
        a constant bias. The observation motivates KS2-specific tests of threshold initialization and learning, but does not
        show that a simpler bias would reproduce the result.</p>
        <p>Widespread saturation of the spine warp is not evident here. Its absolute tanh output exceeds 0.95 on fewer than
        0.2% of sampled entries in every layer of the learned-spine 100k checkpoint. The average learned warp scale decreases
        from 1.137 at initialization to about 0.87–0.89. Geometry remains active: its RMS is about 0.76–1.45 times coordinate
        RMS, and its relative change to the positive quadratic feature vector is about 0.30–0.44. These are descriptive
        single-seed measurements, not proof that the same mechanism governs every configuration.</p>''')

    cp=a.root/'gradient_clip_probe.json'
    if cp.exists():
        clip_rows=[]
        for r in json.loads(cp.read_text())['rows']:
            clip_rows.append([esc(r['run']),f"{100*r['mean_geometry_squared_norm_fraction']:.2f}%",r['clipped_batches'],f"{r['mean_clip_factor']:.4f}",f"{r['mean_clip_factor_excluding_geometry']:.4f}"])
        section('Does geometry dominate gradient clipping?',table(['Checkpoint','Geometry share of squared gradient norm','Clipped batches / 8','Mean global clip factor','Factor if geometry omitted from norm'],clip_rows)+'''
        <p>Freezing geometry removes its entries from the global gradient norm as well as preventing its updates. This CPU
        diagnostic measures that possible coupling on eight fixed training minibatches, with no optimizer steps.
        At the tested seed-11 endpoints, learned geometry contributes about 1.6–3.4% of squared gradient norm. Excluding it
        from the clipping norm changes the mean clip factor by roughly 0.006–0.013 in absolute terms.</p>
        <p>These observations do not support geometry dominating global clipping at these checkpoints. They do not rule out
        different behavior earlier in training. AdamW normalizes using its moment estimates, so clipping factors are not
        direct learning-rate multipliers or measured parameter-update changes. See <a href="gradient_clip_probe.json">
        per-batch measurements and protocol</a>.</p>''')

    fp=a.root/'feature_probe.json'
    if fp.exists():
        feature_rows=[]
        for row in json.loads(fp.read_text())['rows']:
            feature_rows.append([esc(row['run']),', '.join(f"{v['features']['covariance_participation_rank']:.1f}" for v in row['layers']),
                                 ', '.join(f"{v['features']['rms']:.2f}" for v in row['layers'])])
        section('Feature-spectrum diagnostic',table(['Checkpoint','Feature participation rank, layers 1–4','Feature RMS, layers 1–4'],feature_rows)+'''
        <p>This CPU probe uses the same 1,024 validation targets for each model and measures the vector entering the output
        projection. Participation rank is trace(C)²/trace(C²) for centered feature covariance. It is scale-dependent and measures
        variance concentration on this sample, not intrinsic information capacity.</p>
        <p>There is no simple monotonic relationship between feature rank and validation quality. The plain learned-projection
        KS2 model has high feature rank in some layers yet worse loss than the quadratic variants. Some strong MLP and KS2
        models have a low-rank, high-magnitude last or penultimate feature distribution. This does not support a blanket
        explanation that all poor KS2 results come from collapsed or unused coordinates. The raw statistics, entropy ranks and
        quantiles are in <a href="feature_probe.json">feature_probe.json</a>.</p>''')

    budget=[]
    for c,h in [(2,25),(4,50),(8,99),(16,197)]:
        ks=[lookup('wave6_budget',f'ks2_c{c}_{s}') for s in (11,22,33)]
        ml=[lookup('wave6_budget',f'mlp_h{h}_{s}') for s in (11,22,33)]
        pairs=[(k,m) for k,m in zip(ks,ml) if k and m]
        if pairs:
            gaps=[k['full_validation_loss']-m['full_validation_loss'] for k,m in pairs]
            budget.append([f'{c} rank-8 / {h} MLP features',len(pairs),f"{pairs[0][0]['parameters']:,} / {pairs[0][1]['parameters']:,}",f"{st.mean(k['full_validation_loss'] for k,m in pairs):.6f}",f"{st.mean(m['full_validation_loss'] for k,m in pairs):.6f}",f'{st.mean(gaps):+.6f}',f'{st.stdev(gaps):.6f}' if len(gaps)>1 else '—'])
    section('Small decoder budgets',table(['KS2 charts / MLP width','Paired seeds','Total parameters KS2 / MLP','KS2 mean','MLP mean','KS2 − MLP','Paired gap SD'],budget)+'''
    <p>These runs replace all four FFNs and keep attention, width, data and training horizon fixed. MLP input scale matches
    the KS2 projection scale. Widths are rounded upward, so the MLP receives slightly more total parameters (less than
    0.2% extra). Unlike earlier inner-two-layer budget tests, this sweep directly tests all-decoder parameter allocation.
    Negative gaps favor KS2. All twelve completed seed/budget pairs favor MLP: the mean KS2 penalties are approximately
    0.0066, 0.0051, 0.0087 and 0.0106 nats/byte. This is evidence against a small-budget advantage for this particular
    one-step full-geometry + ReLU² implementation under the tested training protocol.</p>'''+fig('budget_curve','Matched short-run total-parameter budgets, three paired seeds where complete.'))

    scale_rows=[]
    for seed in (11,22,33):
        lo=lookup('wave7b_init',f'mlp384_inputscale1_{seed}');hi=lookup('wave7b_init',f'mlp384_inputscale2_{seed}')
        if lo and hi:
            scale_rows.append([seed,f"{lo['full_validation_loss']:.6f}",f"{hi['full_validation_loss']:.6f}",f"{hi['full_validation_loss']-lo['full_validation_loss']:+.6f}"])
    section('Input-scale control with identical random directions',table(['Seed','Original input scale','Twice input scale','Twice − original'],scale_rows)+'''
    <p>The original MLP and FeatureCell-based matched MLP have the same mathematical FFN shape but use different initialization
    paths as well as different input scales. Their difference alone therefore cannot isolate scale. This additional control
    constructs the same 384-wide MLP twice from the same seed, then doubles only its input weights in one arm. Attention,
    embeddings, output weights and all random directions are identical; optimizer and training positions are paired.</p>
    <p>This completed diagnostic tests input scale directly. It does not replace the original MLP reference, and no further
    MLP initialization search follows it. Its relevance to future work is that KS2 input scaling deserves controlled tests
    within KS2 itself.</p>''')

    section('Repeated geometry with shared parameters',run_table([
        ('wave7c_iteration',f'{name}_{seed}',f'{label}, seed {seed}')
        for seed in (11,22) for name,label in [
            ('plain_two','Two geometry iterations, no outer activation'),
            ('relu_two','Two geometry iterations + ReLU²'),
            ('relu_two_half','Two half-size geometry iterations + ReLU²'),
            ('relu_four_quarter','Four quarter-size geometry iterations + ReLU²')]])+'''
        <p>The earlier reference denoising KS2 used two geometry iterations; the language-model study initially used one.
        This follow-up applies the same learned geometry repeatedly before the output projection, with shared parameters:
        c ← c + αG(c). Every arm retains 31 rank-8 charts and 613,168 learned parameters. α=1 in the full two-step update,
        α=1/2 for two half steps, and α=1/4 for four quarter steps. ReLU², when present, is applied only after all iterations.</p>
        <p>The reduced-step variants distinguish repeated nonlinear refinement from simply increasing the total update size.
        The plain two-step forward and all gradients match the reference stabilized cell exactly on CPU at nonzero output
        weights. GPU structural, finite-gradient and causal checks precede training. Repeated iterations cost more compute,
        so equal parameter count does not imply equal runtime.</p>''')

    section('Cross-subspace geometry refinement',run_table([
        ('wave7d_shuffle',f'{name}_{seed}',f'{label}, seed {seed}') for seed in (11,22)
        for name,label in [('shuffle_two','Two updates with cross-subspace shuffle'),('shuffle_two_half','Two half updates with cross-subspace shuffle')]])+'''
        <p>The second geometry update receives coordinates from eight different first-stage charts. A fixed reshape/transpose
        permutation creates those groups and is inverted after the update. The same geometry parameters are reused. This
        tests interactions across the otherwise independent rank-8 groups without adding learned parameters.</p>
        <p>The coordinate residual is exactly preserved when geometry is zero, verified independently. All configurations have
        613,168 parameters; the extra geometry evaluation and permutation increase compute. The unchanged original MLP remains
        the external reference. This is a KS2-only architectural intervention.</p>''')

    geometry_learning_groups=[short_groups[0],short_groups[5],short_groups[6]]+[
        cohort(label,[('wave9_geometry_learning',f'{name}_{seed}',seed) for seed in (11,22,33)])
        for name,label in [('spine_geo01','Geometry LR ×0.1'),('spine_geo001','Geometry LR ×0.01'),
                           ('spine_warps_only','Learn warps only'),('spine_coefficients_only','Learn coefficients only')]]
    section('Which geometry parameters should learn?',table(['Configuration','Seeds','Mean loss','Seed SD','Individual scores'],geometry_learning_groups)+run_table([
        ('wave9_geometry_learning',f'{name}_{seed}',f'{label}, seed {seed}')
        for seed in (11,22,33) for name,label in [('spine_geo01','Spine geometry LR ×0.1'),
        ('spine_geo001','Spine geometry LR ×0.01'),('spine_warps_only','Learn warps; fixed spine coefficients'),
        ('spine_coefficients_only','Learn spine coefficients; fixed warps')]])+'''
        <p>Freezing spine geometry improved the replicated short-run result relative to learning it at the original rate.
        These KS2-only follow-ups isolate two explanations: geometry may need slower optimization, or one parameter family
        may be useful to adapt while another is better held fixed. All variants start from exactly the same geometry values
        as the learned-spine model; only trainability or its learning rate changes.</p>
        <p>There are 46 rank-8 charts in each FFN. Warps-only training retains 368 learned geometry parameters across four
        layers, while coefficient-only training retains 8,832. Fixed geometry tensors still consume storage and computation.
        The original unchanged MLP and already completed learned/frozen spine runs are reused as references; no MLP variant
        is trained for this follow-up.</p>''')

    transfer=[]
    for name in ('soft_mlp','soft_matched_mlp','soft_spine','soft_undamped'):
        transfer.append(cohort(name,[('wave5_transfer',f'{name}_{s}',s) for s in (11,22,33)]))
    section('Standard attention and RWKV projection allocation',table(['Softmax-attention family','Seeds','Mean loss','Seed SD','Individual scores'],transfer)+'''
    <p>The transfer uses causal softmax attention with three 32D heads and RoPE, pre-normalization, and no RWKV FFN time mixing.
    All compared FFNs receive the same attention implementation and paired attention initialization. These models have about
    490–493k parameters, so their absolute losses should not be compared to 615k RWKV models as if only attention changed.</p>
    <p>This test distinguishes a uniquely RWKV-specific failure from a decoder mismatch that persists under ordinary attention.
    The alternate MLP initialization changes results here too. At this short horizon the KS2 variants do not show a consistent broad lead.</p>
    <p>The earlier Hadamard projection sweep identified RWKV gate-up as an apparently inexpensive place to remove learned weights.
    This session tests whether its saved 24,192 parameters can buy wider FFNs. The gate mux adds fixed-code storage as well.
    Historical near-equality on the original history model does not guarantee that it transfers to a different FFN.</p>'''+run_table([('wave5_transfer',f'{name}_{seed}',f'{name}, seed {seed}') for seed in (11,22) for name in ('gate_mlp384','gate_mlp408','gate_mlp416','gate_spine50','gate_undamped35')])+'''
    <p>The 416-feature gate-mux MLP has 616,320 learned parameters, 384 more than the 615,936 original budget (0.062%).
    The 408-feature control stays below budget. Counts are explicit rather than calling every variant exactly matched.</p>''')

    context=[]
    if (a.root/'aligned_context_results.json').exists():
        for row in json.loads((a.root/'aligned_context_results.json').read_text()):
            s=row['scores'];context.append([esc(row['run']),str(s[0].get('targets')),f"{s[0]['loss']:.6f}",f"{s[1]['loss']:.6f}",f"{s[1]['loss']-s[0]['loss']:+.6f}"])
    context_repeats=[]
    for seed in (11,22,33):
        m=lookup('wave7_context','matched_mlp_b4_t256') if seed==11 else lookup('wave7b_init',f'context_mlp_{seed}')
        k=lookup('wave7_context','spine_b4_t256') if seed==11 else lookup('wave7b_init',f'context_spine_{seed}')
        if m and k:context_repeats.append([seed,f"{m['full_validation_loss']:.6f}",f"{k['full_validation_loss']:.6f}",f"{k['full_validation_loss']-m['full_validation_loss']:+.6f}"])
    section('Context length and optimizer updates',table(['Run','Same target visits','64-input context loss','256-input context loss','256 − 64'],context)+'''
    <p>The context study keeps training target count fixed at 3.84 million: batch 1 × length 256 for 15k updates,
    batch 16 × length 64 for 3,750 updates, and batch 4 × length 256 for 3,750 updates. It compares MLP384, spine46 and
    undamped33 at seed 11. The original batch 4 × length 64 × 15k runs serve as references.</p>
    <p>Changing context length changes both usable history and the number of recurrent positions. Changing batch size at
    fixed targets changes the number of optimizer updates. These are separate factors; a single longer-context score
    cannot identify the cause. The table evaluates exactly the same 64 target bytes with 64 versus 256 input context,
    avoiding the different context-position mixture of ordinary full-window averages.</p>
    <p><strong>Most of the apparent longer-context gain comes from inference context, not longer-context training.</strong>
    On the same target positions with 256 input bytes, the MLP trained at length 64 scores 1.573303 and the MLP trained at
    length 256 scores 1.573143. Spine KS2 scores 1.575246 versus 1.576085; undamped KS2 scores 1.581946 versus 1.584891.
    Thus the longer training windows do not improve these seed-11 comparisons meaningfully at matched training targets.
    The 64-trained recurrent model already benefits from additional evaluation context. This supports keeping the faster
    64-byte training setup for the main long-horizon comparison.</p>
    <p>Batch 4 × length 256 for 3,750 updates gave spine KS2 a small initial lead over MLP. The table below adds two independent
    paired seeds. These scores use full 256-byte tilings, unlike the suffix-only table above; compare within each table.
    This completed diagnostic used an alternate MLP initialization and is not a primary-reference win. The queued narrower
    MLP follow-up was canceled when the user clarified that the original reference must remain fixed.</p>'''+table(['Seed','Alternate MLP384, B4 T256 3750 steps','Spine KS2, same protocol','KS2 − alternate MLP'],context_repeats))

    longrows=[];longgaps=[]
    for pool in ('original','larger'):
        for name in ('original_mlp','spine46','undamped33','frozen46','history'):
            rs=[r for r in long if r['job'].startswith(pool+'_'+name+'_')]
            if not rs:continue
            vals=[r['full_validation_loss'] for r in rs]
            longrows.append([pool,name,len(rs),f"{rs[0]['parameters']:,}",f'{st.mean(vals):.6f}',f'{st.stdev(vals):.6f}' if len(vals)>1 else '—',f"{st.mean(r['sampled_training_loss'] for r in rs):.6f}",f"{st.mean(r['ms_per_step'] for r in rs):.2f}"])
        for name in ('spine46','undamped33','frozen46','history'):
            for control in ('original_mlp',):
                pairs=[]
                for seed in (11,22,33):
                    k=lookup('wave8_long',f'{pool}_{name}_{seed}');m=lookup('wave8_long',f'{pool}_{control}_{seed}')
                    if k and m:pairs.append(k['full_validation_loss']-m['full_validation_loss'])
                if pairs:longgaps.append([pool,name,control,len(pairs),f'{st.mean(pairs):+.6f}',f'{st.stdev(pairs):.6f}' if len(pairs)>1 else '—',', '.join(f'{v:+.6f}' for v in pairs)])
    checkpoint_rows=[]
    for pool in ('original','larger'):
        for name in ('spine46','undamped33','frozen46','history'):
            for step in (15000,50000,100000):
                pairs=[]
                for seed in (11,22,33):
                    k=lookup('wave8_long',f'{pool}_{name}_{seed}');m=lookup('wave8_long',f'{pool}_original_mlp_{seed}')
                    if not k or not m:continue
                    def at(r):
                        if step==100000:return r['full_validation_loss']
                        return next((v['loss'] for v in r['intermediate_evaluations'] if v['evaluation_step']==step),None)
                    kl,ml=at(k),at(m)
                    if kl is not None and ml is not None:pairs.append((kl,ml))
                if pairs:
                    checkpoint_rows.append([pool,name,f'{step:,}',len(pairs),f'{st.mean(x for x,y in pairs):.6f}',f'{st.mean(y for x,y in pairs):.6f}',f'{st.mean(x-y for x,y in pairs):+.6f}'])
    section('Long training and a larger data pool',table(['Training pool','Model','Seeds','Parameters','Mean validation','Seed SD','Mean sampled train','Mean ms/step'],longrows)+table(['Pool','Candidate','Control','Pairs','Mean paired gap','Gap SD','Per-seed gaps'],longgaps)+'''
    <p>All runs in this section train for 100,000 optimizer steps from fresh initialization, with the same full-horizon cosine
    schedule and 25.6 million training byte targets. The larger pool changes the sampling source, not the number of updates
    or target visits. Each model on a given pool uses paired training positions for its seed.</p>
    <p>At this target count the original pool is visited repeatedly, whereas the larger pool has more unique available data.
    The ratio of training target visits to available bytes is about 5.73 for the original pool and 0.57 for the larger pool;
    these are sampling ratios, not deterministic epochs or measured unique-byte coverage.
    The larger pool also adds different text, so this intervention does not isolate repetition from data diversity or composition.
    A train/validation gap can suggest overfitting, but sampled training loss and full-shard validation are different estimators.
    Intermediate full-shard scores at 15k and 50k are stored with every completed long run. These are fixed-budget endpoints; the experiment does not establish that every model has converged.</p>
    <p>The following checkpoint comparisons use only completed, paired long runs. All checkpoints belong to the same 100k
    schedule; their 15k scores must not be substituted for the separately annealed 15k screens. A changing gap can identify
    a transient lead without attributing it prematurely to overfitting or convergence.</p>
    <p>For seed 11 on the original pool, the separately annealed 15k frozen-spine run beat the original MLP, but the frozen
    model is already behind at the 15k checkpoint of the 100k schedule. Thus the difference between short and long rankings
    cannot be explained solely as the MLP overtaking KS2 after 15k steps. The changed learning-rate trajectory matters too.
    The long runs test a complete training regime; they do not isolate horizon from schedule.</p>
    <p>The completed three-seed original-pool means are 1.509071 for the original MLP, 1.509901 for frozen spine and
    1.512073 for learned spine. Frozen spine wins two seeds but loses the mean slightly. On the larger pool, learned spine
    averages 1.465473 versus 1.465737 for MLP: a difference of only −0.000265, with one seed win and two losses.
    This is practical near-equality, not evidence of a reliable KS2 lead. The reserved interval below provides a separate
    check on the ordering.</p>'''+table(['Pool','KS2 candidate','Checkpoint','Pairs','KS2 mean','Original MLP mean','KS2 − MLP'],checkpoint_rows)+fig('long_learning_curves','100k learning curves, same sampled validation metric within each panel.')+fig('long_final_comparison','Final 100k full-shard scores by training pool; individual seeds, means and seed SD.'))

    holdrows=[];short_holdrows=[];holdpairs=[];short_holdpairs=[]
    hp=a.root/'final_checks/holdout_results.json'
    if hp.exists():
        h=json.loads(hp.read_text())
        for row in sorted(h['rows'],key=lambda x:x['job']):
            if row.get('phase','long').startswith('extended_'):continue
            if row.get('phase')=='short':
                source=json.loads((a.root/row['run']/'completed.json').read_text())
                short_holdrows.append([esc(row['job']),f"{source['tiled_evaluation']['loss']:.6f}",f"{row['evaluation']['loss']:.6f}",row['evaluation']['targets']])
                continue
            r=lookup('wave8_long',row['job']);ev=row['evaluation']
            holdrows.append([esc(row['job']),f"{r['full_validation_loss']:.6f}",f"{ev['loss']:.6f}",ev['targets']])
        measured={(row.get('phase','long'),row['job']):row['evaluation']['loss'] for row in h['rows']}
        for pool in ('original','larger'):
            for model in ('spine46','undamped33','frozen46','history'):
                paired=[(measured[('long',f'{pool}_{model}_{s}')],measured[('long',f'{pool}_original_mlp_{s}')])
                        for s in (11,22,33) if ('long',f'{pool}_{model}_{s}') in measured and ('long',f'{pool}_original_mlp_{s}') in measured]
                if paired:
                    gaps=[x-y for x,y in paired]
                    holdpairs.append([pool,model,len(paired),f'{st.mean(x for x,y in paired):.6f}',f'{st.mean(y for x,y in paired):.6f}',f'{st.mean(gaps):+.6f}',sum(v<0 for v in gaps)])
        for model in ('learned_spine','frozen_spine','spine_geo01','spine_geo001','spine_warps_only','spine_coefficients_only'):
            paired=[(measured[('short',f'{model}_{s}')],measured[('short',f'original_mlp_{s}')])
                    for s in (11,22,33) if ('short',f'{model}_{s}') in measured and ('short',f'original_mlp_{s}') in measured]
            if paired:
                gaps=[x-y for x,y in paired]
                short_holdpairs.append([model,len(paired),f'{st.mean(x for x,y in paired):.6f}',f'{st.mean(y for x,y in paired):.6f}',f'{st.mean(gaps):+.6f}',sum(v<0 for v in gaps)])
    holdout_meta=json.loads((a.root/'reserved_holdout_extension2.json').read_text()) if (a.root/'reserved_holdout_extension2.json').exists() else json.loads((a.root/'reserved_holdout.json').read_text())
    holdout_description=f'<p>The final reserved interval is source validation token positions {holdout_meta["source_token_start"]:,} through {holdout_meta["source_token_start"]+holdout_meta["source_token_count"]-1:,}, disjoint from all earlier evaluated intervals. Its SHA-256 is <code>{holdout_meta["sha256"]}</code>.</p>'
    section('Reserved validation interval',table(['Training pool','100k KS2 configuration','Paired seeds','KS2 mean','Original MLP mean','KS2 − MLP','KS2 lower-loss seeds'],holdpairs)+table(['Run','Exploration shard','Reserved shard','Reserved target visits'],holdrows)+holdout_description+'''
    <p>
    Absolute loss can differ because text difficulty differs. Compare models within each shard. These results check whether
    conclusions survive a new interval; they do not demonstrate transfer to another corpus or model scale.</p>
    <h3>Short-run KS2 geometry-learning follow-up</h3><p>These checkpoints all use 15,000 steps. The MLP rows reuse the
    original unmodified reference checkpoints. They are separate from the 100k comparisons above.</p>'''+table(['15k KS2 configuration','Paired seeds','KS2 mean','Original MLP mean','KS2 − MLP','KS2 lower-loss seeds'],short_holdpairs)+table(['15k run','Exploration shard','Reserved shard','Target visits'],short_holdrows))

    speed=[]
    sp=a.root/'final_checks/speed_results.json'
    if sp.exists():
        for r in json.loads(sp.read_text())['summary']:
            speed.append([esc(r['model']),f"{r['median_ms']:.3f}",f"{r['min_ms']:.3f}",f"{r['max_ms']:.3f}"])
    pruning=''
    if (a.root/'history_final_prune_cpu_check.json').exists():
        pruning=f'''<h3>Prune the final unused side output</h3><p>The final history decoder's 2048D side update has no
        consumer. An optional KS2 implementation keeps every decoder input and all geometry, but projects only the final
        96D main update. On the trained 100k history checkpoint, CPU logits are identical and the maximum surviving-gradient
        absolute difference is 2.98×10⁻⁸. This removes 2,048 unused learned gains and 4,194,304 fixed-buffer bytes:
        565,504 learned parameters and 33,295,360 buffer bytes remain. It also removes 1,048,576 forward projection MACs per
        target. These are functional-equivalence checks and an implementation optimization, not a new loss-improvement claim.
        The archived training runs and primary baseline remain unchanged. See <a href="history_final_prune_cpu_check.json">
        the trained CPU check</a> and {artifact_link('final_checks/history_prune_check.json','GPU check')}.</p>'''
        if sp.exists():
            measured={r['model']:r['median_ms'] for r in json.loads(sp.read_text())['summary']}
            if 'history_pruned_fused' in measured:
                before=measured['history_fused'];after=measured['history_pruned_fused']
                pruning+=f'<p>Same-GPU median step time is {before:.3f} ms before pruning and {after:.3f} ms after pruning ({100*(before-after)/before:.1f}% lower). The five raw measurements per case are retained; this short benchmark is distinct from sustained training time.</p>'
    telemetry=''
    if (a.root/'telemetry_summary.json').exists():
        tm=json.loads((a.root/'telemetry_summary.json').read_text())
        telemetry=table(['Physical GPU','Mean reported busy %','Samples ≥90% busy','Median busy clock MHz'],[
            [r['gpu'],f"{r['mean_reported_use_percent']:.1f}",f"{100*r['fraction_samples_at_least_90_percent_busy']:.1f}%",r['busy_clock_median_mhz']] for r in tm['cards']])
        telemetry+=f'''<p>Telemetry covers {tm['samples']} point samples from {esc(tm['first_sample_utc'])} through
        {esc(tm['last_sample_utc'])}, approximately every 30 seconds. All four cards report at least 90% activity in
        {100*tm['all_four_at_least_90_percent_fraction']:.1f}% of those samples. Monitoring began after the study started;
        these are sampled activity readings, not a whole-study utilization integral or a measurement of peak FLOPs.
        Initialization, compilation and scoring transitions are included. See <a href="telemetry_summary.json">the summary</a>
        and <a href="gpu_telemetry.jsonl">raw observations</a>.</p>'''
        if tm.get('windows'):
            telemetry+='<p>The initial study and the explicitly requested two-hour combination extension are separate observation windows. The intervening gap and post-training samples from the first window are excluded; the first-to-last timestamps do not imply continuous work throughout that span.</p>'+table(['Observation window','Samples','All four GPUs ≥90% busy'],[[esc(w['name']),w['samples'],f"{100*w['all_four_at_least_90_percent_fraction']:.1f}%" if w['all_four_at_least_90_percent_fraction'] is not None else '—'] for w in tm['windows']])
    section('Speed, numerical validation and memory',table(['Model','Median ms/update','Minimum','Maximum'],speed)+telemetry+'''
    <p>The final timing test interleaves the recorded reference and KS2 configurations on the same physical GPU in five randomized-order rounds.
    Each fresh process measures 100 captured training updates after warmup/reset, including batch copies and learning-rate
    writes. It excludes compilation and evaluation. This controls GPU assignment better than comparing independent runs on
    cards with different clocks. Sustained whole-training times remain available in the run ledger.</p>
    <p>Times drift downward across the five rounds for every configuration. Other GPUs concurrently score held-out models
    during part of this window; the cause of the drift is not isolated. Randomized within-round ordering and the reported
    ranges matter more than millisecond precision. <a href="final_timing_round_comparisons.json">Same-round timing ratios</a>
    preserve the comparison: uniform rank-1 KS2 is about 9% slower than MLP, and the mixed MLP-branch prototype about 21% slower.
    These short captured-step measurements do not guarantee the same sustained throughput under another workload.</p>
    <p>Full geometry fusion reduces an isolated 64-chart forward/backward benchmark from about 1.430 ms to 0.418 ms.
    Undamped geometry falls from 0.685 to 0.168 ms, and spine-only from 0.270 to 0.073 ms. These are isolated kernels;
    total model speedups are smaller because attention, projections, optimizer and memory operations remain.</p>
    <p>Validation covers random and extreme coordinates, masked tail sizes, zero initialization, all geometry gradients,
    captured execution and trained model parameters. At shared trained parameters, full-model maximum logit error was
    1.57×10⁻⁵ and maximum parameter-gradient relative L2 error 3.26×10⁻⁶. Partial kernels have similarly small errors.
    They are numerically close, not bit-identical. Full training trajectories can diverge by several thousandths of loss.</p>
    <p>One initial partial-kernel benchmark crashed because the checker reused a retained autograd graph across streams;
    using fresh copied cells fixed the harness. An earlier tanh approximation was replaced with the precise device tanh.
    The original full and partial kernels passed their checks; generalized-rank kernels are reported separately. These engineering failures and corrections are retained in the logs, not hidden as training failures.</p>
    <p>Learned parameter count omits material storage. The history baseline has 2,270,208 bytes of FP32 learned weights but
    37,489,664 bytes of fixed buffers for its stored projection codes. The original MLP has 2,463,744 weight bytes and no
    such buffers; canonical dense KS2 has 2,452,672 weight bytes and 1,488 buffer bytes. These figures describe the original dense-matrix code representation.
    The extended study below implements and evaluates compact blockwise Hadamard codes separately. Optimizer and activation memory are
    additional; per-run peak allocation is in the CSV/JSON.</p>
    <p>There is a compute distinction as well. Summing input/output matrix sizes across four FFNs gives 9,371,648 projection
    multiply-accumulates per target for the history model, 190,464 for dense 31-chart KS2, and 294,912 for the original MLP.
    These counts exclude geometry, attention, normalization and backward. The fixed-code history implementation therefore
    performs about 49 times the projection work of dense 31-chart KS2 despite having fewer learned parameters. Its muxes
    are dense matrix operations, not implicit fast Hadamard transforms. Parameter compression alone does not deliver the
    corresponding compute or storage saving in this implementation.</p>
    <p>Frozen geometry saves learned parameters and optimizer state, but retains its fixed geometry values and computation.
    Parameter-saving claims for frozen variants should therefore be read alongside storage and timing, not as free removal.</p>'''+pruning)

    section('How this compares with the earlier experiments','''
    <p>The earlier three-seed 15k MLP reference averaged 1.622722; the selected demuxed-history model averaged 1.666841.
    Fresh controls reproduce that broad gap. Increasing the decoder-only side residual to 2048 dimensions and increasing
    charts from 32 to 64 helped relative to smaller side streams, but did not by itself beat the MLP.</p>
    <p>The single-mux architecture changed the information path: KS2 and RWKV branches ran in parallel, so KS2 no longer
    received the same-layer post-attention state. Static input/output gains only modestly helped. An extra KS2 modulation
    subspace, whose weighted output vectors were summed then RMS-normalized, improved that model to about 1.6758,
    still behind the demuxed-history reference at about 1.6669. Raw-history and attention-pass-through variants did not
    resolve the gap. The current evidence favors retaining useful routing while testing decoder changes separately.</p>
    <p>Replacing every RWKV linear projection with fixed Hadamard muxes reduced learned parameters from 567,552 to 306,944
    but raised loss from 1.665535 to 1.767811. Individual read/key/value/output replacements each hurt. Gate-up was the
    promising exception: 543,360 parameters and 1.665681. The new gate-budget experiments test that exception in a different
    decoder rather than assuming it is universally free.</p>
    <p>The requested RWKV-8 comparison was performed using a local, similarly sized adaptation of BlinkDL's RWKV7+ROSA
    parallel hybrid, pinned at commit <code>20be0f8cf0bdf506d8c0030c87a9a3416f95b143</code>.
    The parameter-matched adaptation scored 1.673763 at 567,296 parameters and 19.73 ms/step, versus baseline 1.666923
    at 567,552 parameters and 15.64 ms/step. Keeping the original RWKV controller sizes plus ROSA scored 1.671107 at
    619,520 parameters and 20.14 ms/step. Reducing controllers without ROSA scored 1.675211, so ROSA partially recovered
    that reduced model's loss but did not beat the original baseline.</p>
    <p>This RWKV-8 result is specifically a 64-byte-context local adaptation. The official demonstration lacked a ROSA training
    backward; this implementation supplies exact single-bit counterfactual credits with a Bernoulli surrogate. Forward
    suffix matching and GPU credits were checked against references. It is not an official full-scale RWKV-8 training recipe,
    and it does not test long-context ROSA benefits. See <a href="../../rwkv8_summary.json">the full RWKV-8 summary</a> and
    <a href="https://github.com/BlinkDL/RWKV-LM/tree/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v8">the pinned upstream source</a>.</p>
    <p>Earlier 105k all-MLP runs averaged 1.540879; adding a tiny KS2 branch to the inner two MLP FFNs averaged 1.540212,
    a very small difference under that protocol. This is evidence of compatibility, not a demonstrated general lead.
    Earlier normal-attention nanoGPT tests also found near-equality (7.819336 MLP versus 7.823028 KS2), but used a full
    GPT-2 vocabulary, only 500 CPU updates and a different scale. Their token losses are not comparable to byte losses here.</p>
    <p>The repository's controlled patch-denoising study does contain a substantial KS2 lead: the reference 433,452-parameter
    KS2 achieved 27.388 dB mean Set12 PSNR versus 26.303 dB for the 435,030-parameter MLP, with three seeds and 3,000 main
    updates. That task uses normalized image patches, squared-error loss, learned projections and two geometry iterations;
    its MLP uses tanh. This is meaningful evidence for KS2 on that task, not evidence that the same cell must beat a ReLU²
    residual language-model FFN. The compact stabilized denoising and quantized exports show additional storage/speed
    opportunities, but use different sizes and training procedures. See <a href="../../../denoising/README.md">the recorded
    denoising protocol and results</a>.</p>''')

    section('Hypotheses tested',table(['Hypothesis','Evidence from this session','Interpretation'],[
        ['KS2 geometry is effectively inactive','Geometry changes features and inference-time removal hurts loss.','Rejected for the probed trained models; activity alone does not prove a training advantage.'],
        ['Fixed decoder routing is the main restriction','Adaptive projection/readout configurations substantially improve KS2.','Supported as a useful intervention, with routing, side/history and parameter-count changes explicitly separated in the ablations.'],
        ['An unsuitable output feature transform limits the cell','Square and ReLU² outperform ReLU, SiLU and the plain cell.','Quadratic output features are a strong practical direction at this scale.'],
        ['All geometry terms are needed','Weak damping; undamped and spine variants retain competitive loss.','Parameter allocation and kernel cost warrant simplifying geometry, while retaining replicated tests.'],
        ['The missing second geometry iteration explains the LLM gap','Two/four shared updates and cross-chart shuffles lack a consistent replicated gain.','Not supported by these short screens; additional iterations increase compute.'],
        ['Longer training windows explain the apparent context gain','Same-target evaluation shows most gain comes from longer inference context.','Do not credit extra training context for an evaluation-context change.'],
        ['Spine warps saturate and stop contributing','Seed-11 long checkpoints show little saturation and substantial feature changes.','Not supported for the measured spine warps; this does not rule out other conditioning problems.'],
        ['Learning all geometry parameters is automatically better than freezing','Frozen spine improves 15k short runs; the long regime changes that ranking.','Geometry learning is protocol-dependent. Selective training and slower geometry learning are tested directly.'],
        ['Larger charts are always better use of the geometry budget','Ranks 1, 2 and 4 beat the fixed MLP in all three 15k seeds; rank 1 and 4 eager/fused repeats agree on the direction.','Smaller spine charts expose more distinct scalar means at the same projected feature width. Long replication is needed; this is not evidence against learned mixing across model dimensions.'],
        ['Correlated polynomial bases explain the whole loss gap','Power coordinates improve sampled conditioning and three-seed spine loss, but full-geometry power models remain behind MLP.','Optimization coordinates matter in this harness, but better design conditioning is not a universal quality fix.'],
        ['Restoring the RHAC local equation and sigmoid inputs resolves the mismatch','All four three-seed local-equation/input interventions remain behind the unchanged MLP at 15k.','Not supported here. RHAC also preserves pretrained gated projections and uses a different history pathway; these tests do not reproduce that whole setting.'],
        ['An implicit Hadamard representation automatically speeds training','Fixed buffers shrink about 12-fold, but the measured compact-code whole step takes 2.45 times as long.','A storage saving is established; a speed saving is not. Dense projections remain faster for the tested workload.'],
        ['Attention is the remaining loss bottleneck','Trained component timings show attention is the largest measured block cost once KS2 is simplified.','This establishes a compute target, not that attention capacity limits achievable loss. Earlier attention upgrades did not establish a quality gain.'],
        ['Small training-pool reuse obscures the ordering','A tenfold larger pool improves long-run validation; compare paired gaps in the long-run table.','Use the complete replicated comparison, not the most favorable single seed.'],
    ]))

    section('What the evidence supports next','''
    <p><strong>Keep the original MLP reference fixed and spend further search on KS2.</strong> Match total parameters,
    training targets, optimizer updates and data when evaluating a KS2 improvement. Preserve the original MLP initialization,
    width and optimizer rather than moving the comparison target during the search.</p>
    <p><strong>Retain adaptive routing and quadratic features in further KS2 tests.</strong> Those variants deliver larger observed
    improvements than static mux gains or wider attention in this study, although the routing comparison also changes the side/history path. Fixed multiplexing can still be useful for
    parameter/storage tradeoffs, but equal exposure of every coordinate is not the same as learned task-specific routing.</p>
    <p><strong>Separate geometry expressivity from budget allocation.</strong> Removing damping reduces parameter and compute cost, but the replicated 100k undamped models
    underperform the unchanged MLP on both pools. The early single-seed result does not justify making that the default. Compare learned, slowed and selectively frozen
    geometry within KS2 while preserving its projection width and initialization. Judge geometry ablations after retraining
    as well as by inference-time lesion tests, and retain the original MLP as the external benchmark.</p>
    <p><strong>Isolate the number of spine inputs from warp parameter count.</strong> A targeted next experiment would
    compare the existing rank-8 spine with 368 scalar spines whose warp parameters are shared in groups of eight.
    That would preserve 368 feature coordinates, six coefficients per feature and the rank-8 warp budget, while changing
    whether outputs evaluate one shared mean or separate scalar inputs. A separate scale factor could then test the
    remaining input-scale confound. This experiment is proposed, not performed in this session.</p>
    <p><strong>Distinguish scalar spine improvements from full geometric interactions.</strong> The rank-1 and mixed-rank
    follow-ups retain the spine-only simplification. Their results do not establish that mixed-rank edge/damping geometry
    is ineffective. A future full-cell mixture would need its own matched-budget comparison, because higher ranks spend
    substantially more parameters on edge interactions. The present results support a useful KS2-derived FFN candidate;
    they do not validate every part of the original multiplexed architecture.</p>
    <p><strong>Carry the verified kernels forward, with trajectory replication.</strong> Fusion makes more comparisons feasible.
    Preserve the eager reference and all-gradient checks. Small single-seed loss leads are not convincing when repeats and
    floating-point paths move by similar amounts.</p>
    <p>The remaining uncertainty is about scaling and task distribution: larger models, more distinct data, longer context,
    alternative KS2 rank/chart allocation and task-adapted coordinate statistics could change the ordering. This session
    tests several of those pressures at small scale; it does not establish that KS2 must win, nor that it cannot.</p>''')

    section('Completed alternate-control diagnostics',table(['Diagnostic model','Seeds','Mean loss','Seed SD','Individual scores'],[short_groups[1],short_groups[2]])+'''
    <p>These alternate MLP results were collected before the user's correction. They remain in the record to make the session
    auditable, but they are not the requested optimization target or the primary reference. The input-scale, small-budget and
    context diagnostic tables above also include alternate MLP configurations. Their comparisons are explicitly local to those
    tests; they do not replace the original MLP benchmark. No additional MLP variant search is planned.</p>'''+fig('short_run_frontier','Exploratory screens including alternate controls, retained for audit; not the primary benchmark.'))
    ledger=[]
    for r in rows:
        timing='suspended; see controlled timing' if r.get('suspended_for_mixed_screen') else f"{r['ms_per_step']:.2f}"
        ledger.append([esc(r['wave']),esc(r['job']),r['seed'],r['steps'],f"{r['batch_size']}×{r['sequence_length']}",f"{r['parameters']:,}",f"{r['full_validation_loss']:.6f}",timing,f'<a href="{esc(r["source"])}">JSON</a>'])
    section('Complete experiment ledger', '''<p>Each result links to its configuration, history, source hashes, data hashes, timing,
    evaluation and checkpoint provenance. The parent run directory contains the exact command and source snapshot. All listed
    rows completed their requested step count and recorded zero nonfinite training steps. Training-worker failures, if any, appear
    separately in <code>all_runs.json</code>. The selected baseline file was not silently replaced by an exploratory candidate.</p>'''+table(['Wave','Job','Seed','Steps','Batch × length','Parameters','Loss ↓','ms/step','Record'],ledger))

    section('Reproduction and artifacts','''
    <p>Use <code>/home/curved/int8-vllm/.venv/bin/python</code> with the recorded ROCm environment. Wave manifests specify
    every configuration and seed. <code>study_checked_v6.py</code> validates model structure, gradients and causality before
    dispatch; versioned <code>study_wave</code> scripts run up to four independent workers. The first long runs use v8 and selective-geometry follow-ups use v9. Extended conditioning screens use v10,
    calibration screens use v12, and generalized-rank freezing uses v14. The original-equation tests and selected extended long runs use v15. The final geometry-input-scale follow-up uses v17, preserving v15 behavior when its optional scale is absent. Extended evaluations use v16/v17, including the documented batch-size handling for the original signed equation.
    The combination screens and long replications use v18, which installs bias/gain calibration after rank changes and freezing so the new calibration parameters remain trainable. The user-requested mixed-rank and MLP-branch follow-up uses v19, which delegates all earlier configurations unchanged to v18. Final reserved scoring uses the v19 evaluator and the fresh interval specified in <code>final_holdout_plan.json</code>. Earlier waves retain their versioned factories.
    The exact source used by each run is copied into its <code>sources/</code> directory; using today's top-level files alone
    is not a substitute for checking those snapshots.</p>
    <pre>python study_collect.py --root gpu_runs/goal_20261022_232213
python study_plot.py --root gpu_runs/goal_20261022_232213
python study_report_figures.py --root gpu_runs/goal_20261022_232213
python study_extended_figures.py --root gpu_runs/goal_20261022_232213
python study_report.py --root gpu_runs/goal_20261022_232213</pre>
    <p>The actual interpreter and repository-relative paths are in each run's <code>command.json</code>.
    <code>study_final_checks_v19.py</code> specifies the final reserved holdout and randomized same-GPU speed protocol.
    <code>gpu_telemetry.jsonl</code> records utilization and clocks. <code>prior_snapshots/</code> and
    <code>prior_additional/</code> preserve historical summaries used for comparison. Figures are standalone SVG and PNG.
    <a href="original_reference_check.json">original_reference_check.json</a> verifies that the restored primary MLP has
    exactly the original initialized tensors and optimizer hyperparameters for all three seeds.
    <a href="control_correction.json">control_correction.json</a> records canceled alternate-control work.</p>
    <p><a href="source_bundle.tar.gz">Download source bundle</a> · <a href="source_bundle_check.json">Bundle verification and coverage</a>.
    The archive indexes the exact source versions for each completed run by SHA256, alongside separately labeled current
    analysis tools and manifests. It excludes datasets and trained weights. Regenerate with <code>study_source_bundle.py --root gpu_runs/goal_20261022_232213</code> after the last run so its coverage includes the complete study.</p>''')
    if (a.root/'goal_completion_review.json').exists():
        title,ident,body=parts[-1]
        parts[-1]=(title,ident,body+'<p><a href="goal_completion_review.json">Goal completion review</a> records the requirements, evidence and any checks still pending. Artifact generation alone does not establish that all research and reporting requirements are finished.</p>')

    from study_extended_report import add_sections
    extended_summary=add_sections(a.root,rows,section,table,esc,lookup,fig)
    from study_combination_report import add_sections as add_combination_sections
    combination_summary=add_combination_sections(a.root,rows,section,table,esc,lookup,fig)
    extended_summary['combinations']=combination_summary
    from study_mixed_report import add_sections as add_mixed_sections
    extended_summary['mixed_ranks']=add_mixed_sections(a.root,rows,section,table,esc,lookup)
    combination_overview=''
    short_combos=combination_summary.get('short_groups',[])
    if len(short_combos)==12 and all(g['n']==3 for g in short_combos):
        reference=next(g for g in short_combos if g['model']=='combo_rank1_reference')
        improvements=sum(g['mean_loss']<reference['mean_loss'] for g in short_combos if g['model']!='combo_rank1_reference')
        combination_overview=f'<p><strong>Combining improvements:</strong> {improvements}/11 combinations beat the fully learned rank-1 KS2 mean in the three-seed 15k screen. Frozen rank-1 plus a learned bias improves its frozen parent, but does not beat the fully learned rank-1 reference. Longer-run comparisons below test whether this ordering persists.</p>'
    combo_long=combination_summary.get('long_groups',[])
    power_original=next((g for g in combo_long if g['pool']=='original' and g['model']=='combo_rank1_powers' and g['n']==3),None)
    if power_original:
        g=power_original
        combination_overview+=f'<p><strong>Original-pool long-run combination:</strong> rank-1 plus the power basis averages {g["mean"]:.6f} at 100k, a gap of {g["mlp_gap"]:+.6f} versus the unchanged MLP ({g["mlp_wins"]}/3 seed wins), and {g["rank1_gap"]:+.6f} versus the original-basis rank-1 KS2 ({g["rank1_wins"]}/3 wins). It retains 73,600 learned parameters per FFN. This is an exploratory comparison on the original pool; the larger-pool and fresh reserved results must be considered separately.</p>'
    if len(combo_long)==8 and all(g['n']==3 for g in combo_long):
        descriptions=[]
        for pool in ('original','larger'):
            poolgroups=[g for g in combo_long if g['pool']==pool]
            mlpwins=sum(g['mlp_gap']<0 for g in poolgroups)
            rankwins=sum(g['rank1_gap']<0 for g in poolgroups)
            descriptions.append(f'{pool.capitalize()} pool: {mlpwins}/4 combination means beat MLP and {rankwins}/4 beat the existing rank-1 KS2 mean.')
        combination_overview+='<p><strong>Completed 100k combinations:</strong> '+' '.join(descriptions)+' These are exploratory mean comparisons, not independent significance tests; individual paired-seed results and the fresh reserved check are reported below.</p>'
    combo_held=combination_summary.get('reserved_long_groups',[])
    if len(combo_held)==8 and all(g['n']==3 for g in combo_held):
        descriptions=[]
        for pool in ('original','larger'):
            poolgroups=[g for g in combo_held if g['pool']==pool]
            descriptions.append(f'{pool.capitalize()} pool: {sum(g["mlp_gap"]<0 for g in poolgroups)}/4 combination means beat MLP, and {sum(g["rank1_gap"]<0 for g in poolgroups)}/4 beat the existing rank-1 KS2 mean.')
        combination_overview+='<p><strong>Fresh reserved confirmation:</strong> '+' '.join(descriptions)+' No further training or model selection uses this interval.</p>'
    mixed=extended_summary.get('mixed_ranks',{}).get('short_groups',[])
    if len(mixed)==3 and all(g['n']==3 for g in mixed):
        hybrid=next(g for g in mixed if g['model']=='mixed_balanced_mlp32')
        parent_wins=sum(v['parent_gap']<0 for v in hybrid['individual'])
        combination_overview+=f'<p><strong>User-requested mixed ranks:</strong> all three mixtures beat the unchanged MLP mean at 15k, but none beats the uniform rank-1 KS2 mean. Replacing 32 rank-1 features in the balanced mixture with an MLP branch changes mean loss by {hybrid["parent_gap"]:+.6f}, with {parent_wins}/3 seed wins. This small difference does not establish a dependable incremental gain; the reserved comparison is reported separately. Every mixed FFN includes two rank-16 charts and stays below the MLP FFN parameter budget.</p>'
    if combination_overview:
        title,ident,body=parts[0]
        assert title=='Findings at a glance'
        parts[0]=(title,ident,body+combination_overview)
    from study_final_overview import final_overview
    overview = final_overview(a.root, rows, table)
    if overview is not None:
        title, ident, _ = parts[0]
        parts[0] = (title, ident, overview)
        for index, (title, ident, body) in enumerate(parts):
            if title == 'What the evidence supports next':
                parts[index] = (title, ident, '''<p><strong>Carry forward both rank-1 basis choices as KS2 candidates.</strong>
                The power-basis combination beats the fixed MLP on all three fresh reserved seeds for both training pools,
                while the original Hermite-basis rank-1 variant retains the stronger larger-pool mean. The power basis improves
                the original-pool parent comparison but does not provide a universal additive improvement. Keep both results
                and their training-pool dependence visible. The mixed-rank MLP branch does not show a dependable incremental
                benefit: its tiny exploratory improvement reverses on fresh reserved text. Further work on that branch needs
                a specific mechanistic hypothesis, rather than treating this result as an established win.</p>''' + body)

    order=[
        'Findings at a glance','Protocol and interpretation','Two-hour extension to combine improvements','Do the short-run improvements accumulate?','Power basis and geometry learning-rate interaction','Variation in the unchanged KS2 reference','Selection of long combination tests','Do combined gains survive 100k updates?','Do combinations improve on the reserved interval?','Architectures and the route information takes',
        'FFN-only parameter comparison','Mixed KS2 ranks and an MLP branch','Mixed-rank results','What the mixed-rank experiment suggests','Mixed ranks on the reserved interval','Scheduling the late mixed-rank request','Where the parameter budget goes','Replicated short-run comparisons','Long training and a larger data pool',
        'Selection of the final long experiments','Extended long-run geometry tests','Training fit and validation improvement','Extended learning curves at matching horizons','KS2 basis conditioning and chart rank','Why chart rank changes the spine pathway','Chart rank also changes basis scale and conditioning','Scaling the spine input without scaling the bypass','How much learned geometry adaptation is needed?','Interaction between chart rank and freezing','Original KS2 equation and bounded geometry inputs','Numerical sensitivity of the original signed-coordinate equation',
        'Reserved validation interval','Extended study on the reserved validation interval','Speed, numerical validation and memory','Compact implementation of the original Hadamard codes','Generalized geometry kernels and frozen-parameter savings','Where the compute is spent inside trained blocks','Compute limits after simplifying KS2',
        'Routing, activation and optimizer interventions','Is geometry inactive or wasted?',
        'What learned geometry does over training','Geometry offsets, input dependence and basis conditioning','Equivalent-basis conditioning probe','Replicated dependence on spine and edge terms','Does geometry dominate gradient clipping?','Which geometry parameters should learn?',
        'Repeated geometry with shared parameters','Cross-subspace geometry refinement','Feature-spectrum diagnostic',
        'Hypotheses tested','How this compares with the earlier experiments','Relation to RHAC history conditioning','What the evidence supports next',
        'Completed alternate-control diagnostics','Small decoder budgets',
        'Input-scale control with identical random directions','Standard attention and RWKV projection allocation',
        'Context length and optimizer updates','Complete experiment ledger','Reproduction and artifacts',
    ]
    priority={title:i for i,title in enumerate(order)}
    parts.sort(key=lambda item:priority.get(item[0],len(order)))
    nav=''.join(f'<a href="#{ident}">{esc(title)}</a>' for title,ident,_ in parts)
    body=''.join(f'<section id="{ident}"><h2>{esc(title)}</h2>{body}</section>' for title,ident,body in parts)
    css='''body{font:17px/1.6 system-ui,sans-serif;color:#152536;background:#f6f8fa;margin:0}main{max-width:1150px;margin:auto;background:white;padding:45px}h1{font-size:36px;line-height:1.2}h2{margin-top:46px;border-bottom:2px solid #dbe5ed;padding-bottom:8px}h3{font-size:20px}p{max-width:100ch}a{color:#145e9d}nav{display:flex;flex-wrap:wrap;gap:8px 20px;padding:20px;background:#edf3f8}nav a{font-size:14px}table{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px;border-bottom:1px solid #dce4eb}th{background:#e9f0f6;position:sticky;top:0}tr:nth-child(even){background:#f6f8fa}.scroll{overflow:auto;max-height:750px;margin:20px 0}pre{white-space:pre-wrap;background:#edf3f8;padding:18px;font-size:13px}code{font-size:.86em;overflow-wrap:anywhere}img{max-width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:14px;color:#456}small{color:#567}@media print{body{background:white}main{padding:0;max-width:none}.scroll{max-height:none;overflow:visible}nav{display:none}h2{break-before:auto}tr{break-inside:avoid}}'''
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    doc=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KS2 language-model study</title><style>{css}</style></head><body><main><h1>KS2 in a small language model: bottlenecks, controls and efficiency</h1><p>{esc(status)} · Generated {stamp}</p><nav>{nav}</nav>{body}</main></body></html>'
    (a.root/'report.html').write_text(doc)
    (a.root/'report_summary.json').write_text(json.dumps(dict(generated_utc=stamp,status=status,completed_runs=len(rows),long_runs=len(long),extended=extended_summary,short_cohorts=short_groups,long_summary=longrows,long_paired_gaps=longgaps,long_checkpoint_pairs=checkpoint_rows,geometry_learning_groups=geometry_learning_groups,budget_pairs=budget,input_scale_pairs=scale_rows,context_replications=context_repeats,holdout_long_pairs=holdpairs,holdout_short_pairs=short_holdpairs),indent=2)+'\n')
    print(a.root/'report.html')


if __name__=='__main__':main()
