"""One joint size-speed cost axis versus loss, across the collected studies."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from study_collect import collect


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def canonical(row):
    cfg=dict(row['config']);cfg.pop('name',None);cfg.setdefault('family',row['family'])
    if cfg['family']=='mlp':
        # output_lr has no effect on MLPs, which have no KS2 output-gain group.
        cfg.pop('output_lr',None)
    return json.dumps(cfg,sort_keys=True)


def original_mlp(row):
    return json.loads(canonical(row))=={'family':'mlp'}


def stratum(row):
    return (row['data_train_sha256'],row['data_validation_sha256'],row['steps'],row['batch_size'],row['sequence_length'])


def frontier(points):
    return [a for a in points if not any(b['cost']<=a['cost'] and b['loss']<=a['loss'] and (b['cost']<a['cost'] or b['loss']<a['loss']) for b in points)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;prior=root.parent/json.loads((root/'study_state.json').read_text())['prior_study']
    groups={};excluded=[];allrows=[]
    for study in (prior,root):
        # The previous index carries explicit scheduler-suspension flags.
        rows=json.loads((study/'all_runs.json').read_text())['rows'] if study==prior else collect(study)
        for row in rows:
            row=dict(row,study=study.name)
            if (row['family']=='mlp' and not original_mlp(row)) or (row['family']=='dense_ks2' and row['config'].get('cell') in ('relu2','square')):
                excluded.append(dict(study=study.name,run=row['source'],reason='Altered MLP control; original control remains fixed.'));continue
            if row.get('suspended_for_mixed_screen'):
                excluded.append(dict(study=study.name,run=row['source'],reason='Recorded training time includes scheduler suspension; unsuitable for size-speed cost.'));continue
            raw=json.loads((study/row['source']).read_text())
            row['ffn_parameters']=st.mean(raw['ffn_parameters_by_layer'])
            row['ffn_by_layer']=raw['ffn_parameters_by_layer']
            allrows.append(row)
            # Keep execution studies separate: sources/hardware operating conditions differ.
            key=(stratum(row),study.name,canonical(row))
            groups.setdefault(key,[]).append(row)
    # Primary historical references at the original horizons; current cohort for new horizons.
    references={}
    for row in allrows:
        if not original_mlp(row):continue
        wanted=(row['study']==prior.name and ((row['wave']=='initial' and row['job']=='mlp') or (row['wave']=='wave3' and row['job']=='mlp_seed33') or row['wave']=='wave8_long'))
        if wanted:references.setdefault(stratum(row),[]).append(row)
    for row in allrows:
        if original_mlp(row) and stratum(row) not in references:
            references[stratum(row)]=[v for v in allrows if original_mlp(v) and stratum(v)==stratum(row) and v['study']==row['study']]
    points=[]
    for (key,study,cfg),cohort in groups.items():
        if key not in references:
            excluded.extend(dict(study=r['study'],run=r['source'],reason='No unchanged original MLP at the same data/horizon/batch/context.') for r in cohort);continue
        ref=references[key]
        if json.loads(cfg)['family']=='mlp' and study==prior.name:
            primary={(r['study'],r['source']) for r in ref}
            cohort=[r for r in cohort if (r['study'],r['source']) in primary]
            if not cohort:continue
        ref_params=st.mean(r['ffn_parameters'] for r in ref)
        ref_ms=st.mean(r['ms_per_step'] for r in ref);ref_loss=st.mean(r['full_validation_loss'] for r in ref)
        per_seed=[]
        for seed in sorted({r['seed'] for r in cohort}):
            same=[r for r in cohort if r['seed']==seed]
            per_seed.append(dict(seed=seed,loss=st.mean(r['full_validation_loss'] for r in same),ms=st.mean(r['ms_per_step'] for r in same),parameters=st.mean(r['ffn_parameters'] for r in same)))
        params=st.mean(v['parameters'] for v in per_seed);ms=st.mean(v['ms'] for v in per_seed)
        loss=st.mean(v['loss'] for v in per_seed)
        cfg_obj=json.loads(cfg);names=sorted({r['config']['name'] for r in cohort})
        label=('Original MLP' if study==ref[0]['study'] else 'Original MLP (current repeat)') if cfg_obj['family']=='mlp' else names[0]
        n=len(per_seed)
        group=dict(id=len(points),name=label,aliases=names,study=study,config=cfg_obj,
                   timing_mode='worker',quality_mode='exploratory',derived_pruning=False,
                   steps=key[2],batch=key[3],context=key[4],train_bytes=cohort[0]['data_train_bytes'],train_sha=key[0],validation_sha=key[1],
                   n=n,runs=len(cohort),seeds=[v['seed'] for v in per_seed],parameters=params,ms=ms,loss=loss,
                   loss_sd=st.stdev(v['loss'] for v in per_seed) if n>1 else None,
                   cost=(params/ref_params)*(ms/ref_ms),raw_parameter_ms=params*ms,
                   mlp_parameters=ref_params,mlp_ms=ref_ms,mlp_loss=ref_loss,loss_gap=loss-ref_loss,
                   perplexity_ratio=math.exp(loss-ref_loss),per_seed=per_seed,
                   buffer_mib=st.mean(r['buffer_bytes'] for r in cohort if r.get('buffer_bytes') is not None)/2**20 if any(r.get('buffer_bytes') is not None for r in cohort) else None,
                   references=[dict(study=r['study'],record=r['source']) for r in ref],
                   sources=[dict(study=r['study'],record=r['source']) for r in cohort])
        points.append(group)
    controlled=root/'final_checks/speed_results.json'
    if controlled.exists():
        measured=json.loads(controlled.read_text())
        timing={canonical(dict(config=row['config'],family=row['config']['family'])):row['model'] for row in measured['rows']}
        summaries={row['model']:row for row in measured['summary']}
        mlp_name=timing[json.dumps({'family':'mlp'},sort_keys=True)]
        reference_ms=summaries[mlp_name]['median_ms']
        prune_path=root/'final_checks/pareto_prune_checks.json'
        prunes=json.loads(prune_path.read_text()) if prune_path.exists() else None
        for parent in list(points):
            if parent['batch']!=4 or parent['context']!=64:continue
            key=json.dumps(parent['config'],sort_keys=True)
            if key not in timing:continue
            benchmark=summaries[timing[key]]
            point=dict(parent,id=len(points),timing_mode='controlled',ms=benchmark['median_ms'],mlp_ms=reference_ms,
                       cost=parent['parameters']/parent['mlp_parameters']*benchmark['median_ms']/reference_ms,
                       raw_parameter_ms=parent['parameters']*benchmark['median_ms'],
                       speed_evidence='final_checks/speed_results.json',speed_model=benchmark['model'],timing_rounds=benchmark['values'])
            points.append(point)
            pruned_key=json.dumps(dict(parent['config'],prune_final_side=True),sort_keys=True)
            if not prunes or parent['study']!=root.name or parent['steps']!=15000 or pruned_key not in timing:continue
            cases=[v for v in prunes['cases'] if v['model'] in parent['aliases']]
            if sorted(v['seed'] for v in cases)!=[11,22,33]:continue
            assert all(v['status']=='PASS' for v in cases)
            removed=st.mean(v['original_parameters']-v['pruned_parameters'] for v in cases)/4
            params=parent['parameters']-removed
            benchmark=summaries[timing[pruned_key]]
            points.append(dict(point,id=len(points),name=parent['name']+' + final-side prune',
                               config=dict(parent['config'],prune_final_side=True),parameters=params,
                               derived_pruning=True,parent_id=parent['id'],prune_evidence=[v['check'] for v in cases],
                               ms=benchmark['median_ms'],cost=params/parent['mlp_parameters']*benchmark['median_ms']/reference_ms,
                               raw_parameter_ms=params*benchmark['median_ms'],speed_model=benchmark['model'],timing_rounds=benchmark['values'],
                               buffer_mib=st.mean(v['pruned_buffer_bytes'] for v in cases)/2**20))
    reserved_path=root/'final_checks/holdout_results.json'
    if reserved_path.exists():
        reserved=json.loads(reserved_path.read_text())
        scores={str(Path(v['run']).resolve()):v['evaluation'] for v in reserved['rows']}
        def score_key(item):return str((root.parent/item['study']/item['record']).parent.resolve())
        reserved_ids={}
        for original in list(points):
            sources=[v for v in original['sources'] if score_key(v) in scores]
            refs=[v for v in original['references'] if score_key(v) in scores]
            if not sources or len(refs)!=len(original['references']):continue
            records=[]
            for item in sources:
                raw=json.loads((root.parent/item['study']/item['record']).read_text())
                records.append(dict(seed=raw['arguments']['seed'],loss=scores[score_key(item)]['loss'],ms=1000*raw['training_seconds']/raw['arguments']['steps'],parameters=st.mean(raw['ffn_parameters_by_layer'])))
            per_seed=[]
            for seed in sorted({r['seed'] for r in records}):
                same=[r for r in records if r['seed']==seed]
                per_seed.append(dict(seed=seed,loss=st.mean(r['loss'] for r in same),ms=st.mean(r['ms'] for r in same),parameters=st.mean(r['parameters'] for r in same)))
            loss=st.mean(r['loss'] for r in per_seed);ref_loss=st.mean(scores[score_key(v)]['loss'] for v in refs)
            ms=st.mean(r['ms'] for r in per_seed) if original['timing_mode']=='worker' else original['ms']
            point=dict(original,id=len(points),quality_mode='reserved',validation_sha=reserved['holdout_sha256'],
                       loss=loss,loss_sd=st.stdev(r['loss'] for r in per_seed) if len(per_seed)>1 else None,
                       mlp_loss=ref_loss,loss_gap=loss-ref_loss,perplexity_ratio=math.exp(loss-ref_loss),
                       n=len(per_seed),runs=len(records),seeds=[r['seed'] for r in per_seed],per_seed=per_seed,
                       ms=ms,cost=original['parameters']/original['mlp_parameters']*ms/original['mlp_ms'],raw_parameter_ms=original['parameters']*ms,
                       sources=sources,references=refs,quality_evidence='final_checks/holdout_results.json')
            if point['derived_pruning']:
                point['parent_id']=reserved_ids[original['parent_id']]
            reserved_ids[original['id']]=point['id'];points.append(point)
    strata=sorted({(p['train_bytes'],p['steps'],p['batch'],p['context'],p['train_sha'],p['validation_sha'],p['timing_mode'],p['quality_mode']) for p in points})
    strata.sort(key=lambda k:(not(k[0]==4468985 and k[1]==15000 and k[2:4]==(4,64)),k[-1]!='reserved',k[-2]!='controlled',k))
    views=[]
    for i,key in enumerate(strata):
        subset=[p for p in points if (p['train_bytes'],p['steps'],p['batch'],p['context'],p['train_sha'],p['validation_sha'],p['timing_mode'],p['quality_mode'])==key]
        certified=[p for p in subset if p['n']>=3]
        front=sorted(frontier(certified),key=lambda p:p['cost'])
        for point in subset:point['view']=i;point['frontier']=point in front
        budgets=[]
        for ratio in (.05,.1,.2,.5,1.,2.,4.):
            affordable=[p for p in certified if p['cost']<=ratio]
            if affordable:
                best=min(affordable,key=lambda p:p['loss']);budgets.append(dict(max_cost=ratio,best_id=best['id'],loss=best['loss']))
        tolerances=[]
        for tolerance in (0.,.005,.01,.02,.05):
            eligible=[p for p in certified if p['loss_gap']<=tolerance]
            if eligible:
                best=min(eligible,key=lambda p:p['cost']);tolerances.append(dict(max_excess_loss=tolerance,best_id=best['id'],cost=best['cost'],loss=best['loss']))
        views.append(dict(id=i,label=f"{key[0]:,} training bytes · {key[1]:,} updates · B{key[2]} T{key[3]} · {key[-2]} timing · {key[-1]} loss",points=[p['id'] for p in subset],frontier=[p['id'] for p in front],budget_winners=budgets,quality_tolerance_winners=tolerances))
    out=root/'efficiency';out.mkdir(exist_ok=True)
    for evidence_name in ('holdout_results.json','speed_results.json','pareto_prune_checks.json'):
        evidence_path=root/'final_checks'/evidence_name
        if evidence_path.exists():(out/evidence_name).write_bytes(evidence_path.read_bytes())
    evidence=dict(generated_epoch=time.time(),metric='X=(mean learned FFN parameters / original MLP FFN parameters) × (measured milliseconds per update / original MLP milliseconds per update); Y=full-shard loss in nats/byte. Timing is the worker mean or controlled median according to the selected view. Lower-left is better.',
                  interpretation='Equal proportional weights for parameter size and step duration: half size at twice duration gives the same cost. This is an explicit preference, not a physical unit of energy or a uniquely optimal score. Learning loss is kept on Y rather than divided into an arbitrary ratio.',
                  timing='Each selectable view uses one timing source consistently: sustained worker means, or five-round controlled medians measured on the same GPU. No worker/controlled timing mixture within a view. Setup/evaluation excluded. Scheduler-suspended runs excluded, with reasons retained. Worker card/clock differences remain a limitation; these are observed cost frontiers, not universal speed rankings.',
                  grouping='Canonical configurations within each study; repeated runs of one seed are averaged before averaging seeds. At least three distinct seeds required for frontier/ranking. Incomplete cohorts remain visible as provisional points. Different training data, validation data, horizons, batches and contexts have separate selectable views of the same chart.',
                  points=points,views=views,excluded=excluded)
    (out/'metrics.json').write_text(json.dumps(evidence,indent=2)+'\n')
    with (out/'metrics.csv').open('w',newline='') as stream:
        fields=['id','name','study','steps','batch','context','train_bytes','n','parameters','ms','loss','loss_sd','cost','raw_parameter_ms','loss_gap','perplexity_ratio','buffer_mib','frontier','timing_mode','quality_mode','derived_pruning']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:p[k] for k in fields} for p in points)
    colors={'mlp':'#222222','hadmux':'#d97706','history':'#d97706','dense_ks2':'#2563eb'}
    fig=go.Figure();trace_views=[]
    for view in views:
        subset=[points[i] for i in view['points']]
        for point in subset:
            family=point['config']['family'];color=colors.get(family,'#64748b')
            symbol='star' if point['frontier'] else ('circle' if point['n']>=3 else 'circle-open')
            text=f"{point['name']}<br>Cost {point['cost']:.3f}× MLP · loss {point['loss']:.6f}<br>{point['parameters']:,.0f} learned parameters/FFN · {point['ms']:.3f} ms/update<br>{point['n']} seeds / {point['runs']} runs · loss SD {point['loss_sd'] if point['loss_sd'] is not None else 'unavailable'}<br>Fixed buffers {point['buffer_mib'] if point['buffer_mib'] is not None else 'unrecorded'} MiB<br>{point['study']}<br>{json.dumps(point['config'],sort_keys=True)}"
            fig.add_trace(go.Scatter(x=[point['cost']],y=[point['loss']],mode='markers',marker=dict(size=14 if point['frontier'] else 8,color=color,symbol=symbol,opacity=1 if point['n']>=3 else .35),hovertext=[text],hoverinfo='text',showlegend=False,visible=view['id']==0,name=point['name']))
            trace_views.append(view['id'])
        front=[points[i] for i in view['frontier']]
        fig.add_trace(go.Scatter(x=[p['cost'] for p in front],y=[p['loss'] for p in front],mode='lines',line=dict(color='#111827',width=1,dash='dot'),hoverinfo='skip',showlegend=False,visible=view['id']==0));trace_views.append(view['id'])
    buttons=[dict(label=v['label'],method='update',args=[{'visible':[i==v['id'] for i in trace_views]},{'title':f"Combined size × step-time cost versus loss<br><sup>{v['label']} — lower-left is better</sup>"}]) for v in views]
    fig.update_layout(title=f"Combined size × step-time cost versus loss<br><sup>{views[0]['label']} — lower-left is better</sup>",xaxis=dict(title='Combined cost = (FFN parameters / MLP) × (ms per update / MLP)',type='log'),yaxis_title='Validation loss (nats/byte)',template='plotly_white',height=740,margin=dict(t=140,b=90),updatemenus=[dict(buttons=buttons,x=0,y=1.15,xanchor='left',yanchor='top')],annotations=[dict(text='Orange: fixed Hadamard · Blue: learned-projection KS2 · Black: original MLP · Stars: three-seed cost/loss frontier · Open: provisional',xref='paper',yref='paper',x=0,y=-.14,showarrow=False,xanchor='left')])
    fig.add_vline(x=1,line_dash='dash',line_color='#9ca3af')
    fig.write_html(out/'size_speed_loss.html',include_plotlyjs='cdn',full_html=True)
    # GitHub's Markdown preview gets the identical default view as one static chart.
    default=views[0];subset=[points[i] for i in default['points']];fig2,ax=plt.subplots(figsize=(12,7))
    for family,label in [('dense_ks2','Learned-projection KS2'),('hadmux','Fixed Hadamard (current study)'),('history','Fixed Hadamard (prior study)'),('mlp','Original MLP')]:
        entries=[p for p in subset if p['config']['family']==family]
        complete=[p for p in entries if p['n']>=3];partial=[p for p in entries if p['n']<3]
        if complete:ax.scatter([p['cost'] for p in complete],[p['loss'] for p in complete],c=colors[family],s=35,label=label,alpha=.75,marker='s' if family=='history' else 'o')
        if partial:ax.scatter([p['cost'] for p in partial],[p['loss'] for p in partial],edgecolors=colors[family],facecolors='none',s=22,alpha=.3)
    front=[points[i] for i in default['frontier']]
    ax.plot([p['cost'] for p in front],[p['loss'] for p in front],':',color='#111827',lw=1)
    ax.scatter([p['cost'] for p in front],[p['loss'] for p in front],marker='*',s=170,c='#111827',zorder=5,label='Three-seed cost/loss frontier')
    # Spread frontier labels along the right margin to keep the dense near-MLP
    # cluster readable without hiding points or creating another chart.
    ymin=min(p['loss'] for p in subset);ymax=max(p['loss'] for p in subset)
    label_positions=[]
    for point in sorted(front,key=lambda p:p['loss']):
        proposed=point['loss']
        if label_positions:proposed=max(proposed,label_positions[-1]+.035*(ymax-ymin))
        label_positions.append(proposed)
        ax.annotate(point['name'],(point['cost'],point['loss']),xytext=(1.02,proposed),textcoords=ax.get_yaxis_transform(),fontsize=8,va='center',arrowprops=dict(arrowstyle='-',lw=.5,color='#64748b'))
    ax.axvline(1,color='#9ca3af',ls='--',lw=1);ax.set_xscale('log');ax.grid(alpha=.2)
    ax.set_xlabel('Combined cost = (learned FFN parameters / MLP) × (ms per update / MLP)\nSmaller size and faster updates both move left')
    ax.set_ylabel('Validation loss (nats/byte)');ax.set_title('Combined size–speed cost versus loss\n'+default['label']+' · lower-left is better')
    ax.legend(fontsize=8,loc='upper right');fig2.tight_layout();fig2.subplots_adjust(right=.76)
    for ext in ('png','svg'):fig2.savefig(out/f'size_speed_loss.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig2)
    report=['# Combined size–speed cost versus loss\n',evidence['metric']+'\n',
            '**The size and speed measures are multiplied into X; loss is Y.** Half the learned FFN parameters at twice the step time has the same X cost. The original MLP reference has X=1. Lower-left is better.\n',
            '![Combined cost versus loss](size_speed_loss.png)\n',
            '[Interactive version: hover and select a matched training regime](size_speed_loss.html). Download/open the HTML in a browser; it loads Plotly from its CDN. This is one chart; the selector changes which comparable data/horizon cohort is shown.\n',
            evidence['interpretation']+'\n',evidence['timing']+'\n',evidence['grouping']+'\n',
            'Once final scoring is available, the default is fresh reserved loss with controlled timings. Exploratory loss and worker timing remain selectable views of the same chart; neither different validation text nor timing sources are mixed within a view. Reserved scores are produced only after architecture selection and training end.\n',
            'Learned size is the average FFN parameter count across four layers. Fixed coefficient buffers are shown in hover/data but are not included in this user-selected learned-parameter cost. Therefore this metric is not total model storage, VRAM, energy, inference speed, or end-to-end training time.\n',
            'The 2k side/history model also learns an average 7,296 LayerNorm/time-mixing parameters per layer outside the FFN module, versus 288 for the original MLP. The frozen model’s 2,144 learned FFN parameters are additional to those external parameters. These external counts stay unchanged in the targeted F512→F1024→F2048 frozen-feature sweep.\n',
            'Pruned points, when available, reuse their parent checkpoint losses after per-seed trained-logit and gradient equivalence checks. They use independently measured pruned timings and reduced parameter counts; they are not additional independent training runs.\n']
    for view in views:
        report += ['## '+view['label']+'\n',table(['Frontier model','Seeds','Learned / FFN','ms/update','Combined cost × MLP','Loss'],[[points[i]['name'],points[i]['n'],f"{points[i]['parameters']:,.0f}",f"{points[i]['ms']:.3f}",f"{points[i]['cost']:.3f}",f"{points[i]['loss']:.6f}"] for i in view['frontier']]),
                   table(['Allowed excess loss vs MLP','Lowest-cost eligible model','Combined cost','Loss'],[[f"{v['max_excess_loss']:.3f}",points[v['best_id']]['name'],f"{v['cost']:.3f}",f"{v['loss']:.6f}"] for v in view['quality_tolerance_winners']])]
    report += ['Frontiers use observed means and are exploratory. Small differences within seed variation or timing variation do not establish a robust winner. No new MLP settings were trained for this analysis.\n','[All metrics and exclusions](metrics.json) · [CSV](metrics.csv).\n']
    if (out/'holdout_results.json').exists():report += ['[Fresh reserved scores](holdout_results.json).\n']
    if (out/'speed_results.json').exists():report += ['[Controlled timing rounds](speed_results.json).\n']
    if (out/'pareto_prune_checks.json').exists():report += ['[Per-seed trained pruning checks](pareto_prune_checks.json).\n']
    (out/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(points=len(points),runs=len(allrows),views=len(views),default_frontier=[points[i]['name'] for i in views[0]['frontier']],report=str(out/'REPORT.md'))))


if __name__=='__main__':main()
