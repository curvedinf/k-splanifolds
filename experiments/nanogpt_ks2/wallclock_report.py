"""Build a reproducible ledger and combined FFN-size / wall-time / loss plot."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect(root):
    rows=[]
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text())
        if 'candidate' not in result:
            continue
        cfg=result['candidate']
        row=dict(source=str(path),wave=path.parent.parent.name,job=path.parent.name,
                 seed=result['seed'],kind=result['kind'],steps=result['steps'],
                 sequence=cfg.get('sequence',128),ffn=result['counts']['ffn'],
                 total_parameters=result['counts']['total'],loss=result['final_val_loss'],
                 seconds=result['seconds'],step_ms=result['step_ms'],candidate=cfg,
                 history=result['history'],gpu=result['gpu'])
        row['size_wall_cost']=row['ffn']/743424*row['seconds']/39.8978326653208
        row['size_step_cost']=row['ffn']/743424*row['step_ms']/76.6449
        rows.append(row)
    prior=Path(__file__).parent/'reports/2026-10-23-ks3-control/all_results.json'
    for wave,items in json.loads(prior.read_text()).items():
        if wave not in ('graph','graph_repeat'):
            continue
        for item in items:
            if item['kind']!='mlp':continue
            seconds=item['seconds'];ms=1000*item['train_seconds']/item['steps']
            rows.append(dict(source=str(prior),wave=wave,job=f"mlp500_{item['seed']}",
                seed=item['seed'],kind='mlp',steps=500,sequence=128,ffn=743424,
                total_parameters=item['counts']['total'],loss=item['final_val_loss'],
                seconds=seconds,step_ms=ms,candidate=dict(kind='mlp',steps=500),
                history=item['history'],gpu=item.get('gpu'),
                size_wall_cost=seconds/39.8978326653208,size_step_cost=ms/76.6449))
    for row in rows:
        row['training_tokens']=row['steps']*row['sequence']
        row['tokens_per_second']=row['training_tokens']/row['seconds']
    return rows


def cohorts(rows):
    groups={}
    for row in rows:
        cfg={k:v for k,v in row['candidate'].items() if k!='wall_budget'}
        key=json.dumps(cfg,sort_keys=True)
        groups.setdefault(key,[]).append(row)
    result=[]
    for key,items in groups.items():
        # Timing reruns are averaged within a seed, not extra quality replicates.
        seeds={seed:[x for x in items if x['seed']==seed] for seed in {x['seed'] for x in items}}
        mean=lambda name:statistics.mean(statistics.mean(x[name] for x in v) for v in seeds.values())
        sd=lambda name:(statistics.stdev(statistics.mean(x[name] for x in v) for v in seeds.values())
                        if len(seeds)>1 else None)
        result.append(dict(candidate=json.loads(key),seeds=sorted(seeds),
                           runs=len(items),ffn=items[0]['ffn'],kind=items[0]['kind'],
                           steps=items[0]['steps'],sequence=items[0]['sequence'],
                           loss_sd=sd('loss'),seconds_sd=sd('seconds'),cost_sd=sd('size_wall_cost'),
                           **{name:mean(name) for name in ('loss','seconds','step_ms','size_wall_cost','size_step_cost',
                                                         'training_tokens','tokens_per_second')}))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('experiments/nanogpt_ks2/runs/wallclock_20261023'))
    ap.add_argument('--out',type=Path,default=Path('experiments/nanogpt_ks2/reports/2026-10-24-ks3-wallclock'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=collect(args.root);groups=cohorts(rows)
    provenance=[]
    for path in sorted(args.root.glob('*/*/config.json')):
        provenance.append(dict(source=str(path),metadata=json.loads(path.read_text())))
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for path in [args.root/'study.json',*args.root.glob('milestone*.json'),*args.root.glob('target*.json')]:
        if path.exists():(args.out/path.name).write_bytes(path.read_bytes())
    reload_check=args.root/'reload_prefix_check.json'
    if reload_check.exists():(args.out/reload_check.name).write_bytes(reload_check.read_bytes())
    activity=args.root/'local_screen/local_zero512_1500_11/geometry_activity.json'
    if activity.exists():(args.out/'geometry_activity.json').write_bytes(activity.read_bytes())
    for name in ('finalists.json','long_selection.json','combined_selection.json','phase_profile.json',
                 'tiled_phase_profile.json','tiled_equivalence.json','tiled_long_equivalence.json',
                 'tiled_long_decision.json','final_verification.json',
                 'final_reload_check.json','reserved_validation.json','final_audit.json',
                 'best_config.json','stopped.json'):
        path=args.root/name
        if path.exists():(args.out/name).write_bytes(path.read_bytes())
    attempts=[]
    for path in sorted(args.root.glob('*/completion.json')):
        attempts.extend(dict(wave=path.parent.name,**record) for record in json.loads(path.read_text()))
    (args.out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    (args.out/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.out/'cohorts.json').write_text(json.dumps(groups,indent=2)+'\n')
    keys=['wave','job','seed','kind','steps','sequence','training_tokens','tokens_per_second',
          'ffn','total_parameters','loss','seconds','step_ms','size_wall_cost','size_step_cost','gpu','source']
    with (args.out/'runs.csv').open('w') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for kind,color,label in [('mlp','#b23a48','Unchanged MLP'),('ks3','#167d9a','KS3')]:
        single=[g for g in groups if g['kind']==kind and len(g['seeds'])<3]
        replicated=[g for g in groups if g['kind']==kind and len(g['seeds'])>=3]
        ax.scatter([g['size_wall_cost'] for g in single],[g['loss'] for g in single],
                   color=color,alpha=.4,s=22,marker='x')
        ax.scatter([g['size_wall_cost'] for g in replicated],[g['loss'] for g in replicated],
                   color=color,s=65,label=label+' (3 seeds)')
        for g in replicated:
            ax.errorbar(g['size_wall_cost'],g['loss'],xerr=g['cost_sd'],yerr=g['loss_sd'],
                        fmt='none',ecolor=color,alpha=.7,capsize=2,zorder=0)
            label=f"{g['steps']:,} × {g['sequence']}"
            offset=(6,8) if kind=='mlp' else (7,-14)
            align='left'
            if kind=='mlp' and g['size_wall_cost']==max(x['size_wall_cost'] for x in groups):
                offset=(-7,8);align='right'
            if kind=='ks3' and g['steps']==650:
                if g['candidate'].get('rank',1)==4:
                    label+=' (R4)';offset=(8,-28)
                else:offset=(-8,-14);align='right'
            ax.annotate(label,(g['size_wall_cost'],g['loss']),xytext=offset,
                        ha=align,textcoords='offset points',fontsize=7)
    # Only replicated configurations establish this reported frontier.
    frontier=[];best=float('inf')
    for g in sorted((g for g in groups if len(g['seeds'])>=3),key=lambda x:x['size_wall_cost']):
        if g['loss']<best:
            frontier.append(g);best=g['loss']
    ax.plot([g['size_wall_cost'] for g in frontier],[g['loss'] for g in frontier],
            color='#444444',ls='--',lw=1,label='Replicated frontier')
    ax.set(xscale='log',xlabel='(FFN parameters / 743,424) × (complete run seconds / 39.898)',
           ylabel='Validation cross-entropy (128-token context)',
           title='Capacity-preserving KS3: combined size × wall-time cost versus loss')
    ax.grid(alpha=.2);ax.legend();fig.savefig(args.out/'size_wall_loss.png',dpi=180)
    plt.close(fig)
    (args.out/'frontier.json').write_text(json.dumps(frontier,indent=2)+'\n')
    finalized=json.loads((args.root/'study.json').read_text()).get('status') in ('complete','stopped_by_user')
    status=('Completed measurements at the study cutoff; see [the report](REPORT.md) for interpretation, interrupted work, and validation limits.' if finalized else
            'This generated ledger is an interim snapshot. Final interpretation and reserved validation will be added after the research window.')
    lines=['# KS3 wall-clock convergence study','',status,'',
           '![Combined size × wall-time cost versus loss](size_wall_loss.png)','',
           'X multiplies learned FFN size by **complete training-run time**; Y is validation loss. Lower-left is better. The earlier size × step-time metric is also retained in the CSV, but run time is the current optimization target. Different training lengths and contexts are explicitly recorded. Crosses are single-seed screens; the frontier uses configurations with three distinct seeds.','',
           '| Kind | Steps | Training context | Training tokens | Seeds | FFN parameters | Seconds | CE |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in sorted(groups,key=lambda x:(x['steps'],x['sequence'],x['loss'])):
        lines.append(f"| {g['kind']} | {g['steps']:,} | {g['sequence']} | {g['training_tokens']:,.0f} | {len(g['seeds'])} | {g['ffn']:,} | {g['seconds']:.2f} | {g['loss']:.6f} |")
    lines += ['', '[All runs and configurations](runs.json) · [CSV](runs.csv) · [Seed-averaged cohorts](cohorts.json) · [Launcher outcomes, including failures](attempts.json)', '']
    (args.out/'LEDGER.md').write_text('\n'.join(lines))
    print(json.dumps(dict(runs=len(rows),cohorts=len(groups),out=str(args.out))))


if __name__=='__main__':main()
