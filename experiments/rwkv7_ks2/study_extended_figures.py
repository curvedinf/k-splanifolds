"""Standalone extended-study figures with individual seeds and fixed references."""
import argparse,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from study_collect import collect


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=collect(a.root);out=a.root/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 for wave,title in [('wave14_spine_scale','Geometry-input scaling:15k updates'),('wave11_conditioning','KS2 conditioning and rank:15k updates'),('wave12_fixed_geometry','Minimal learned geometry:15k updates'),('wave12b_rank_freezing','Chart rank and freezing:15k updates'),('wave12c_original_geometry','Original equation and bounded inputs:15k updates')]:
  rs=[r for r in rows if r['wave']==wave]
  if not rs:continue
  names=sorted({r['config']['name'] for r in rs},key=lambda name:st.mean(r['full_validation_loss'] for r in rs if r['config']['name']==name))
  fig,ax=plt.subplots(figsize=(10,max(4,.35*len(names)+1.5)))
  for i,name in enumerate(names):
   vals=[r['full_validation_loss'] for r in rs if r['config']['name']==name];ax.scatter(vals,[i]*len(vals),s=30,color='#246f9d',alpha=.7)
   ax.scatter(st.mean(vals),i,marker='D',s=32,color='black',zorder=3)
  ax.axvline(1.620779315630595,color='#ba493a',linestyle='--',label='Original unchanged MLP:3-seed mean')
  ax.axvline(1.6165178616841633,color='#427a47',linestyle=':',label='Earlier warp-only KS2:3-seed mean')
  ax.set_yticks(range(len(names)),[f'{n} (n={sum(r["config"]["name"]==n for r in rs)})' for n in names]);ax.invert_yaxis();ax.set_xlabel('Exploratory full-shard loss, nats/byte (lower is better)');ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.legend(loc='best',fontsize=8)
  fig.text(.99,.01,'Dots:seeds; black diamond:mean. Partial cohorts are provisional.',ha='right',fontsize=8);fig.tight_layout(rect=(0,.03,1,1))
  for ext in ('svg','png'):fig.savefig(out/f'{wave}.{ext}',dpi=180)
  plt.close(fig)
 rs=[r for r in rows if r['steps']>=100000 and r['family']=='dense_ks2']
 if rs:
  fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True)
  for ax,pool in zip(axes,('original','larger')):
   found=[r for r in rs if r['job'].startswith(pool+'_')];names=sorted({r['config']['name'] for r in found})
   for i,name in enumerate(names):
    subgroup=[r for r in found if r['config']['name']==name];gaps=[]
    for r in subgroup:
     ref=next(x for x in rows if x['wave']=='wave8_long' and x['job']==f'{pool}_original_mlp_{r["seed"]}')
     gaps.append(r['full_validation_loss']-ref['full_validation_loss'])
    ax.scatter(gaps,[i]*len(gaps),s=30,color='#246f9d');ax.scatter(st.mean(gaps),i,marker='D',color='black',s=32)
   ax.axvline(0,color='#ba493a',linestyle='--');ax.set_yticks(range(len(names)),[f'{name} (n={sum(r["config"]["name"]==name for r in found)})' for name in names]);ax.invert_yaxis();ax.set_title(pool+' training pool');ax.set_xlabel('KS2 loss − unchanged MLP loss');ax.grid(axis='x',alpha=.2)
   ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  fig.suptitle('100k updates:paired seed gaps against the fixed reference');fig.text(.99,.01,'Negative favors KS2; dots are paired seeds, diamond is mean. Partial cohorts provisional.',ha='right',fontsize=8);fig.tight_layout(rect=(0,.04,1,.95))
  for ext in ('svg','png'):fig.savefig(out/f'extended_long_gaps.{ext}',dpi=180)
  plt.close(fig)
 selected=[r for r in rows if r['wave']=='wave13_selected_long']
 if selected:
  curves=[]
  def scores(record):
   values={x['evaluation_step']:x['loss'] for x in record['intermediate_evaluations']}
   values[record['steps']]=record['full_validation_loss'];return values
  for r in selected:
   pool='larger' if r['data_train_bytes']>10000000 else 'original'
   reference=next(x for x in rows if x['wave']=='wave8_long' and x['job']==f'{pool}_original_mlp_{r["seed"]}')
   control=scores(reference)
   for step,loss in scores(r).items():
    assert step in control
    curves.append(dict(pool=pool,model=r['config']['name'],seed=r['seed'],step=step,ks2_loss=loss,mlp_loss=control[step],gap=loss-control[step],source=r['source']))
  (a.root/'extended_long_trajectories.json').write_text(json.dumps(dict(protocol='Intermediate checkpoints and final scores from the same100k cosine schedule. Each gap pairs the same seed, data pool and update count with the original unchanged MLP. These are not separately annealed15k runs.',rows=curves),indent=2)+'\n')
  names=sorted({r['model'] for r in curves});colors=dict(zip(names,plt.get_cmap('tab10').colors))
  fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
  for ax,pool in zip(axes,('original','larger')):
   for name in names:
    subset=[r for r in curves if r['model']==name and r['pool']==pool]
    if not subset:continue
    steps=sorted({r['step'] for r in subset});values=[[r['gap'] for r in subset if r['step']==step] for step in steps]
    ax.plot(steps,[st.mean(v) for v in values],marker='o',color=colors[name],label=f'{name} (n={len(values[-1])})')
    ax.fill_between(steps,[min(v) for v in values],[max(v) for v in values],alpha=.12,color=colors[name])
   ax.axhline(0,color='#ba493a',linestyle='--');ax.set_title(pool+' training pool');ax.set_xlabel('Updates within the same 100k schedule');ax.grid(alpha=.2);ax.legend(fontsize=7)
  axes[0].set_ylabel('KS2 loss − unchanged MLP loss');fig.suptitle('Selected KS2 follow-ups: matched-horizon learning curves')
  fig.text(.99,.01,'Line:paired-seed mean. Band:seed range, not confidence interval. Negative favors KS2.',ha='right',fontsize=8);fig.tight_layout(rect=(0,.04,1,.95))
  for ext in ('svg','png'):fig.savefig(out/f'selected_long_trajectories.{ext}',dpi=180)
  plt.close(fig)
 combinations=[r for r in rows if r['wave']=='wave16_combination_long']
 if combinations:
  curves=[]
  def checkpoint_losses(record):
   values={x['evaluation_step']:x['loss'] for x in record['intermediate_evaluations']}
   values[record['steps']]=record['full_validation_loss'];return values
  reference_rows=[r for r in rows if r['wave']=='wave13_selected_long' and r['config']['name']=='spine_rank1_fused_repeat']
  for r in combinations+reference_rows:
   pool='larger' if r['data_train_bytes']>10000000 else 'original';control=next(x for x in rows if x['wave']=='wave8_long' and x['job']==f'{pool}_original_mlp_{r["seed"]}')
   controls=checkpoint_losses(control);name=r['config']['name'] if r in combinations else 'Existing rank-1 KS2'
   for step,loss in checkpoint_losses(r).items():curves.append(dict(pool=pool,model=name,seed=r['seed'],step=step,loss=loss,mlp_loss=controls[step],gap=loss-controls[step]))
  (a.root/'combination_long_trajectories.json').write_text(json.dumps(dict(protocol='Paired-seed gaps at matching checkpoints within100k cosine schedules. Existing rank1 reference frompreviouswave, unchangedMLP fromwave8. No reserved data.',rows=curves),indent=2)+'\n')
  names=sorted({r['model'] for r in curves});colors=dict(zip(names,plt.get_cmap('tab10').colors));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
  for ax,pool in zip(axes,('original','larger')):
   for name in names:
    subset=[r for r in curves if r['pool']==pool and r['model']==name]
    if not subset:continue
    steps=sorted({r['step'] for r in subset});values=[[r['gap'] for r in subset if r['step']==step] for step in steps]
    color='#555555' if name=='Existing rank-1 KS2' else colors[name]
    ax.plot(steps,[st.mean(v) for v in values],marker='o',linestyle='--' if name=='Existing rank-1 KS2' else '-',color=color,label=f'{name} (n={len(values[-1])})')
    ax.fill_between(steps,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.10)
   ax.axhline(0,color='#ba493a',linestyle=':');ax.set_title(pool+' training pool');ax.set_xlabel('Updates within the 100k schedule');ax.grid(alpha=.2);ax.legend(fontsize=7)
  axes[0].set_ylabel('KS2 loss − unchanged MLP loss');fig.suptitle('Do combined KS2 gains persist with longer training?');fig.text(.99,.01,'Line:paired-seed mean. Shading:seed range, not confidence interval. Partial cohorts provisional.',ha='right',fontsize=8);fig.tight_layout(rect=(0,.04,1,.95))
  for ext in ('svg','png'):fig.savefig(out/f'combination_long_trajectories.{ext}',dpi=180)
  plt.close(fig)
 probe=a.root/'rank_input_statistics.json'
 if probe.exists():
  stats=json.loads(probe.read_text())['rows'];rank_names={1:'spine_rank1',2:'spine_rank2',4:'spine_rank4',8:'spine_eager_reference',16:'spine_rank16'}
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for state,label,color in [('initialized','Initialized','#8b8b8b'),('trained_15k','Trained 15k','#246f9d')]:
   values={rank:[l for row in stats if row['model']==name and row['state']==state for l in row['layers']] for rank,name in rank_names.items()}
   for ax,field,title in zip(axes,('basis_argument_rms','basis_design_condition_median','summary_covariance_participation_ratio'),('Basis-input RMS','Basis design condition','Summary covariance participation')):
    agg=st.median if field=='basis_design_condition_median' else st.mean
    ax.plot(list(rank_names),[agg(v[field] for v in values[rank]) for rank in rank_names],marker='o',label=label,color=color)
    ax.set_xscale('log',base=2);ax.set_xticks(list(rank_names),[str(rank) for rank in rank_names]);ax.set_xlabel('Coordinates per chart');ax.set_title(title);ax.grid(alpha=.2)
    if field=='basis_design_condition_median':ax.set_yscale('log')
  axes[0].legend();fig.suptitle('Identical training windows; 368 projected features, three seeds')
  fig.text(.99,.01,'Aggregated over four layers × three seeds. Descriptive statistics, not causal effects.',ha='right',fontsize=8);fig.tight_layout(rect=(0,.05,1,.94))
  for ext in ('svg','png'):fig.savefig(out/f'rank_input_statistics.{ext}',dpi=180)
  plt.close(fig)
if __name__=='__main__':main()
