"""Standalone research figures; complete run index remains the authoritative data."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_collect import collect

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=collect(a.root);out=a.root/'figures';out.mkdir(exist_ok=True)
 def save(fig,name):
  fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
 controls=[r for r in rows if r['steps']==15000 and r['batch_size']==4 and r['sequence_length']==64 and r['seed']==11 and r['config'].get('attention')!='softmax']
 def group(r):
  if not r['geometry_parameters'] and not r['frozen_geometry']:return 'ReLU² / feature controls'
  if r['family']=='history':return 'History KS2'
  if r['config'].get('input_fixed'):return 'Fixed-input KS2'
  return 'Learned-input KS2'
 colors={'ReLU² / feature controls':'#2166ac','History KS2':'#b35806','Fixed-input KS2':'#1b7837','Learned-input KS2':'#762a83'}
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for label,color in colors.items():
  rs=[r for r in controls if group(r)==label]
  for ax,key,scale in [(axes[0],'parameters',1000),(axes[1],'ms_per_step',1)]:
   ax.scatter([r[key]/scale for r in rs],[r['full_validation_loss'] for r in rs],label=label,color=color,s=28,alpha=.75)
 for r in controls:
  if (r['wave']=='initial' and r['job'] in ('mlp','history')) or r['job'] in ('dense31','dense31_relu2','undamped33_relu'):
   label={'mlp':'MLP','history':'History baseline','dense31':'KS2, learned projections','dense31_relu2':'KS2 + ReLU²','undamped33_relu':'KS2, no damping + ReLU²'}[r['job']]
   # Keep dense-budget labels inside the panel rather than spilling into its neighbour.
   if r['job'] in ('history','dense31','dense31_relu2'):
    positions={'history':(.48,.64),'dense31':(.48,.27),'dense31_relu2':(.52,.13)}
    axes[0].annotate(label,(r['parameters']/1000,r['full_validation_loss']),xytext=positions[r['job']],
                     textcoords='axes fraction',fontsize=7,arrowprops=dict(arrowstyle='-',color='#777',lw=.6))
   axes[1].annotate(label,(r['ms_per_step'],r['full_validation_loss']),xytext=(4,5),textcoords='offset points',fontsize=7)
 for ax in axes:ax.set_ylabel('Full-shard validation loss (nats / byte)');ax.grid(alpha=.2)
 axes[0].set_xlabel('Learned parameters (thousands)');axes[1].set_xlabel('Observed training time (ms / optimizer step)');axes[1].legend(fontsize=8)
 fig.suptitle('Exploratory screens: 15,000 steps, 256 training bytes per step, seed 11');fig.tight_layout();save(fig,'short_run_frontier')
 selected=[('initial','mlp','MLP'),('initial','history','History KS2'),('wave2','dense31','KS2, learned projections'),('wave2','dense31_relu2','KS2 + ReLU²')]
 fig,ax=plt.subplots(figsize=(8,4.8))
 for wave,job,label in selected:
  rs=[r for r in rows if r['wave']==wave and r['job']==job and r['seed']==11]
  if not rs:continue
  history=json.loads((a.root/rs[0]['source']).read_text())['history'];h=[v for v in history if v['step']>=1000]
  ax.plot([v['step'] for v in h],[v['val_loss'] for v in h],label=label)
 ax.set(xlabel='Optimizer steps',ylabel='Fixed 8,192-target validation sample (nats / byte)',title='Learning curves use the same sampled validation metric');ax.legend();ax.grid(alpha=.2);save(fig,'short_learning_curves')
 tracks=[('History KS2',[('wave1','geometry_lr_quarter',.25),('wave1','baseline_repeat',1),('wave1','geometry_lr4',4),('wave3','history_geo8',8),('wave3','history_geo16',16)]),
 ('Learned-projection KS2',[('wave2','dense31_geolr_quarter',.25),('wave2','dense31',1),('wave2','dense31_geolr4',4)]),
 ('KS2 + ReLU²',[('wave4_activation','relu_geo_quarter',.25),('wave4_activation','relu_geo_half',.5),('wave2','dense31_relu2',1),('wave4_activation','relu_geo2',2),('wave3','dense_relu_geo4',4),('wave3','dense_relu_geo8',8),('wave3','dense_relu_geo16',16)])]
 fig,ax=plt.subplots(figsize=(8,4.8))
 for label,items in tracks:
  points=[]
  for wave,job,lr in items:
   rs=[r for r in rows if r['wave']==wave and r['job']==job and r['seed']==11]
   if rs:points.append((lr,rs[0]['full_validation_loss']))
  if points:ax.plot(*zip(*sorted(points)),marker='o',label=label)
 ax.set_xscale('log',base=2);ax.set(xlabel='Geometry learning-rate multiplier',ylabel='Full-shard loss (nats / byte)',title='Geometry learning rate interacts with the decoder architecture');ax.legend();ax.grid(alpha=.2);save(fig,'geometry_learning_rate')
 print(str(out))
if __name__=='__main__':main()
