"""Final comparative figures from audited completed runs, no smoothing or best-seed selection."""
import argparse
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_collect import collect


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=collect(a.root);out=a.root/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def get(wave,job,seed=None):
        return next((r for r in rows if r['wave']==wave and r['job']==job and (seed is None or r['seed']==seed)),None)
    original=[get('initial','mlp',11),get('initial','mlp',22),get('wave3','mlp_seed33',33)]
    original=[r for r in original if r]
    stages=[('Demuxed history',[get('initial','history',11),get('initial','history',22)]),
            ('Learned projections',[get('wave2','dense31')]),
            ('Learned projections + ReLU²',[get('wave2','dense31_relu2'),get('wave3','dense_relu_seed22'),get('wave3','dense_relu_seed33')]),
            ('Undamped + ReLU²',[get('wave4_activation',f'undamped_fused_{s}') for s in (11,22,33)]),
            ('Learned spine + ReLU²',[get('wave4_activation',f'spine_fused_{s}') for s in (11,22,33)]),
            ('Frozen spine + ReLU²',[get('wave4_activation','spine_frozen'),get('wave6_budget','spine_frozen_22'),get('wave6_budget','spine_frozen_33')])]
    fig,ax=plt.subplots(figsize=(10,5))
    ticklabels=[]
    for y,(name,rs) in enumerate(stages):
        rs=[r for r in rs if r]
        if not rs:continue
        values=[r['full_validation_loss'] for r in rs]
        ax.errorbar(st.mean(values),y,xerr=st.stdev(values) if len(values)>1 else 0,fmt='o',color='#80468a',capsize=4)
        ax.scatter(values,[y]*len(values),s=15,color='#80468a',alpha=.4)
        ticklabels.append(f"{name}\n{rs[0]['parameters']:,} params · n={len(rs)}")
    if original:ax.axvline(st.mean(r['full_validation_loss'] for r in original),ls='--',color='#1a6095',label='Original unchanged MLP: three-seed mean')
    ax.set_yticks(range(len(ticklabels)),ticklabels);ax.invert_yaxis()
    ax.set(xlabel='15k full-shard validation loss (nats / byte; lower is better)',title='KS2 configurations against the fixed original MLP')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=8);save(fig,'ks2_progress')
    fig,ax=plt.subplots(figsize=(8,5))
    for prefix,label,color in [('ks2_c','KS2 + ReLU²','#a64277'),('mlp_h','MLP, matched input scale','#206ca5')]:
        points=[]
        for width in ([2,4,8,16] if prefix=='ks2_c' else [25,50,99,197]):
            rs=[r for r in rows if r['wave']=='wave6_budget' and r['job'].startswith(prefix+str(width)+'_')]
            if not rs:continue
            x=st.mean(r['parameters'] for r in rs)/1000;ys=[r['full_validation_loss'] for r in rs]
            points.append((x,st.mean(ys),st.stdev(ys) if len(ys)>1 else 0))
            ax.scatter([x]*len(ys),ys,color=color,s=18,alpha=.45)
        if points:
            x,y,e=zip(*points);ax.errorbar(x,y,yerr=e,color=color,marker='o',capsize=4,label=label)
    ax.set(xlabel='Total learned parameters (thousands)',ylabel='Full-shard loss (nats / byte)',title='Small decoder budgets: 15k steps, up to three seeds')
    ax.grid(alpha=.2);ax.legend();save(fig,'budget_curve')
    long=[r for r in rows if r['wave']=='wave8_long']
    labels={'original_mlp':'Original MLP','spine46':'Spine KS2 46','undamped33':'Undamped KS2 33','frozen46':'Frozen spine 46','history':'History KS2'}
    colors={'original_mlp':'#206ca5','spine46':'#a64277','undamped33':'#e38d25','frozen46':'#7e9c32','history':'#777777'}
    if long:
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
        for ax,pool in zip(axes,('original','larger')):
            for name in labels:
                rs=[r for r in long if r['job'].startswith(pool+'_'+name+'_')]
                if not rs:continue
                hs=[json.loads((a.root/r['source']).read_text())['history'] for r in rs]
                steps=sorted(set.intersection(*[{h['step'] for h in hist if h['step']>=1000} for hist in hs]))
                ys=[]
                for step in steps:ys.append(st.mean(next(h['val_loss'] for h in hist if h['step']==step) for hist in hs))
                ax.plot(steps,ys,color=colors[name],label=f'{labels[name]} (n={len(rs)})')
            ax.set(title=f'{pool.capitalize()} training pool',xlabel='Optimizer steps');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('Fixed sampled validation loss (nats / byte)');save(fig,'long_learning_curves')
        fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
        for ax,pool in zip(axes,('original','larger')):
            names=[name for name in labels if any(r['job'].startswith(pool+'_'+name+'_') for r in long)]
            for x,name in enumerate(names):
                rs=[r for r in long if r['job'].startswith(pool+'_'+name+'_')];ys=[r['full_validation_loss'] for r in rs]
                offsets=[(i-(len(ys)-1)/2)*.08 for i in range(len(ys))]
                ax.scatter([x+v for v in offsets],ys,color=colors[name],s=35)
                ax.errorbar(x,st.mean(ys),yerr=st.stdev(ys) if len(ys)>1 else 0,color='black',marker='_',markersize=15,capsize=4)
            ax.set_xticks(range(len(names)),[labels[n]+f" (n={sum(r['job'].startswith(pool+'_'+n+'_') for r in long)})" for n in names],rotation=25,ha='right')
            ax.set_title(pool.capitalize()+' pool');ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('100k full-shard loss (nats / byte)')
        fig.suptitle('Points: individual seeds · Black marks: mean ± seed SD',fontsize=10)
        save(fig,'long_final_comparison')
    print(out)


if __name__=='__main__':main()
