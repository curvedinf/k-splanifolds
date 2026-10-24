"""Plot complete replicated cohorts and actual checkpoint training times."""
import argparse
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_collect import collect


LABELS={
    'had_full_r8':'Full R8, F512, side2k',
    'had_spine_r4_relu2':'R4 spine + ReLU², F512, side2k',
    'had_spine_r8_frozen':'Frozen R8 spine, F512, side2k',
    'had_full_r8_relu2':'Full R8 + ReLU², F512, side2k',
    'had_spine_r1_powers':'R1 powers, F512, side2k',
    'scale_full_f1024_s2048':'Full R8 + ReLU², F1024, side2k',
    'scale_full_f512_s8192':'Full R8 + ReLU², F512, side8k',
    'scale_r1_f2048_s2048':'R1 powers, F2048, side2k',
    'scale_r1_matched_s2048':'R1 powers, F6560, side2k',
    'scale_r8spine_f2048_s2048':'R8 spine + ReLU², F2048, side2k',
    'scale_r8spine_f2048_s8192':'R8 spine + ReLU², F2048, side8k',
    'original_mlp_repeat':'Original MLP',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'figures';out.mkdir(exist_ok=True)
    rows=collect(root);summary=json.loads((root/'comparison_summary.json').read_text())
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    long=[r for r in rows if r['wave']=='wave3_long']
    names=sorted({r['config']['name'] for r in long})
    if not names:return
    fig,axes=plt.subplots(1,2,figsize=(13,5.6));evidence=[]
    for index,name in enumerate(names):
        cohort=[r for r in long if r['config']['name']==name]
        if len(cohort)!=3:continue
        points={}
        for r in cohort:
            raw=json.loads((root/r['source']).read_text())
            events=raw['intermediate_evaluations']+[raw['tiled_evaluation']]
            for event in events:
                step=event['evaluation_step'];time=next(h['training_seconds'] for h in raw['history'] if h['step']==step)
                points.setdefault(step,[]).append(dict(seed=r['seed'],seconds=time,loss=event['loss']))
        complete=sorted(k for k,v in points.items() if len(v)==3)
        loss=[st.mean(v['loss'] for v in points[k]) for k in complete]
        sd=[st.stdev(v['loss'] for v in points[k]) for k in complete]
        seconds=[st.mean(v['seconds'] for v in points[k]) for k in complete]
        label=LABELS.get(name.removeprefix('long_'),name);color=plt.get_cmap('tab10')(index)
        for ax,x in zip(axes,[complete,seconds]):
            ax.plot(x,loss,'o-',label=label,color=color,markersize=4)
            ax.fill_between(x,[m-s for m,s in zip(loss,sd)],[m+s for m,s in zip(loss,sd)],color=color,alpha=.12)
        evidence.append(dict(name=name,points=points))
    for ax in axes:
        ax.set_ylabel('Exploratory validation loss (nats/byte)');ax.grid(alpha=.25)
    axes[0].set_xlabel('Updates');axes[1].set_xlabel('Measured training seconds')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Common training horizon and cosine schedule — three seeds, shaded ± SD')
    fig.subplots_adjust(bottom=.23,top=.88,wspace=.25)
    for ext in ('svg','png'):fig.savefig(out/f'long_curves.{ext}',dpi=150,bbox_inches='tight')
    (root/'learning_curve_points.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
