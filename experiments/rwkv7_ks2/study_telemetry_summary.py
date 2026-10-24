"""Summarize point-sampled GPU activity; do not equate busy time with peak FLOPs."""
import argparse
import datetime
import json
from pathlib import Path
import re
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.root/'study_state.json').read_text())
    cutoff=state['deadline_epoch']
    if (a.root/'training_complete.json').exists():
        marker=json.loads((a.root/'training_complete.json').read_text())
        cutoff=marker.get('finished_epoch',marker.get('epoch',cutoff))
    start=datetime.datetime.fromisoformat(state['start_utc']).timestamp()
    windows=[dict(name='initial_and_four_hour_extension',start=start,end=cutoff)]
    archived=a.root/'training_complete_before_extension2.json'
    if archived.exists() and (a.root/'extension2_start.json').exists():
        previous=json.loads(archived.read_text());extension=json.loads((a.root/'extension2_start.json').read_text())
        windows=[dict(name='initial_and_four_hour_extension',start=start,end=previous['finished_epoch']),
                 dict(name='two_hour_combination_extension',start=extension['start_epoch'],end=cutoff)]
    rows=[]
    for line in (a.root/'gpu_telemetry.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if r['returncode']==0 and any(w['start']<=r['epoch']<=w['end'] for w in windows) and all('GPU use (%)' in r['stats'].get(f'card{i}',{}) for i in range(4)):
            rows.append(r)
    cards=[]
    for i in range(4):
        values=[r['stats'][f'card{i}'] for r in rows]
        busy=[v for v in values if float(v['GPU use (%)'])>=90]
        clocks=[int(re.search(r'(\d+)Mhz',v['sclk clock speed:']).group(1)) for v in busy]
        cards.append(dict(gpu=i,samples=len(values),mean_reported_use_percent=statistics.mean(float(v['GPU use (%)']) for v in values),
                          fraction_samples_at_least_90_percent_busy=len(busy)/len(values),
                          busy_clock_median_mhz=statistics.median(clocks),busy_clock_min_mhz=min(clocks),busy_clock_max_mhz=max(clocks)))
    window_summary=[]
    for window in windows:
        subset=[r for r in rows if window['start']<=r['epoch']<=window['end']]
        window_summary.append(dict(**window,samples=len(subset),all_four_at_least_90_percent_fraction=statistics.mean(all(float(r['stats'][f'card{i}']['GPU use (%)'])>=90 for i in range(4)) for r in subset) if subset else None))
    result=dict(protocol='About one point sample every30s; monitoring began after the study start. Activity is not measured peak-FLOP utilization. Only samples within the requested exploration windows and before their training completion/deadline are included. The gap between the completed first study and the explicit two-hour extension is excluded; these samples do not establish uninterrupted utilization.',
                first_sample_utc=datetime.datetime.fromtimestamp(rows[0]['epoch'],datetime.timezone.utc).isoformat(),
                last_sample_utc=datetime.datetime.fromtimestamp(rows[-1]['epoch'],datetime.timezone.utc).isoformat(),
                samples=len(rows),all_four_at_least_90_percent_fraction=statistics.mean(all(float(r['stats'][f'card{i}']['GPU use (%)'])>=90 for i in range(4)) for r in rows),cards=cards,windows=window_summary)
    (a.root/'telemetry_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
