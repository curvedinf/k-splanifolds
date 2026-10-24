"""Four-GPU common-target context evaluation after context training finishes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from sweep_plateau import environment

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parent
 while not (a.root/'wave7_context/all_results.json').exists():time.sleep(5)
 paths=[p.parent for p in (a.root/'wave7_context').glob('*/completed.json')]
 paths += [a.root/'wave3/matched_init_mlp384',a.root/'wave4_activation/spine_fused_11',a.root/'wave4_activation/undamped_fused_11']
 queue=Queue()
 for path in paths:queue.put(path)
 def worker(gpu):
  rows=[]
  while True:
   try:path=queue.get_nowait()
   except Empty:return rows
   scores=[]
   for context in (64,256):
    dest=path/f'aligned_context{context}.json'
    with (path/f'aligned_context{context}.log').open('w') as log:subprocess.run([sys.executable,str(root/'study_aligned_context_eval.py'),'--run',str(path),'--context',str(context),'--output',str(dest)],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    scores.append(json.loads(dest.read_text()))
   assert scores[0]['targets']==scores[1]['targets'];row=dict(run=str(path.relative_to(a.root)),scores=scores);rows.append(row);print(json.dumps(row),flush=True)
 with ThreadPoolExecutor(max_workers=4) as pool:rows=[row for group in pool.map(worker,range(4)) for row in group]
 (a.root/'aligned_context_results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
