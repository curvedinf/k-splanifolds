"""Wait for every declared run, close training, then perform final checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).resolve().parent
    plan=json.loads((root/'final_holdout_plan.json').read_text())
    print(json.dumps(dict(event='waiting_for_declared_training',manifests=plan['manifests'])),flush=True)
    while True:
        specs=[]
        for name in plan['manifests']:
            if not (root/name).exists():break
            spec=json.loads((root/name).read_text());out=Path(spec['out'])
            failures=list(out.glob('*/failure.json'))
            assert not failures,failures
            if not (out/'all_results.json').exists():break
            assert all((out/j['id']/'completed.json').exists() for j in spec['jobs'])
            specs.append(spec)
        if len(specs)==len(plan['manifests']):break
        time.sleep(5)
    record=dict(status='PASS',finished_epoch=time.time(),runs=sum(len(s['jobs']) for s in specs),manifests=plan['manifests'])
    (root/'training_complete.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(event='training_complete',**record)),flush=True)
    with (root/'final_checks_dispatch.jsonl').open('w') as log:
        subprocess.run([sys.executable,str(code/'hadmux_final_checks.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(event='final_checks_complete',epoch=time.time())),flush=True)


if __name__=='__main__':main()
