"""Low-rate read-only four-GPU clock/load log for interpreting training timings."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
while time.time()<json.loads((a.root/'study_state.json').read_text())['deadline_epoch']+3600 and not (a.root/'stop_telemetry').exists():
 r=subprocess.run(['/opt/rocm/bin/rocm-smi','--showtemp','--showclocks','--showpower','--showuse','--json'],capture_output=True,text=True)
 try:stats=json.loads(r.stdout)
 except ValueError:stats=dict(output=r.stdout,error=r.stderr)
 with (a.root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),returncode=r.returncode,stats=stats))+'\n')
 time.sleep(30)
