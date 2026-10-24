"""Resume suspended long workers as each mixed-screen GPU becomes free."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import time
from study_mixed_interleave import process
from study_wave_v19 import available_gpus


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    path=root/'mixed_rank_interleave.json'
    initial=json.loads(path.read_text())
    assert initial['status']=='SHORT_SCREEN_RUNNING'
    events=[json.loads(line) for line in Path('/tmp/ks2_study_wave16.log').read_text().splitlines() if line.startswith('{')]
    by_job={event['job']['id']:event['gpu'] for event in events if event.get('event')=='start'}
    workers={};dispatcher=None
    for item in initial['processes']:
        command=shlex.split(item['command'])
        if '--out' in command:
            job=Path(command[command.index('--out')+1]).name
            workers[by_job[job]]=item
        else:
            dispatcher=item
    assert set(workers)==set(range(4)) and dispatcher
    resumed={};dispatcher_resumed=False
    def resume(item,gpu):
        current=process(item['pid'])
        assert current['start_ticks']==item['start_ticks']
        os.kill(item['pid'],signal.SIGCONT)
        record=dict(pid=item['pid'],gpu=gpu,resumed_epoch=time.time(),start_ticks=item['start_ticks'])
        print(json.dumps(dict(event='resume_on_released_gpu',**record)),flush=True)
        return record
    while True:
        state=json.loads(path.read_text())
        if state['status'] in ('COMPLETE_RESUMED','ERROR_RESUMED'):
            break
        free=available_gpus(root/'wave17_mixed_ranks_manifest.json',Path('/tmp/ks2_study_wave17.log'))
        for gpu in sorted(free):
            item=workers[gpu]
            if item['pid'] in resumed:
                continue
            resumed[item['pid']]=resume(item,gpu)
            if not dispatcher_resumed:
                resumed[dispatcher['pid']]=resume(dispatcher,None)
                dispatcher_resumed=True
            (root/'mixed_rank_early_resume.json').write_text(json.dumps(list(resumed.values()),indent=2)+'\n')
        time.sleep(5)
    # The interleave wrapper's final SIGCONT is harmless for already resumed
    # processes. Preserve the actual earlier timestamps after its final write.
    for item in state['processes']:
        earlier=resumed.get(item['pid'])
        if earlier and earlier['resumed_epoch'] < item.get('resumed_epoch',float('inf')):
            item['wrapper_final_cont_epoch']=item.get('resumed_epoch')
            item['resumed_epoch']=earlier['resumed_epoch']
            item['pause_seconds']=item['resumed_epoch']-item['suspended_epoch']
            item['resumed_as_gpu_released']=True
            item['gpu']=earlier['gpu']
    state['per_gpu_resume_monitor_complete']=True
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(dict(event='resume_timestamps_reconciled',status=state['status'],early_resumes=len(resumed))),flush=True)


if __name__=='__main__':
    main()
