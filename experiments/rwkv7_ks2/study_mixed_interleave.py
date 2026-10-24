"""Suspend long workers in memory, run the requested short screen, then resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process(pid):
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    return dict(pid=pid, ppid=int(stat[1]), state=stat[0], start_ticks=stat[19], command=command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dispatcher-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.root
    record_path = root / 'mixed_rank_interleave.json'
    assert not record_path.exists()
    dispatcher = process(args.dispatcher_pid)
    assert 'study_wave_v18.py' in dispatcher['command']
    assert 'wave16_combination_long_manifest.json' in dispatcher['command']
    assert dispatcher['state'] not in ('T', 'Z')
    suspended = []
    record = dict(status='SUSPENDING', started_epoch=time.time(), processes=suspended,
                  reason='Late user-requested mixed-rank screen; preserve all long-run steps and in-memory model/optimizer state, with only one active training process per GPU.',
                  timing_limit='Elapsed training time for suspended long runs includes the pause and must not be interpreted as active step speed. Controlled final timing is separate.')
    def save():
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(record_path)
    def suspend(item):
        os.kill(item['pid'], signal.SIGSTOP)
        item['suspended_epoch'] = time.time()
        suspended.append(item)
        save()
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Stop dispatch first, so no new child can be launched during enumeration.
        suspend(dispatcher)
        for _ in range(100):
            if process(dispatcher['pid'])['state'] == 'T':
                break
            time.sleep(.01)
        assert process(dispatcher['pid'])['state'] == 'T'
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                item = process(int(path.name))
                if item['ppid'] == dispatcher['pid'] and item['state'] != 'Z':
                    suspend(item)
            except (FileNotFoundError, ProcessLookupError):
                continue
        time.sleep(2)  # Let already submitted device work drain.
        for item in suspended:
            assert process(item['pid'])['state'] in ('T', 'Z'), item
        record['status'] = 'SHORT_SCREEN_RUNNING'
        save()
        print(json.dumps(dict(event='long_workers_suspended', processes=suspended)), flush=True)
        output = root / 'wave17_mixed_ranks'
        if output.exists():
            assert not list(output.iterdir())
            output.rmdir()
        code = Path(__file__).resolve().parent
        with Path('/tmp/ks2_study_wave17.log').open('w') as log:
            subprocess.run([sys.executable, str(code / 'study_wave_v19.py'), '--manifest',
                            str(root / 'wave17_mixed_ranks_manifest.json'), '--requires-check',
                            str(root / 'compact_code_checks/general_geometry_check.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record['screen_finished_epoch'] = time.time()
        assert len(list(output.glob('*/completed.json'))) == 9
        assert not list(output.glob('*/failure.json'))
        record['status'] = 'SHORT_SCREEN_COMPLETE'
    finally:
        # Resume children before dispatch, checking PID identity to avoid reuse.
        for item in reversed(suspended):
            try:
                current = process(item['pid'])
                assert current['start_ticks'] == item['start_ticks'], item
                os.kill(item['pid'], signal.SIGCONT)
                item['resumed_epoch'] = time.time()
                item['pause_seconds'] = item['resumed_epoch'] - item['suspended_epoch']
            except (FileNotFoundError, ProcessLookupError):
                item['exited_before_resume'] = True
        record['resume_finished_epoch'] = time.time()
        record['status'] = 'COMPLETE_RESUMED' if record['status'] == 'SHORT_SCREEN_COMPLETE' else 'ERROR_RESUMED'
        save()
        print(json.dumps(dict(event='long_workers_resumed', record=record)), flush=True)


if __name__ == '__main__':
    main()
