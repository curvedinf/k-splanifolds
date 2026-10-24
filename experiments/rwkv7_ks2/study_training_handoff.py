"""Release final scoring only after the latest deadline and all manifest jobs.

The agent launches this after preparing the final long-run queue. It does not
train, select models, score data, or claim the overall research goal is complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--required-manifest', required=True)
    a = p.parse_args()
    assert (a.root / a.required_manifest).exists()
    marker = a.root / 'training_complete.json'
    assert not marker.exists()
    last_status = None
    while True:
        try:
            state = json.loads((a.root / 'study_state.json').read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        deadline = state['deadline_epoch']
        manifests, missing, failures, completed = [], [], [], []
        for path in sorted(a.root.glob('*_manifest.json')):
            raw = path.read_bytes()
            try:
                spec = json.loads(raw)
            except json.JSONDecodeError:
                missing.append(dict(manifest=path.name, job='manifest_write_in_progress'))
                continue
            if 'jobs' not in spec or 'out' not in spec:
                continue
            manifests.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), jobs=len(spec['jobs'])))
            for job in spec['jobs']:
                dest = Path(spec['out']) / job['id']
                result = dest / 'completed.json'
                if not result.exists():
                    result = dest / f"fixed_both_{job.get('seed', 11)}" / 'completed.json'
                if (dest / 'failure.json').exists():
                    failures.append(str(dest))
                if not result.exists():
                    missing.append(dict(manifest=path.name, job=job['id']))
                    continue
                try:
                    data = json.loads(result.read_text())
                except json.JSONDecodeError:
                    missing.append(dict(manifest=path.name, job=job['id']))
                    continue
                expected = job.get('steps', data['arguments']['steps'])
                assert data['history'][-1]['step'] == expected and data['invalid_steps'] == 0, result
                completed.append(result)
        status = (deadline, len(completed), len(missing), len(failures))
        if status != last_status:
            print(json.dumps(dict(event='training_handoff_wait', epoch=time.time(), deadline=deadline,
                                  completed=len(completed), pending=len(missing), failures=len(failures))), flush=True)
            last_status = status
        if time.time() >= deadline and not missing and not failures:
            # Re-read deadline immediately before writing; an explicit user
            # extension must not be bypassed by a previously cached value.
            if json.loads((a.root / 'study_state.json').read_text())['deadline_epoch'] != deadline:
                continue
            result = dict(status='TRAINING_COMPLETE', deadline_epoch=deadline, recorded_epoch=time.time(),
                          finished_epoch=max(path.stat().st_mtime for path in completed),
                          completed_jobs=len(completed), manifests=manifests,
                          note='Automatically released final scoring after latest deadline and every manifest job completed requested steps with zero invalid steps. Full checkpoint/source audit and report completion remain separate requirements.')
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            assert not marker.exists()
            temporary.replace(marker)
            print(json.dumps(dict(event='training_handoff_released', **result)), flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
