"""Create a GitHub-readable report package without training data or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root;out=a.out
    for name in ('audit.json','protocol_audit.json','source_bundle_check.json','final_checks/complete.json'):
        assert json.loads((root/name).read_text())['status']=='PASS',name
    assert json.loads((root/'study_state.json').read_text())['status']=='complete'
    assert not out.exists(),out
    out.mkdir(parents=True)
    copied=[]
    def copy(source,relative):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        copied.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for source in sorted(root.rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(root)
        if any(part in ('sources','evaluation_sources') for part in relative.parts):continue
        if source.suffix not in ('.json','.jsonl','.csv','.md','.svg','.png','.html'):continue
        copy(source,relative)
    copy(root/'source_bundle.tar.gz',Path('source_bundle.tar.gz'))
    plan=json.loads((root/'final_holdout_plan.json').read_text())
    for item in plan['prior_references']:
        copy(Path(item['run'])/'completed.json',Path('prior_context')/(item['label']+'.json'))
    for manifest in plan['manifests']:
        wave=Path(json.loads((root/manifest).read_text())['out']).name
        for gpu in range(4):
            copy(root/wave/f'validation_gpu{gpu}/check.log',Path(wave)/f'validation_gpu{gpu}/check.log')
    readme='''# Hadamard KS2 follow-up study

Read [the Markdown report](REPORT.md), which renders directly in GitHub.

The package includes every completed run record, three-seed comparisons, fresh reserved scores, timing rounds, numerical and protocol checks, figures, and a content-addressed archive of exact training sources. FFN learned parameter counts are separate from fixed matrix storage.

Training text, reserved text, model checkpoints, and optimizer states are intentionally excluded. Their hashes and local provenance paths remain in the evidence records. `source_bundle.tar.gz` contains an index mapping each executed source file to its SHA256-named blob; current analysis tooling is indexed separately. `prior_context/` preserves records for the earlier reference checkpoints scored on the new reserved interval.
'''
    (out/'README.md').write_text(readme)
    links=[]
    for source in out.rglob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',source.read_text()):
            if target.startswith(('https://','http://','#')):continue
            destination=(source.parent/target.split('#')[0]).resolve()
            assert destination.is_file(),(source,target)
            assert destination.is_relative_to(out.resolve()),target
            links.append(dict(source=source.name,target=target))
    assert not list(out.rglob('*.pt')) and not list(out.rglob('*.bytes'))
    (out/'package_check.json').write_text(json.dumps(dict(status='PASS',files=copied,validated_links=links,excluded='Training/heldout bytes, model and optimizer checkpoints; source code is in the content-addressed source archive.'),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',files=len(copied),bytes=sum(v['bytes'] for v in copied),links=len(links),out=str(out))))


if __name__=='__main__':main()
