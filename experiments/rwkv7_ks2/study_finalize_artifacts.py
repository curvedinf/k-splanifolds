"""Build and audit final artifacts after scoring; human analysis remains required.

This does not launch training, select candidates, or mark the research goal done.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'id' in values:
            assert values['id'] not in self.ids, values['id']
            self.ids.add(values['id'])
        if tag == 'a' and 'href' in values:
            self.links.append(values['href'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-runs', type=int, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root
    code = Path(__file__).resolve().parent
    finished = root / 'final_checks/complete.json'
    if args.wait:
        print(json.dumps(dict(event='waiting_for_final_scoring', epoch=time.time())), flush=True)
        while not finished.exists():
            time.sleep(10)
    assert finished.exists(), 'Final scoring must finish before artifact finalization'
    commands = []
    for script in ('study_collect.py', 'study_audit.py', 'study_protocol_audit.py',
                   'study_telemetry_summary.py', 'study_plot.py', 'study_report_figures.py',
                   'study_extended_figures.py', 'study_source_bundle.py', 'study_report.py'):
        command = [sys.executable, str(code / script), '--root', str(root)]
        print(json.dumps(dict(event='artifact_command', script=script, epoch=time.time())), flush=True)
        subprocess.run(command, check=True)
        commands.append(script)
        if script in ('study_audit.py', 'study_protocol_audit.py'):
            output = 'audit.json' if script == 'study_audit.py' else 'protocol_audit.json'
            assert json.loads((root / output).read_text())['status'] == 'PASS', output
    ledger = json.loads((root / 'all_runs.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    bundle = json.loads((root / 'source_bundle_check.json').read_text())
    report = json.loads((root / 'report_summary.json').read_text())
    assert len(ledger['rows']) == audit['completed'] == bundle['completed_runs'] == report['completed_runs'] == args.expected_runs
    assert bundle['status'] == 'PASS' and report['status'] == 'Final'
    parsed = Links()
    parsed.feed((root / 'report.html').read_text())
    for link in parsed.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (root / unquote(url.path)).exists(), link
        elif url.fragment:
            assert unquote(url.fragment) in parsed.ids, link
    result = dict(status='PASS', completed_runs=args.expected_runs, epoch=time.time(),
                  commands=commands, report_links=len(parsed.links), report_sections=len(parsed.ids),
                  remaining='Review final evidence and figures, update recommendations, and perform requirement-by-requirement goal completion audit. This artifact does not mark the research goal complete.')
    (root / 'artifact_finalization.json').write_text(json.dumps(result, indent=2) + '\n')
    (root / 'stop_telemetry').touch()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
