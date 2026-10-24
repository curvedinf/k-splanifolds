"""Export exact executed sources and study tooling without datasets/checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    code = Path(__file__).resolve().parent
    blobs = {}
    runs = []
    for path in sorted(root.rglob('completed.json')):
        result = json.loads(path.read_text())
        mapping = {}
        for name, expected in result['source_hashes'].items():
            source = path.parent / 'sources' / name.replace('../', 'parent_')
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == expected, (path, name)
            blobs[digest] = data
            mapping[name] = digest
        evaluators = {}
        for name, expected in result['tiled_evaluation'].get('evaluation_source_sha256', {}).items():
            data = (path.parent / 'evaluation_sources' / name).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == expected, (path, name)
            blobs[digest] = data
            evaluators[name] = digest
        runs.append(dict(run=str(path.parent.relative_to(root)),
                         training_sources=mapping, evaluation_sources=evaluators))
    tooling = {}
    # These are the current analysis tools, explicitly distinguished from
    # the immutable versions actually executed by each training run above.
    for path in sorted(code.glob('*.py')):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blobs[digest] = data
        tooling[path.name] = digest
    manifests = {}
    for path in sorted(root.glob('*_manifest.json')):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blobs[digest] = data
        manifests[path.name] = digest
    index = dict(created_epoch=time.time(), completed_runs=len(runs), runs=runs,
                 current_tooling=tooling, manifests=manifests,
                 format='Each mapping value is the SHA256 filename under blobs/. '
                        'Training mappings preserve exact per-run source versions. '
                        'Current tooling is separate and must not be substituted for those versions. '
                        'Data, trained weights, optimizer states and held-out bytes are not included.')
    index_bytes = (json.dumps(index, indent=2) + '\n').encode()
    archive = root / 'source_bundle.tar.gz'
    def add(handle, name, data):
        item = tarfile.TarInfo(name)
        item.size = len(data)
        item.mode = 0o644
        handle.addfile(item, io.BytesIO(data))
    with tarfile.open(archive, 'w:gz') as handle:
        add(handle, 'index.json', index_bytes)
        for digest, data in sorted(blobs.items()):
            add(handle, 'blobs/' + digest, data)
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle.getmembers():
            data = handle.extractfile(member).read()
            if member.name.startswith('blobs/'):
                assert hashlib.sha256(data).hexdigest() == member.name.split('/')[1]
        assert json.loads(handle.extractfile('index.json').read()) == index
    record = dict(status='PASS', completed_runs=len(runs), unique_source_blobs=len(blobs),
                  archive=archive.name, bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  created_epoch=index['created_epoch'])
    (root / 'source_bundle_check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
