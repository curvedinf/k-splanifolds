"""Use the existing raw-mux protocol with the direct attention-state skip."""
import hashlib
import json
from pathlib import Path
import sys
import run_raw_mux_gpu
from raw_mux_skip import RawSkipModel


def main():
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    run_raw_mux_gpu.RawMuxModel = RawSkipModel
    run_raw_mux_gpu.main()
    path = out / 'results.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result['architecture'] += '; direct previous-attention-state addition before RWKV'
    result['arguments']['attention_skip'] = True
    result['coding_protocol']['attention_skip'] = (
        'Layers 2-4: combined = Hadamard_mux_96(history) + previous_post_attention_state; '
        'post_attention_state = combined + RWKV(LayerNorm(combined)). '
        'Layer 1 unchanged; coefficient 1; no detach; history still contains the previous state.')
    for name in ('raw_mux_skip.py', 'run_raw_mux_skip_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
