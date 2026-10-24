"""Run the same side-stream GPU protocol with separate attention/FFN history."""
import hashlib
import json
from pathlib import Path
import sys

import run_side_gpu
from side_history import HistoryModel


def main():
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    run_side_gpu.SideModel = HistoryModel
    run_side_gpu.main()
    path = out / 'results.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result['architecture'] += '; separate post-attention and post-FFN main residual history'
    result['arguments']['history_mode'] = 'previous_layers_post_attention_and_post_ffn'
    width = 96 + result['arguments']['side_width']
    widths = [width + 192 * i for i in range(4)]
    result['coding_protocol'].update(
        input_widths=widths, output_width=width,
        input_coverage='Every current main/side feature and every historical 96D feature enters every coordinate of every subspace',
        output='Unchanged baseline fixed output matrix with unit row norms and zero-init learned gains',
        note='History contains post-residual-addition main states from attention and FFN separately, '
             'in chronological order for all strictly previous layers. No detach or extra attention. '
             'LayerNorm and causal FFN time mix cover the expanded input; current-stream mix initialization '
             'is preserved. History is read-only; output updates only current main/side. '
             'Width-dependent input codes change; geometry/output initialization matches the baseline.')
    for name in ('side_history.py', 'run_side_history_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
