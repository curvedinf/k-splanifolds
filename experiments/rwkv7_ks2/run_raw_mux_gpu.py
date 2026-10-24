"""Established GPU training/evaluation protocol for raw-coordinate history."""
import hashlib
import json
from pathlib import Path
import sys
import run
import run_spread_gpu
from raw_mux import RawMuxModel, make_optimizer, parameter_breakdown


def main():
    out = Path(sys.argv[sys.argv.index('--out') + 1])

    def factory(seed, variant, **kwargs):
        assert kwargs.get('width', 96) == 96 and kwargs.get('layers', 4) == 4
        return RawMuxModel(seed, charts=kwargs.get('ffn_charts', 64))

    run.Model = factory
    run.make_optimizer = make_optimizer
    run_spread_gpu.install = lambda model, coding: model
    sys.argv = [sys.argv[0], '--coding', 'fixed_both', *sys.argv[1:]]
    run_spread_gpu.main()
    path = out / 'results.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    cfg = result['arguments']
    charts = cfg['ffn_charts']
    result['architecture'] = 'Raw KS2 coordinate history; 96D Hadamard view before each RWKV; final-only 96D demux'
    result['arguments']['history_mode'] = 'embedding_then_each_post_attention_96_and_raw_ks2_coordinates'
    result['output_lr_multiplier'] = 5.5 if cfg['control_update_scale'] else 1.
    result['coding_protocol'] = dict(
        charts=charts, rank=8, attention_view_charts=12, attention_view_width=96,
        attention_input_widths=[96 + i*(96 + charts*8) for i in range(4)],
        ffn_input_widths=[192 + i*(96 + charts*8) for i in range(4)],
        raw_output_width=charts*8, final_output_width=96,
        view_seed=4242+99173, hadamard_width=8,
        input_coverage='Every history input enters every coordinate of every Hadamard view',
        output='No intermediate demux or gains; final-only fixed row-normalized 512-to-96 projection with unit-init learned gains',
        note='Embedding stays in history; append separate post-attention view+RWKV update and raw KS2 coordinates '
             'at each layer. KS2 reads history plus current post-attention state. All depth gradients retained. '
             'No 2048D side stream. Final head reads only final demux; no residual bypass. '
             'Initial logits differ from earlier residual models.')
    result['parameter_breakdown'] = parameter_breakdown(RawMuxModel(cfg['seed'], charts))
    for name in ('side_stream.py', 'raw_mux.py', 'run_raw_mux_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
