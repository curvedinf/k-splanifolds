"""Save/load GS32 KS3 exports with FP16 quantization scales and FP32 state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import run
from study_models_v19 import build_model
from ks3_gpu import chart_range, convert, inference_recurrence


def load_export(directory, rank=0, device='cuda', nonlinear='fp16', reducer=None):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())
    if (meta.get('format_version'), meta.get('quantization_group_size'),
            meta.get('scale_dtype')) != (2, 32, 'float16'):
        raise ValueError('This loader requires a GS32/FP16-scale v2 export; re-export the original checkpoint.')
    world = meta['tensor_parallel_size']
    if not 0 <= rank < world:
        raise ValueError('Invalid tensor-parallel rank')
    if world > 1:
        assert dist.is_initialized() and dist.get_world_size() == world
        assert dist.get_rank() == rank
    model = convert(build_model(meta['seed'], meta['study_config']), rank, world, nonlinear, reducer)
    record = meta['shards'][rank]
    path = directory / record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    run.recurrence = inference_recurrence
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.tp < 1:
        parser.error('--tp must be positive')
    torch.set_num_threads(2)
    metadata = json.loads((args.checkpoint_run / 'completed.json').read_text())
    seed, cfg = metadata['arguments']['seed'], metadata['study_config']
    checkpoint = args.checkpoint_run / 'checkpoint.pt'
    model = build_model(seed, cfg)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['model'])
    model = model.to(args.device)
    layouts = []
    for block in model.blocks:
        cell = block.ffn
        ranges = [chart_range(cell.num_subspaces, cell.subspace_rank, rank, args.tp)
                  for rank in range(args.tp)]
        layouts.append(dict(in_features=cell.down.weight.shape[1], out_features=cell.up.weight.shape[0],
                            num_subspaces=cell.num_subspaces, subspace_rank=cell.subspace_rank,
                            chart_ranges_by_rank=ranges,
                            subspaces_by_rank=[end-start for start, end in ranges],
                            features_by_rank=[(end-start)*cell.subspace_rank for start, end in ranges],
                            trained_parameters=sum(p.numel() for p in cell.parameters())))
    args.out.mkdir(parents=True, exist_ok=False)
    shards = []
    for rank in range(args.tp):
        exported = convert(model, rank, args.tp)
        state = {k: v.cpu() for k, v in exported.state_dict().items()}
        matrices = [(n, b) for n, b in state.items() if n.endswith('qweight')]
        scales = [(n, b) for n, b in state.items() if n.endswith(('.scales', '.spine_scale'))]
        assert matrices and all(v.dtype == torch.int8 for _, v in matrices)
        assert scales and all(v.dtype == torch.float16 for _, v in scales)
        name = f'rank_{rank}.pt'
        torch.save(state, args.out / name)
        shards.append(dict(rank=rank, file=name, sha256=hashlib.sha256((args.out/name).read_bytes()).hexdigest(),
                           bytes=(args.out/name).stat().st_size,
                           tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),
                           int8_elements=sum(v.numel() for v in state.values() if v.dtype == torch.int8),
                           float16_elements=sum(v.numel() for v in state.values() if v.dtype == torch.float16),
                           float32_elements=sum(v.numel() for v in state.values() if v.dtype == torch.float32),
                           linear_matrices=len(matrices),
                           ffn_tensor_bytes=sum(v.numel()*v.element_size() for n, v in state.items() if '.ffn.' in n)))
    heads = model.blocks[0].attention.heads
    sources = {name: Path(__file__).with_name(name) for name in
               ('ks3_gpu.py', 'ks3_attention_tp.py', 'export_ks3.py')}
    sources['experiments/common/ks3_int8.py'] = Path(__file__).resolve().parents[1]/'common/ks3_int8.py'
    result = dict(algorithm='KS3', format_version=2, quantization_group_size=32, scale_dtype='float16',
                  seed=seed, study_config=cfg, tensor_parallel_size=args.tp, ffn_layouts=layouts,
                  attention_heads_by_rank=[heads//args.tp+int(r<heads%args.tp) for r in range(args.tp)],
                  ffn_subspaces_by_rank=layouts[0]['subspaces_by_rank'],
                  trained_parameters_per_ffn=layouts[0]['trained_parameters'],
                  checkpoint_source=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  recurrent_state_dtype='float32', default_nonlinear='fp16', nonlinear_options=['fp32','fp16','int8'],
                  quantization='GS32 INT8 weights, embeddings, coefficients and activations; FP16 quantization scales; INT32 dots. Partial rows form partial GS32 groups.',
                  nonlinear_scope='KS3 geometry and ReLU-squared. Warp scalars, RWKV controller activations, norms, time mixing, residuals and recurrent state remain FP32.',
                  code_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}, shards=shards)
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', shards=len(shards), ffn_features_by_rank=layouts[0]['features_by_rank'],
                          tensor_bytes=sum(s['tensor_bytes'] for s in shards))))


if __name__ == '__main__':
    main()
