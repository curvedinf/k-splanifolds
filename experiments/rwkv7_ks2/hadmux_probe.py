"""Training-input activation diagnostics, without touching reserved data."""
import argparse
import json
from pathlib import Path
import torch
from hadmux_models_v2 import build_model


def rms(x):
    return float(x.float().square().mean().sqrt()) if x.numel() else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wave', default='wave1_transfer')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    raw = Path(__file__).parent/'gpu_runs/data/train.bytes'
    values = torch.frombuffer(bytearray(raw.read_bytes()), dtype=torch.uint8).long()
    positions = torch.randint(len(values)-65, (4,), generator=torch.Generator().manual_seed(9471))
    tokens = values[positions[:, None]+torch.arange(64)]
    rows = []
    for path in sorted((args.root/args.wave).glob('*/completed.json')):
        record = json.loads(path.read_text())
        if record['arguments']['seed'] != 11 or record['study_config'].get('family')!='hadmux':
            continue
        cfg = record['study_config']
        for state in ('initialized', 'trained'):
            model = build_model(11, dict(cfg, fused_general=False)).eval()
            if state == 'trained':
                checkpoint = torch.load(path.parent/'checkpoint.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model'])
                del checkpoint
            layers = [dict(layer=i) for i in range(4)]
            handles = []
            for i, block in enumerate(model.blocks):
                def block_input(module, inputs, index=i):
                    layers[index].update(main_input_rms=rms(inputs[0]), side_input_rms=rms(inputs[1]))
                def projected(module, inputs, output, index=i, cell=block.ffn):
                    coords = output.float().reshape(-1,cell.num_subspaces,cell.subspace_rank)
                    geometry = cell.geo_step(coords)
                    updated = coords+geometry
                    layers[index].update(coordinate_rms=rms(coords), scalar_summary_rms=rms(coords.mean(-1)),
                                         geometry_rms=rms(geometry), geometry_to_coordinate_rms=rms(geometry)/max(rms(coords),1e-20),
                                         pre_activation_negative_fraction=float((updated<0).float().mean()),
                                         main_gain_rms=rms(cell.up.weight[:96]), side_gain_rms=rms(cell.up.weight[96:]))
                def ffn_output(module, inputs, output, index=i):
                    layers[index].update(main_update_rms=rms(output[..., :96]), side_update_rms=rms(output[..., 96:]))
                handles.extend([block.register_forward_pre_hook(block_input),
                                block.ffn.down.register_forward_hook(projected),
                                block.ffn.register_forward_hook(ffn_output)])
            with torch.no_grad():
                logits = model(tokens)
                assert torch.isfinite(logits).all()
            for handle in handles:
                handle.remove()
            rows.append(dict(model=cfg['name'],state=state,seed=11,steps=record['arguments']['steps'] if state=='trained' else 0,layers=layers))
            del model
    result = dict(protocol='Descriptive activations on the same4x64training bytes, seed11. No reserved validation. Eager CPU model with exactly equivalent fixed-code construction; not a causal intervention or a quality comparison.',rows=rows)
    (args.root/f'{args.wave}_activation_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(observations=len(rows),output=str(args.root/f'{args.wave}_activation_probe.json'))))


if __name__ == '__main__':
    main()
