"""Descriptive chart-input statistics on identical training windows, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import run  # Sets repository import path and retains the CPU recurrence.
from study_models_v15 import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    jobs = []
    for seed in (11, 22, 33):
        for name in ('spine_rank1', 'spine_rank2', 'spine_rank4', 'spine_eager_reference', 'spine_rank16'):
            jobs.append(f'wave11_conditioning/{name}_{seed}')
        for name in ('spine_rank1_frozen', 'spine_rank4_frozen'):
            jobs.append(f'wave12b_rank_freezing/{name}_{seed}')
    data_path = Path(__file__).parent / 'gpu_runs/data/train.bytes'
    raw = data_path.read_bytes()
    train = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).long()
    starts = torch.randint(len(train) - 65, (4, 4), generator=torch.Generator().manual_seed(682143))
    batches = train[starts[:, :, None] + torch.arange(65)]
    rows = []
    destination = args.root / 'rank_input_statistics.json'
    for job in jobs:
        path = args.root / job
        result = json.loads((path / 'completed.json').read_text())
        assert result['data']['train']['sha256'] == hashlib.sha256(raw).hexdigest()
        cfg = dict(result['study_config'], fused_geometry=False, fused_partial=False, fused_general=False)
        for state in ('initialized', 'trained_15k'):
            model = build_model(result['arguments']['seed'], cfg).eval()
            if state == 'trained_15k':
                checkpoint = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model'])
                del checkpoint
            saved = [[] for _ in model.blocks]
            for index, block in enumerate(model.blocks):
                original = block.ffn.geo_step
                def capture(x, index=index, original=original):
                    geometry = original(x)
                    saved[index].append((x.detach(), geometry.detach()))
                    return geometry
                block.ffn.geo_step = capture
            with torch.no_grad():
                for batch in batches:
                    model(batch[:, :-1])
            layers = []
            for index, block in enumerate(model.blocks):
                x = torch.cat([pair[0] for pair in saved[index]]).double()
                geometry = torch.cat([pair[1] for pair in saved[index]]).double()
                mean = x.mean(-1, keepdim=True)
                t = 4 * torch.tanh(mean / 4)
                coordinate_rms = x.square().mean().sqrt()
                basis = block.ffn.spine_basis(t.float()).double()
                gram = torch.einsum('nsm,nsj->smj', basis, basis) / len(basis)
                eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
                condition = (eigen[:, -1] / eigen[:, 0].clamp_min(1e-20)).sqrt()
                centered_t = t[..., 0] - t[..., 0].mean(0, keepdim=True)
                covariance = centered_t.T @ centered_t / len(centered_t)
                participation = covariance.trace().square() / covariance.square().sum().clamp_min(1e-30)
                rank = x.shape[-1]
                within_correlation = None
                if rank > 1:
                    centered = x - x.mean(0, keepdim=True)
                    standardized = centered / centered.square().mean(0, keepdim=True).clamp_min(1e-30).sqrt()
                    correlations = torch.einsum('nsr,nsk->srk', standardized, standardized) / len(x)
                    diagonal = correlations.diagonal(dim1=-2, dim2=-1).sum()
                    within_correlation = float((correlations.sum() - diagonal) / (x.shape[1] * rank * (rank - 1)))
                layers.append(dict(layer=index, rank=rank, charts=x.shape[1],
                                   coordinate_rms=float(coordinate_rms), mean_rms=float(mean.square().mean().sqrt()),
                                   basis_argument_rms=float(t.square().mean().sqrt()),
                                   basis_argument_to_coordinate_rms=float(t.square().mean().sqrt() / coordinate_rms),
                                   within_chart_mean_correlation=within_correlation,
                                   summary_covariance_participation_ratio=float(participation),
                                   geometry_rms=float(geometry.square().mean().sqrt()),
                                   geometry_to_coordinate_rms=float(geometry.square().mean().sqrt() / coordinate_rms),
                                   relu_active_fraction=float(((x + geometry) > 0).double().mean()),
                                   basis_design_condition_median=float(condition.median())))
            rows.append(dict(job=job, seed=result['arguments']['seed'], model=cfg['name'], state=state, layers=layers))
            payload = dict(protocol='CPU eager forward on the same1024 training targets in16windows, B4T64; position seed682143. '
                           'Initialized and trained15k checkpoints, all3seeds. No validation or reserved targets. '
                           'Statistics describe each model at its own learned activations; they do not isolate a causal effect. '
                           'Covariance participation ratio is a scale-sensitive effective-dimension summary, not algebraic matrix rank.',
                           train_sha256=hashlib.sha256(raw).hexdigest(), starts=starts.tolist(),
                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
            destination.write_text(json.dumps(payload, indent=2) + '\n')
            print(json.dumps(dict(job=job, state=state, t_rms=[round(row['basis_argument_rms'], 4) for row in layers])), flush=True)
    print(json.dumps(dict(status='PASS', model_states=len(rows), artifact=str(destination))), flush=True)


if __name__ == '__main__':
    main()
