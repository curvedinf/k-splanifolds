"""Controlled CPU patch denoising; see README.md for citation and historical results.

python experiments/denoising/run.py --data-dir data/denoising --out runs/denoising
Dependencies: torch, numpy, Pillow. Defaults reproduce our original CPU protocol,
not the paper's full training procedure. KS1 adds a projected residual wrapper.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.ks2 import KS2Cell
from experiments.common.splanifold import splanifold_map

P, D, SIGMA = 13, 169, 25 / 255
KINDS = ('MLP', 'KS1', 'KS2')
REVISION = 'e93b27812d3ff523a3a79d19e5e50d233d7a8d0a'
DATA_DIGEST = '788c7ece1f859db24954968d341da994f7a6185437e6a3116b34fb4a3a4aa6a9'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dataset(root):
    groups = [('train400', 'TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400',
               [f'test_{i:03d}.png' for i in range(1, 401)]),
              ('set12', 'testsets/Set12', [f'{i:02d}.png' for i in range(1, 13)])]
    jobs = [(root / group / name, f'{group}/{name}',
             f'https://raw.githubusercontent.com/cszn/DnCNN/{REVISION}/{remote}/{name}')
            for group, remote, names in groups for name in names]

    def fetch(job):
        path, name, url = job
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(url, timeout=60) as response:
                data = response.read()
            tmp = path.with_suffix('.download')
            tmp.write_bytes(data)
            tmp.replace(path)
        return name, digest(path.read_bytes())

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = dict(sorted(pool.map(fetch, jobs)))
    actual = digest(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())
    if actual != DATA_DIGEST:
        raise ValueError(f'Dataset hash mismatch: {actual}; expected {DATA_DIGEST}')
    return manifest


class KS1Denoiser(nn.Module):
    def __init__(self, spaces):
        super().__init__()
        self.spaces = spaces
        self.down = nn.Linear(D, spaces * 8, bias=False)
        self.up = nn.Linear(spaces * 8, D, bias=False)
        for name in ('anchor_start', 'anchor_end'):
            setattr(self, name, nn.Parameter(torch.randn(spaces, 8) * .05))
        for name in ('basis_start', 'basis_end', 'pos_tangent_start', 'pos_tangent_end',
                     'basis_tangent_start', 'basis_tangent_end'):
            setattr(self, name, nn.Parameter(torch.randn(spaces, 8, 8) * .05))

    def forward(self, x):
        c = self.down(x).reshape(-1, self.spaces, 8)
        delta = splanifold_map(
            (1 + c.tanh()) * .5, self.anchor_start, self.anchor_end,
            self.basis_start, self.basis_end, self.pos_tangent_start,
            self.pos_tangent_end, self.basis_tangent_start, self.basis_tangent_end,
            sigma=1., extrapolation=0.)
        return self.up((c + delta).flatten(1))


def model(kind, seed, args):
    torch.manual_seed(seed)
    if kind == 'MLP':
        return nn.Sequential(nn.Linear(D, 511), nn.Tanh(), nn.Linear(511, 511),
                             nn.Tanh(), nn.Linear(511, D))
    if kind == 'KS1':
        return KS1Denoiser(args.ks1_spaces)
    return KS2Cell(D, D, num_subspaces=123, subspace_rank=8, geo_steps=2)


def read(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'), dtype=np.float32) / 255


def patches(paths, count, seed):
    rng = np.random.default_rng(seed)
    images = [read(p) for p in paths]
    result = np.empty((count, D), np.float32)
    for i in range(count):
        im = images[rng.integers(len(images))]
        y, x = rng.integers(im.shape[0] - P + 1), rng.integers(im.shape[1] - P + 1)
        patch = np.rot90(im[y:y + P, x:x + P], int(rng.integers(4)))
        result[i] = (patch[:, ::-1] if rng.integers(2) else patch).flatten()
    return torch.from_numpy(result)


@torch.no_grad()
def validate(m, vx, vy):
    m.eval()
    mse = sum((m(x) - y).square().sum().item()
              for x, y in zip(vx.split(128), vy.split(128))) / vy.numel() * .04
    m.train()
    if not math.isfinite(mse):
        raise RuntimeError('Nonfinite validation loss')
    return mse


def train(kind, seed, lr, steps, clean, vx, vy, args, tag):
    m = model(kind, seed, args)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 10000)
    best, state, elapsed, history = math.inf, None, 0., []
    for step in range(1, steps + 1):
        start = time.perf_counter()
        y = clean[torch.randint(len(clean), (64,), generator=generator)]
        x = (y + torch.randn(y.shape, generator=generator) * SIGMA - .5) / .2
        opt.zero_grad(set_to_none=True)
        loss = (m(x) - (y - .5) / .2).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss: {kind}, {seed}, {step}')
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        elapsed += time.perf_counter() - start
        if step == 1 or step % 250 == 0 or step == steps:
            val = validate(m, vx, vy)
            history.append(dict(step=step, train_mse_normalized=loss.item(),
                                train_mse_pixel=loss.item() * .04, val_mse_pixel=val))
            if val < best:
                best, best_step = val, step
                state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            print(f'{tag} {kind} seed={seed} lr={lr:g} step={step} '
                  f'train_MSE={loss.item() * .04:.7f} val_MSE={val:.7f}', flush=True)
    record = dict(kind=kind, seed=seed, lr=lr, steps=steps, best_step=best_step,
                  val_mse_pixel=best, train_seconds=elapsed,
                  parameters=sum(p.numel() for p in m.parameters()))
    save(args.out / f'{tag}_{kind}_{seed}_{lr:g}.json', dict(record=record, history=history))
    if tag == 'main':
        torch.save(state, args.out / f'{kind}_{seed}.pt')
    return record


@torch.no_grad()
def denoise(m, noisy):
    h, w = noisy.shape
    x = F.pad(torch.from_numpy(noisy)[None, None],
              (6, 6 + (-w + 1) % 3, 6, 6 + (-h + 1) % 3), mode='reflect')
    shape = x.shape[-2:]
    cols = F.unfold(x, P, stride=3)[0].T.contiguous()
    start = time.perf_counter()
    pred = torch.cat([m((c - .5) / .2) * .2 + .5 for c in cols.split(128)])
    seconds = time.perf_counter() - start
    pred = pred.T[None]
    image = F.fold(pred, shape, P, stride=3) / F.fold(torch.ones_like(pred), shape, P, stride=3)
    return image[0, 0, 6:6 + h, 6:6 + w].clamp(0, 1).numpy(), seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/denoising'))
    parser.add_argument('--out', type=Path, default=Path('runs/denoising'))
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--pilot-steps', type=int, default=400)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--lrs', nargs='+', type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument('--train-patches', type=int, default=120000)
    parser.add_argument('--val-patches', type=int, default=4096)
    parser.add_argument('--ks1-spaces', type=int, default=140)
    args = parser.parse_args()
    if min(args.steps, args.pilot_steps, args.threads, args.train_patches,
           args.val_patches, args.ks1_spaces) < 1 or any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        parser.error('Budgets, threads, spaces and learning rates must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = dataset(args.data_dir)
    save(args.out / 'data_manifest.json', manifest)
    paths = sorted((args.data_dir / 'train400').glob('test_*.png'))
    if len(paths) != 400:
        raise ValueError('Expected exactly 400 training images')
    order = np.random.default_rng(20261022).permutation(400)
    tr, va = [paths[i] for i in order[:360]], [paths[i] for i in order[360:]]
    test = [args.data_dir / 'set12' / f'{i:02d}.png' for i in range(1, 13)]
    save(args.out / 'splits.json', {k: [str(p.relative_to(args.data_dir)) for p in ps]
                                  for k, ps in [('train', tr), ('validation', va), ('test', test)]})
    sources = [Path(__file__), Path(sys.modules[KS2Cell.__module__].__file__),
               Path(sys.modules[splanifold_map.__module__].__file__)]
    save(args.out / 'config.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
        platform=platform.platform(), dataset_revision=REVISION, dataset_sha256=DATA_DIGEST,
        source_sha256={p.name: digest(p.read_bytes()) for p in sources},
        patch_size=P, sigma=SIGMA, batch_size=64, normalization='(pixel-.5)/.2',
        split_seed=20261022, patch_seeds=[321, 456], validation_noise_seed=789,
        test_noise_seed='12345 + zero-based image index', optimizer='Adam; clip norm 1',
        reconstruction='reflect pad; stride 3; uniform average; output clip [0,1]',
        inference_timing='eager; normalization/model/concat only; excludes unfold/fold'))
    clean, v = patches(tr, args.train_patches, 321), patches(va, args.val_patches, 456)
    vx = (v + torch.randn(v.shape, generator=torch.Generator().manual_seed(789)) * SIGMA - .5) / .2
    vy = (v - .5) / .2
    pilots, chosen, records = [], {}, []
    for kind in KINDS:
        runs = [train(kind, 777, lr, args.pilot_steps, clean, vx, vy, args, 'pilot') for lr in args.lrs]
        pilots.extend(runs)
        chosen[kind] = min(runs, key=lambda r: r['val_mse_pixel'])['lr']
    save(args.out / 'pilot.json', dict(runs=pilots, chosen=chosen))
    for seed in args.seeds:
        for kind in KINDS:
            records.append(train(kind, seed, chosen[kind], args.steps, clean, vx, vy, args, 'main'))
            save(args.out / 'training.json', records)
    # Test images are evaluated only after every model and checkpoint is selected.
    rows = []
    for seed in args.seeds:
        for kind in KINDS:
            m = model(kind, seed, args)
            m.load_state_dict(torch.load(args.out / f'{kind}_{seed}.pt', weights_only=True))
            m.eval()
            for i, path in enumerate(test):
                target = read(path)
                noise = np.random.default_rng(12345 + i).normal(0, SIGMA, target.shape).astype(np.float32)
                pred, seconds = denoise(m, target + noise)
                mse = float(np.mean((pred - target) ** 2))
                rows.append(dict(kind=kind, seed=seed, image=path.name, mse_pixel=mse,
                                 psnr=-10 * math.log10(mse), inference_seconds=seconds,
                                 noisy_mse_pixel=float(np.mean(noise ** 2))))
    with (args.out / 'test_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for kind in KINDS:
        selected = [r for r in rows if r['kind'] == kind]
        seed_psnr = [float(np.mean([r['psnr'] for r in selected if r['seed'] == seed])) for seed in args.seeds]
        summary[kind] = dict(parameters=next(r['parameters'] for r in records if r['kind'] == kind),
                             test_mse_pixel=float(np.mean([r['mse_pixel'] for r in selected])),
                             mean_image_psnr=float(np.mean(seed_psnr)), seed_mean_psnr=seed_psnr,
                             psnr_sample_sd=float(np.std(seed_psnr, ddof=1)) if len(seed_psnr) > 1 else None)
        print(f"{kind:3s} MSE={summary[kind]['test_mse_pixel']:.7f} PSNR={np.mean(seed_psnr):.4f}")
    save(args.out / 'summary.json', summary)


if __name__ == '__main__':
    main()
