"""Small FP32 RWKV-7 controller ablation; CPU, byte-level FineWeb."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.ks2_stabilized import KS2StabilizedCell
from experiments.common.ks2_cpu.training import fuse_training


@torch.jit.script
def recurrence(read, decay, key, value, normalized_key, rate):
    batch, length, heads, width = read.shape
    state = torch.zeros(batch, heads, width, width, dtype=read.dtype, device=read.device)
    outputs = []
    for pos in range(length):
        removal = torch.matmul(state, normalized_key[:, pos].unsqueeze(-1))
        state = state * decay[:, pos].unsqueeze(-2) - removal * (normalized_key[:, pos] * rate[:, pos]).unsqueeze(-2)
        state = state + value[:, pos].unsqueeze(-1) * key[:, pos].unsqueeze(-2)
        outputs.append(torch.matmul(state, read[:, pos].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, 1)


class Controller(nn.Module):
    def __init__(self, width, rank, activation):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.activation = activation
        nn.init.zeros_(self.down.weight)
        nn.init.orthogonal_(self.up.weight, gain=.1)

    def forward(self, inputs):
        hidden = self.down(inputs)
        if self.activation == 'tanh': hidden = hidden.tanh()
        if self.activation == 'sigmoid': hidden = hidden.sigmoid()
        return self.up(hidden)


class TimeMix(nn.Module):
    def __init__(self, width, layer, layers):
        super().__init__()
        self.width, self.heads = width, width // 32
        fraction = torch.arange(width).float() / width
        depth = 1 - layer / layers
        for name, power in zip(('read', 'decay', 'key', 'value', 'rate', 'gate'), (.2, .9, .7, .7, .9, .2)):
            self.register_parameter('mix_' + name, nn.Parameter(1 - fraction.pow(power * depth)))
        zigzag = (torch.arange(width) % 32 - 15.5) / 15.5
        zigzag = zigzag * zigzag.abs()
        linear = torch.linspace(-.5, .5, width)
        self.decay_bias = nn.Parameter(-6 + 6 * torch.linspace(0, 1, width).pow(1 + (layer / (layers - 1)) ** .3) + .5 + zigzag * 2.5)
        self.rate_bias = nn.Parameter(-.19 + zigzag * .3 + linear * .4)
        self.value_bias = nn.Parameter(.73 - linear * .4)
        self.key_scale = nn.Parameter(.71 - linear * .1)
        self.key_rate = nn.Parameter(torch.full((width,), 1.02))
        self.bonus = nn.Parameter(torch.full((self.heads, 32), -.04))
        self.decay_control = Controller(width, 32, 'tanh')
        self.rate_control = Controller(width, 32, 'linear')
        self.gate_control = Controller(width, 64, 'sigmoid')
        self.value_control = Controller(width, 32, 'linear') if layer else None
        for name, scale in [('read', .5), ('key', .05), ('value', .5)]:
            module = nn.Linear(width, width, bias=False)
            nn.init.uniform_(module.weight, -scale / math.sqrt(width), scale / math.sqrt(width))
            setattr(self, name, module)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.norm = nn.GroupNorm(self.heads, width, eps=64e-5)

    def forward(self, inputs, first_value):
        batch, length, width = inputs.shape
        shift = F.pad(inputs[:, :-1], (0, 0, 1, 0)) - inputs
        mixed = {name: inputs + shift * getattr(self, 'mix_' + name) for name in ('read', 'decay', 'key', 'value', 'rate', 'gate')}
        read = self.read(mixed['read'])
        decay = torch.exp(-torch.sigmoid(self.decay_bias + self.decay_control(mixed['decay'])) * math.exp(-.5))
        rate = torch.sigmoid(self.rate_bias + self.rate_control(mixed['rate']))
        key, value = self.key(mixed['key']), self.value(mixed['value'])
        if first_value is None: first_value = value
        else: value = value + (first_value - value) * torch.sigmoid(self.value_bias + self.value_control(mixed['value']))
        shape = (batch, length, self.heads, 32)
        normalized_key = F.normalize((key * self.key_scale).reshape(shape), dim=-1)
        key = key * (1 + (rate - 1) * self.key_rate)
        read, decay, key, value, rate = [item.reshape(shape) for item in (read, decay, key, value, rate)]
        output = recurrence(read, decay, key, value, normalized_key, rate)
        output = self.norm(output.reshape(batch * length, width)).reshape(batch, length, width)
        output = output + ((read * key * self.bonus).sum(-1, keepdim=True) * value).reshape(batch, length, width)
        return self.output(output * self.gate_control(mixed['gate'])), first_value


class Block(nn.Module):
    def __init__(self, width, layer, layers):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = TimeMix(width, layer, layers)
        self.mix = nn.Parameter(1 - (torch.arange(width).float() / width).pow((1 - layer / layers) ** 4))
        self.key = nn.Linear(width, 4 * width, bias=False)
        self.value = nn.Linear(4 * width, width, bias=False)
        nn.init.uniform_(self.key.weight, -.5 / math.sqrt(width), .5 / math.sqrt(width))
        nn.init.zeros_(self.value.weight)

    def forward(self, inputs, first_value):
        update, first_value = self.attention(self.norm1(inputs), first_value)
        inputs = inputs + update
        hidden = self.norm2(inputs)
        hidden = hidden + (F.pad(hidden[:, :-1], (0, 0, 1, 0)) - hidden) * self.mix
        update = self.ffn(hidden) if hasattr(self, 'ffn') else self.value(F.relu(self.key(hidden)).square())
        return inputs + update, first_value


class Model(nn.Module):
    def __init__(self, seed, variant, charts=1, width=128, layers=2, ffn_charts=11):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(256, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, layer, layers) for layer in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)
        nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)
        if variant in ('ks2', 'ks2_ffn'):
            torch.manual_seed(seed + 1000)
            for block in self.blocks:
                for name in ('decay_control', 'rate_control'):
                    cell = KS2StabilizedCell(width, width, num_subspaces=charts, geo_steps=1)
                    # Zero output preserves the original initial control signals.
                    nn.init.zeros_(cell.up.weight)
                    setattr(block.attention, name, fuse_training(cell, threads=1))
        if variant == 'ks2_ffn':
            torch.manual_seed(seed + 2000)
            for block in self.blocks:
                cell = KS2StabilizedCell(width, width, num_subspaces=ffn_charts, geo_steps=1)
                nn.init.zeros_(cell.up.weight)
                block.ffn = fuse_training(cell, threads=1)
                del block.key, block.value

    def forward(self, tokens):
        hidden, first_value = self.input_norm(self.embedding(tokens)), None
        for block in self.blocks: hidden, first_value = block(hidden, first_value)
        return self.head(self.output_norm(hidden))


def prepare_data(root, out):
    if (root / 'data_manifest.json').exists():
        manifest = json.loads((root / 'data_manifest.json').read_text())
        arrays = []
        for split in ('train', 'val'):
            raw = (root / (split + '.bytes')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == manifest[split]['sha256']
            arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).long())
        return *arrays, manifest
    import tiktoken
    encoding = tiktoken.get_encoding('gpt2')
    result, manifest = [], {}
    for split, filename, count in [('train', 'fineweb_train_000001.bin', 1000000), ('val', 'fineweb_val_000000.bin', 100000)]:
        source = root / filename
        tokens = np.memmap(source, dtype='<u2', mode='r', offset=1024)[:count]
        raw = b''.join(encoding.decode_single_token_bytes(int(token)) for token in tokens)
        (out / (split + '.bytes')).write_bytes(raw)
        result.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).long())
        manifest[split] = dict(source=str(source), source_tokens=count, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return *result, manifest


def make_optimizer(model, control_update_scale=False, ffn_charts=11):
    scale = 11 / ffn_charts if control_update_scale else 1.
    if control_update_scale:
        outputs = [block.ffn.up.weight for block in model.blocks]
        output_ids = {id(p) for p in outputs}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in output_ids], lr_scale=1., weight_decay=.01),
                  dict(params=outputs, lr_scale=scale, weight_decay=.01 / scale)]
    else:
        groups = [dict(params=list(model.parameters()), lr_scale=1., weight_decay=.01)]
    optimizer = torch.optim.AdamW(groups, lr=.001)
    for group in optimizer.param_groups: group['lr'] = .001 * group['lr_scale']
    return optimizer


def set_learning_rate(optimizer, step, steps):
    rate = .001 * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / steps)))
    for group in optimizer.param_groups: group['lr'] = rate * group['lr_scale']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['mlp', 'ks2', 'ks2_ffn'], required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--charts', type=int, default=1)
    parser.add_argument('--ffn-charts', type=int, default=11)
    parser.add_argument('--control-update-scale', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.control_update_scale and args.variant != 'ks2_ffn': parser.error('Update scaling requires ks2_ffn')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    train, validation, manifest = prepare_data(args.data, args.out)
    model = Model(args.seed, args.variant, args.charts, ffn_charts=args.ffn_charts)
    optimizer = make_optimizer(model, args.control_update_scale, args.ffn_charts)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    positions = torch.randint(len(train) - 65, (args.steps, 4), generator=generator)
    val_positions = torch.randint(len(validation) - 65, (32, 4), generator=torch.Generator().manual_seed(12345))
    offsets = torch.arange(65)
    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for starts in val_positions:
                batch = validation[starts[:, None] + offsets]
                losses.append(F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten()).item())
        model.train()
        return float(np.mean(losses))
    history = [dict(step=0, val_loss=evaluate())]
    training_seconds = 0.
    for step, starts in enumerate(positions, 1):
        start = time.perf_counter()
        batch = train[starts[:, None] + offsets]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        set_learning_rate(optimizer, step, args.steps)
        optimizer.step()
        training_seconds += time.perf_counter() - start
        if step % 100 == 0 or step == args.steps:
            entry = dict(step=step, train_loss=loss.item(), val_loss=evaluate(), training_seconds=training_seconds, gradient_norm=float(gradient))
            history.append(entry)
            print(json.dumps(entry), flush=True)
    model.eval()
    timings = []
    with torch.no_grad():
        for iteration in range(25):
            start = time.perf_counter()
            model(batch[:, :-1])
            if iteration >= 5: timings.append(time.perf_counter() - start)
    controls = sum(p.numel() for block in model.blocks for name in ('decay_control', 'rate_control') for p in getattr(block.attention, name).parameters())
    ffn_parameters = sum(p.numel() for block in model.blocks for module in ([block.ffn] if hasattr(block, 'ffn') else [block.key, block.value]) for p in module.parameters())
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, parameters=sum(p.numel() for p in model.parameters()), controller_parameters=controls, ffn_parameters=ffn_parameters, history=history, training_seconds=training_seconds, inference_ms=float(np.median(timings) * 1000), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, data=manifest, torch=torch.__version__)
    name = args.variant + '_' + str(args.seed)
    (args.out / (name + '.json')).write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), args.out / (name + '.pt'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('history', 'data')}), flush=True)


if __name__ == '__main__': main()
