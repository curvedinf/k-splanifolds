"""Raw KS2 coordinates in a depth history; mux before RWKV, final-only demux."""
import torch
from torch import nn
from torch.nn import functional as F

from side_stream import BaseModel
from experiments.common.ks2_stabilized import KS2StabilizedCell, _coordinates
from spread_projection import FixedInput, FixedOutput, make_codes


class RawCell(KS2StabilizedCell):
    def __init__(self, charts):
        # Same constructor draws as the 64-chart baseline; discard dense projections.
        super().__init__(96, 96, num_subspaces=charts, subspace_rank=8, geo_steps=1)
        del self.up

    def forward(self, x):
        coords = self.down(x).float().reshape(-1, self.num_subspaces, self.subspace_rank)
        for _ in range(self.geo_steps):
            coords = _coordinates(coords)
            coords = coords + self.geo_step(coords)
        return coords.reshape(*x.shape[:-1], -1)


class RawBlock(nn.Module):
    def __init__(self, base, cell, layer, charts):
        super().__init__()
        self.history_width = 96 + layer * (96 + charts * 8)
        self.ffn_input_width = self.history_width + 96
        self.attention, self.norm1 = base.attention, base.norm1
        # Twelve independent rank-8 views together provide a 96D RWKV input.
        attention_codes, _, _ = make_codes(self.history_width, 12, 8)
        self.attention_mux = FixedInput(attention_codes)
        codes, _, _ = make_codes(self.ffn_input_width, charts, 8)
        cell.down = FixedInput(codes)
        self.ffn = cell
        self.norm2 = nn.LayerNorm(self.ffn_input_width)
        mix = 1 - (torch.arange(self.ffn_input_width).float() / self.ffn_input_width).pow((1 - layer / 4) ** 4)
        self.mix = nn.Parameter(mix)

    def forward(self, history, first_value):
        view = self.attention_mux(history)
        update, first_value = self.attention(self.norm1(view), first_value)
        attention_state = view + update
        hidden = self.norm2(torch.cat((history, attention_state), dim=-1))
        hidden = hidden + (F.pad(hidden[:, :-1], (0, 0, 1, 0)) - hidden) * self.mix
        raw = self.ffn(hidden)
        return attention_state, raw, first_value


class RawMuxModel(nn.Module):
    def __init__(self, seed, charts=64):
        super().__init__()
        base = BaseModel(seed, 'mlp', width=96, layers=4, fused_cpu=False)
        self.embedding, self.input_norm = base.embedding, base.input_norm
        self.output_norm, self.head = base.output_norm, base.head
        torch.manual_seed(seed + 2000)
        self.blocks = nn.ModuleList(RawBlock(block, RawCell(charts), i, charts)
                                   for i, block in enumerate(base.blocks))
        codes, _, _ = make_codes(96, charts, 8)
        self.demux = FixedOutput(codes)
        # There is no bypass around the final demux; start with nonzero signal.
        nn.init.ones_(self.demux.weight)

    def forward(self, tokens):
        history = self.input_norm(self.embedding(tokens))
        first_value = None
        for block in self.blocks:
            attention_state, raw, first_value = block(history, first_value)
            history = torch.cat((history, attention_state, raw), dim=-1)
        return self.head(self.output_norm(self.demux(raw)))


def parameter_breakdown(model):
    count = lambda m: sum(p.numel() for p in m.parameters())
    result = dict(total=count(model), rwkv=sum(count(b.attention) for b in model.blocks),
                  embedding_and_head=count(model.embedding) + count(model.head),
                  ks2_geometry=sum(p.numel() for b in model.blocks for n,p in b.ffn.named_parameters()
                                   if n.startswith('geo_')), final_demux_gains=count(model.demux))
    result['norms_and_ffn_mix'] = result['total'] - sum(v for k,v in result.items() if k != 'total')
    return result


def make_optimizer(model, control_update_scale=False, ffn_charts=64):
    gain = model.demux.weight
    scale = 5.5 if control_update_scale else 1.
    groups = [dict(params=[p for p in model.parameters() if p is not gain],
                   lr_scale=1., weight_decay=.01, lr=.001),
              dict(params=[gain], lr_scale=scale, weight_decay=.01 / scale, lr=.001 * scale)]
    return torch.optim.AdamW(groups, lr=.001)
