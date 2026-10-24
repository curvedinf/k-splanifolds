"""Four fixed-code KS2 FFNs with a side residual that bypasses attention.

The main stream, all RWKV modules, embedding and logits remain 96D.
Each FFN normalizes/time-mixes [main, side], then adds a full residual
update. Side starts at zero, stays differentiable between FFNs, and is
discarded after the final FFN. All four FFNs have 32 rank-8 subspaces.
"""
import torch
from torch import nn
from torch.nn import functional as F

import run
from experiments.common.ks2_stabilized import KS2StabilizedCell
from spread_projection import FixedInput, FixedOutput, make_codes

# Preserve the original constructor when runners replace run.Model.
BaseModel = run.Model


class SideBlock(nn.Module):
    def __init__(self, base, cell, side_width, layer, layers):
        super().__init__()
        self.main_width = 96
        self.side_width = side_width
        width = self.main_width + side_width
        self.attention = base.attention
        self.norm1 = base.norm1
        self.norm2 = nn.LayerNorm(width)
        mix = 1 - (torch.arange(width).float() / width).pow((1 - layer / layers) ** 4)
        # Match every main-stream learned initialization across side widths.
        mix[:self.main_width] = base.mix.detach()
        self.mix = nn.Parameter(mix)
        self.ffn = cell

    def forward(self, main, side, first_value):
        update, first_value = self.attention(self.norm1(main), first_value)
        main = main + update
        combined = torch.cat((main, side), dim=-1)
        hidden = self.norm2(combined)
        hidden = hidden + (F.pad(hidden[:, :-1], (0, 0, 1, 0)) - hidden) * self.mix
        combined = combined + self.ffn(hidden)
        return combined[..., :self.main_width], combined[..., self.main_width:], first_value


class SideModel(nn.Module):
    def __init__(self, seed, side_width=0):
        super().__init__()
        if side_width < 0 or side_width % 8:
            raise ValueError('side_width must be a nonnegative multiple of eight')
        base = BaseModel(seed, 'mlp', width=96, layers=4, fused_cpu=False)
        self.embedding, self.input_norm = base.embedding, base.input_norm
        self.output_norm, self.head = base.output_norm, base.head
        self.side_width = side_width
        width = 96 + side_width
        codes, _, _ = make_codes(width=width, charts=32, rank=8)
        blocks = []
        torch.manual_seed(seed + 2000)
        for i, block in enumerate(base.blocks):
            # Canonical 96D constructor consumes the same RNG draws for every
            # side width. Its dense projections are immediately discarded.
            cell = KS2StabilizedCell(96, 96, num_subspaces=32, subspace_rank=8, geo_steps=1)
            cell.down = FixedInput(codes)
            cell.up = FixedOutput(codes)
            blocks.append(SideBlock(block, cell, side_width, i, 4))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, tokens, initial_side=None):
        main = self.input_norm(self.embedding(tokens))
        side = main.new_zeros(*main.shape[:-1], self.side_width) if initial_side is None else initial_side
        first_value = None
        for block in self.blocks:
            main, side, first_value = block(main, side, first_value)
        return self.head(self.output_norm(main))


def parameter_breakdown(model):
    count = lambda m: sum(p.numel() for p in m.parameters())
    result = dict(total=count(model), rwkv=sum(count(b.attention) for b in model.blocks),
                  embedding_and_head=count(model.embedding) + count(model.head),
                  ks2_geometry=sum(p.numel() for b in model.blocks for n,p in b.ffn.named_parameters()
                                   if n.startswith('geo_')),
                  output_gains=sum(b.ffn.up.weight.numel() for b in model.blocks))
    result['norms_and_ffn_mix'] = result['total'] - sum(v for k,v in result.items() if k != 'total')
    return result
