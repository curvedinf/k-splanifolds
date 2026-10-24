"""RHAC-style fixed signed Hadamard views, folded into two rank-8 charts.

Reference: ../SDGraft/RHAC/src/rhac/rhac.py, SpreadViews. Unlike that
history encoder, fold each full view to rank coordinates instead of
concatenating full-width views. Every input contributes to EVERY output
coordinate of EVERY chart. No history bank or stop-gradient is introduced.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def hadamard(rank):
    if rank < 1 or rank & (rank - 1):
        raise ValueError('rank must be a positive power of two')
    h = torch.ones(1, 1)
    while h.shape[0] < rank:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(rank)


def make_codes(width=96, charts=2, rank=8, seed=4242 + 99173):
    if width % rank:
        raise ValueError('width must be divisible by rank')
    gen = torch.Generator().manual_seed(seed)
    # Preserve RHAC's ordering: all permutations, then all sign draws.
    perms = torch.stack([torch.randperm(width, generator=gen) for _ in range(charts)])
    signs = torch.stack([torch.randint(0, 2, (width,), generator=gen,
                                     dtype=torch.int8).float() * 2 - 1
                         for _ in range(charts)])
    h = hadamard(rank)
    identity = torch.eye(width)
    codes = []
    for perm, sign in zip(perms, signs):
        mixed = (identity[:, perm].reshape(width, -1, rank) @ h.T).reshape(width, width)
        folded = (mixed * sign).reshape(width, -1, rank).sum(1) / math.sqrt(width // rank)
        codes.append(folded.T.contiguous())
    return torch.stack(codes), perms, signs


class FixedInput(nn.Module):
    def __init__(self, codes):
        super().__init__()
        self.register_buffer('weight', codes.flatten(0, 1).clone())

    def forward(self, x):
        return F.linear(x, self.weight)


class FixedOutput(nn.Module):
    """Transpose codes with unit row norms and zero-init per-output gains.

    Gains preserve the existing zero initial FFN residual. They do not learn
    a dense mixing matrix. `weight` names the gain for the existing optimizer.
    """
    def __init__(self, codes):
        super().__init__()
        matrix = codes.flatten(0, 1).T.contiguous()
        matrix = matrix / matrix.square().sum(1, keepdim=True).sqrt()
        self.register_buffer('matrix', matrix)
        self.weight = nn.Parameter(torch.zeros(matrix.shape[0]))

    def forward(self, x):
        return F.linear(x, self.matrix) * self.weight


def install(model, coding, view_seed=4242 + 99173):
    if coding not in ('learned', 'fixed_input', 'fixed_both'):
        raise ValueError(coding)
    if coding == 'learned':
        return model
    for block in model.blocks:
        if not hasattr(block, 'ffn'):
            continue
        cell = block.ffn
        codes, _, _ = make_codes(cell.down.in_features, cell.num_subspaces,
                                cell.subspace_rank, view_seed)
        cell.down = FixedInput(codes)
        if coding == 'fixed_both':
            cell.up = FixedOutput(codes)
    return model
