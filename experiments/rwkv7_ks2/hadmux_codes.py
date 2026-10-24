"""Construct the same dense signed/folded Hadamard maps without identity GEMMs.

This only accelerates model construction. The training projection operators
and their storage remain identical to the original fixed dense-code model.
"""
import math
import torch
from spread_projection import hadamard


def make_codes(width=96, charts=2, rank=8, seed=4242+99173):
    if width % rank:
        raise ValueError('width must be divisible by rank')
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack([torch.randperm(width, generator=generator) for _ in range(charts)])
    signs = torch.stack([torch.randint(0, 2, (width,), generator=generator,
                                     dtype=torch.int8).float()*2-1 for _ in range(charts)])
    blocks = width // rank
    coefficients = hadamard(rank).repeat(1, blocks)[None]
    coefficients = coefficients * signs.reshape(charts, blocks, rank).transpose(1, 2).repeat_interleave(rank, dim=2)
    coefficients = coefficients / math.sqrt(blocks)
    codes = torch.empty(charts, rank, width)
    codes.scatter_(2, permutations[:, None, :].expand(-1, rank, -1), coefficients)
    return codes, permutations, signs
