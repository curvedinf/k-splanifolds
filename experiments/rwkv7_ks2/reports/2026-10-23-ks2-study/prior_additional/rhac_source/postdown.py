"""Post-Down Dense Residual Conditioner.

Implements the post-down design from the architecture specification
verbatim: prior completed decoder-block states (same token, depth history,
detached) -> destination-specific balanced signed sketch (l*D -> l*C,
every scalar in exactly one bucket, fixed random sign, 1/sqrt(count)
normalization, seed 1234+l) -> one zero-initialized dense [D, l*C] matrix
-> added after down_proj alongside the ordinary residual commit:

    x_next = r + down_proj(silu(g)*u) + W_cond @ sketch(history)

The pretrained MLP is untouched; the residual conditioner is a parallel
residual branch. Layer 0 has no conditioner. No Splanifold, no gate
conditioning, no bias, no normalization on the correction. The completed
corrected state is what enters the history bank for later layers (depth
recurrence).

Integration mirrors the RHAC tuner pattern: each decoder layer's MLP is
wrapped so its output becomes y + c, and a per-layer forward hook appends
the completed block output to a shared per-forward bank (reset at the text
model's forward pre-hook).

Terminology (user direction): the learned module is the
**residual conditioner** — not "historical receiver" — everywhere
(class names, checkpoint keys, config keys, docs).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

ROUTE_SEED_BASE = 1234


def build_route(n_in: int, n_out: int, seed: int):
    """Balanced signed bucket assignment over n_in scalars -> n_out buckets.

    Each scalar lands in exactly one bucket; bucket counts differ by at
    most one; every scalar gets a fixed +1/-1 sign. Returns
    (index [n_in], sign [n_in], inv_sqrt_counts [n_out]).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_in, generator=g)
    counts = torch.full((n_out,), n_in // n_out, dtype=torch.long)
    counts[: n_in % n_out] += 1
    bucket_of = torch.repeat_interleave(torch.arange(n_out), counts)
    index = torch.empty(n_in, dtype=torch.long)
    index[perm] = bucket_of
    sign = torch.randint(0, 2, (n_in,), generator=g) * 2 - 1
    inv_sqrt_counts = 1.0 / torch.sqrt(counts.float())
    return index, sign.to(torch.float32), inv_sqrt_counts


class ResidualConditioner(nn.Module):
    """Destination-layer residual conditioner (spec §14, §22).

    W_cond [D, l*C], zero-initialized, no bias; input is the signed
    sketched depth history of completed decoder states.
    """

    def __init__(self, dest_layer: int, d_model: int, channels_per_layer: int,
                 seed_base: int = ROUTE_SEED_BASE, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.l = dest_layer
        self.D = d_model
        self.C = channels_per_layer
        self.n_in = dest_layer * d_model
        self.n_out = dest_layer * channels_per_layer
        idx, sign, inv = build_route(self.n_in, self.n_out, seed_base + dest_layer)
        self.register_buffer("route_index", idx, persistent=True)
        self.register_buffer("route_sign", sign, persistent=True)
        self.register_buffer("route_inv_sqrt_counts", inv, persistent=True)
        self.weight = nn.Parameter(torch.zeros(d_model, self.n_out, dtype=dtype))
        self._src_cache = None  # (idx_slices, sign_slices), rebuilt if buffers move

    def _src_slices(self):
        """Per-source (bucket, sign) slices derived from the live route
        buffers, so checkpoint loading can never desynchronize them."""
        if self._src_cache is None or self._src_cache[0][0].data_ptr() != self.route_index.data_ptr():
            self._src_cache = (
                [self.route_index[j * self.D : (j + 1) * self.D] for j in range(self.l)],
                [self.route_sign[j * self.D : (j + 1) * self.D] for j in range(self.l)],
            )
        return self._src_cache

    def forward(self, history_bank: list[torch.Tensor]) -> torch.Tensor:
        """history_bank: l tensors of (B,T,D) completed states -> (B,T,D).

        Incremental per-source sketch: each source layer's D channels
        index_add directly into the (B,T,l*C) accumulator, so the
        (B,T,l*D) concatenation is never materialized (at T=2000/depth 63
        that cat is ~1 GB per layer per forward — the prefill bottleneck;
        it also previously drove the caching allocator into a retry spiral
        at fp32). Same math: flat position j*D+c is source j channel c;
        its bucket/sign come from the destination route slices.
        """
        B, T, _ = history_bank[0].shape
        z = torch.zeros(B, T, self.n_out, dtype=torch.float32,
                        device=history_bank[0].device)
        src_idx, src_sign = self._src_slices()
        for j, x in enumerate(history_bank):
            z.index_add_(-1, src_idx[j], x.detach().float() * src_sign[j])
        z = z * self.route_inv_sqrt_counts
        out = F.linear(z.to(self.weight.dtype), self.weight)
        return out.to(history_bank[0].dtype)  # stay in the residual-stream dtype


class ConditionedMLP(nn.Module):
    """Wraps a decoder MLP: out = base_mlp(x) + residual_conditioner(history)."""

    def __init__(self, base_mlp: nn.Module,
                 residual_conditioner: Optional[ResidualConditioner],
                 ctx: "ConditionerContext", layer_idx: int):
        super().__init__()
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        self.act_fn = getattr(base_mlp, "act_fn", None) or F.silu
        self.residual_conditioner = residual_conditioner
        self.ctx = ctx
        self.layer_idx = layer_idx

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        y = self.down_proj(self.act_fn(g) * u)
        if (
            self.residual_conditioner is not None
            and self.ctx.enabled
            and not (self.ctx.skip_prefill and x.shape[1] > 1)
        ):
            y = y + self.residual_conditioner(self.ctx.bank[: self.layer_idx])
        return y


class ConditionerContext:
    """Per-forward depth bank of completed decoder states (same token).

    push is idempotent per layer index: gradient-checkpoint recompute
    re-fires layer hooks, and in-place replacement keeps the bank at L
    entries with identical (detached, deterministic) values.

    skip_prefill: serving knob — apply the conditioner only during
    single-token decode steps (T==1) and skip it on multi-token prefill
    forwards. Semantics: the request enters decode with conditioner-off
    context (a train/inference mismatch — the conditioner was trained with
    it active at all T); the A/B battery quantifies the trade.
    """

    def __init__(self):
        self.enabled = True
        self.skip_prefill = False
        self.bank: list[torch.Tensor] = []

    def reset(self) -> None:
        self.bank = []

    def push(self, layer_idx: int, x: torch.Tensor) -> None:
        if layer_idx < len(self.bank):
            self.bank[layer_idx] = x.detach()
        else:
            self.bank.append(x.detach())


def _layer_hidden(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    for attr in ("hidden_states", "last_hidden_state"):
        if getattr(output, attr, None) is not None:
            return getattr(output, attr)
    raise TypeError(f"cannot extract hidden state from {type(output)}")


def attach_postdown(text_model: nn.Module, channels_per_layer: int,
                    seed_base: int = ROUTE_SEED_BASE,
                    dtype: torch.dtype = torch.float32) -> ConditionerContext:
    """Graft post-down dense residual conditioners onto every layer l > 0.

    dtype selects the conditioner parameter dtype: fp32 default; bf16
    halves optimizer state for the largest C values where fp32 residency
    (params+grads+momentum) tips the allocator back into a malloc spiral.
    """
    layers = text_model.layers
    d_model = layers[0].mlp.down_proj.out_features  # residual width, NOT gate_proj's intermediate
    ctx = ConditionerContext()
    for i, layer in enumerate(layers):
        cond = ResidualConditioner(i, d_model, channels_per_layer, seed_base, dtype) if i else None
        wrapped = ConditionedMLP(layer.mlp, cond, ctx, i)
        setattr(layer, "mlp", wrapped)
        layer.register_forward_hook(
            lambda mod, args, output, _i=i: ctx.push(_i, _layer_hidden(output)))
    text_model.register_forward_pre_hook(lambda module, args: ctx.reset())
    return ctx


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if ".residual_conditioner." in n)


def conditioner_state_dict(model: nn.Module) -> dict:
    """Conditioner weights + persistent route buffers (spec §33)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ".residual_conditioner." in k}
