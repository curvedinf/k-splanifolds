"""RHAC graft: spread-coded decoder-history conditioning with the KS2 local surface.

Ports the best-known decoder-history configuration from the SmolLM2-135M
experiments (RHAC_code_artifacts/) to any HF causal LM with a SwiGLU MLP
(Qwen3_5 family included), with the KS2 block cell as the local nonlinear
surface (see ks2.py):

  frozen base MLP computation:  h = [SiLU(g) + R(g, history)] * u
  R = out_scale * (KS2(q) - q)  +  linear-in-spine external field over fixed
      spread-coded views of the collapsed decoder-block history

Memory adaptations for large models (math unchanged):
  - external-field contraction chunked over activation groups
  - history bank kept as detached bf16 per-layer tensors; the depth-collapse
    summary is recomputed on demand as a pure function of bank[:i] so it is
    correct under gradient-checkpoint recomputation

Fidelity: parameter shapes/init, route-table construction (generator call
order), spread-view construction order, collapse (RMS-matched depth mean),
and slot cycling match the original artifacts. All adapter state bf16 with
fp32 math; landing vectors v0/v1 zero-init and out_scale zero-init make the
graft an exact no-op at initialization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ks2 import KS2BlockBank


def hadamard(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def pick_block_size(inter: int, preferred: int = 0) -> int:
    """K must divide the MLP intermediate size. Prefer the SmolLM2 optimum
    (K=12), then the next sizes up/down the whitepaper's monotonic trend."""
    order = [12, 16, 8, 32, 24, 64, 4, 2]
    if preferred:
        order = [preferred] + [k for k in order if k != preferred]
    for k in order:
        if inter % k == 0:
            return k
    raise ValueError(f"no usable block size divides intermediate_size={inter}")


class SpreadViews(nn.Module):
    """Fixed spread encoder: R views of a (B,T,D) summary.

    Per view: permute dims -> mix chunks of width `mix` with an orthonormal
    Hadamard -> random +/- signs. No trainable parameters. The composite
    transform per view is a fixed D x D matrix (perm . blockdiag(H) . signs),
    so all R views are computed as ONE matmul against W_flat (R*D, D).
    """

    def __init__(self, dim: int, R: int, mix: int, seed: int):
        super().__init__()
        assert mix > 0 and (mix & (mix - 1)) == 0, "mix must be a power of two"
        assert dim % mix == 0, f"dim {dim} not divisible by mix {mix}"
        gen = torch.Generator().manual_seed(seed)
        perms = torch.stack([torch.randperm(dim, generator=gen) for _ in range(R)])
        signs = torch.stack(
            [(torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int8).float() * 2 - 1) for _ in range(R)]
        )
        self.register_buffer("perms", perms, persistent=False)
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("H", hadamard(mix), persistent=False)
        self.dim, self.R, self.mix = dim, R, mix
        # composite view matrices: W[r, i, j] = signs[r,i] * H[i % mix, j] on
        # position perm[r][(i//mix)*mix + j]; W_flat (R*D, D) enables a single
        # GEMM for all views
        W = torch.zeros(R, dim, dim)
        for r in range(R):
            for i in range(dim):
                c, j = i // mix, i % mix
                for ip in range(mix):
                    W[r, i, perms[r][c * mix + ip]] += signs[r, i] * self.H[j, ip]
        # bf16 storage: views are consumed in bf16; casting a 400 MB fp32
        # matrix per forward was pure traffic
        self.register_buffer("W_flat", W.reshape(R * dim, dim).contiguous().to(torch.bfloat16),
                             persistent=False)
        self._w_dtype = torch.bfloat16

    def forward(self, s: torch.Tensor) -> list[torch.Tensor]:
        flat = self.forward_flat(s)
        return [flat[..., r * self.dim : (r + 1) * self.dim] for r in range(self.R)]

    def forward_flat(self, s: torch.Tensor) -> torch.Tensor:
        """All R views: (B,T,D) -> (B,T,R*D).

        Prefill (large T): one bf16 GEMM against W_flat (R*D, D) — the
        weight read amortizes over T tokens. Decode (tiny T): the composite
        W is never touched (a T=1 GEMV would stream the whole R*D x D
        matrix per layer per token); the cheap elementwise path reads only
        the small perm/sign/Hadamard buffers.
        """
        if s.shape[1] <= 4:
            # elementwise decode path: the small perm/sign/Hadamard buffers
            # must live on-device (pageable H2D per call is a sync per copy —
            # 576 per token across layers — and is illegal during CUDA-graph
            # capture), so cache device copies lazily like the GEMM path's W
            dev = getattr(self, "_ew_dev", None)
            if dev is None or dev[0][0].device != s.device or dev[2].dtype != s.dtype:
                dev = (
                    [p.to(s.device) for p in self.perms],
                    [sg.to(device=s.device, dtype=s.dtype) for sg in self.signs],
                    self.H.to(device=s.device, dtype=s.dtype),
                )
                self._ew_dev = dev
            perms_d, signs_d, H_d = dev
            views = []
            for r in range(self.R):
                y = s[:, :, perms_d[r]]
                y = y.view(*y.shape[:-1], self.dim // self.mix, self.mix)
                y = torch.matmul(y, H_d.t()).reshape(*s.shape)
                views.append(y * signs_d[r])
            return torch.cat(views, dim=-1)
        # GEMM path always bf16 (W is stored bf16; upcasting per call would
        # re-stream the full matrix). The encoder is not a model submodule,
        # so cache the device-side copy lazily.
        w = getattr(self, "_w_dev", None)
        if w is None or w.device != s.device:
            w = self.W_flat.to(s.device)
            self._w_dev = w
        return s.to(torch.bfloat16) @ w.t()


class RHAC(nn.Module):
    """KS2 local surface plus spread-coded decoder-history external field.

    Every history scalar is assigned to exactly one activation group (one
    landing site). The external field is linear in the local spine t:
    sum_e c_e * [(1-t) v0_e + t v1_e], contracted directly (never
    materializing B x T x G x M x K). Landing vectors and out_scale are
    zero-initialized, so the adapter is an exact no-op at initialization.
    """

    def __init__(self, ff_dim: int, history_dim: int, K: int, seed: int = 0,
                 group_chunk: int = 256, geo_steps: int = 2,
                 dtype: torch.dtype = torch.bfloat16,
                 dim: int = 0, views: int = 4):
        super().__init__()
        assert ff_dim % K == 0
        self.K, self.G, self.history_dim = K, ff_dim // K, history_dim
        self.group_chunk = group_chunk
        self.ks2 = KS2BlockBank(self.G, K, geo_steps=geo_steps, dtype=dtype, seed=seed + 7777)
        if history_dim:
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(history_dim, generator=gen)
            gid = torch.arange(history_dim) % self.G
            gid = gid[torch.randperm(history_dim, generator=gen)]
            buckets = [[] for _ in range(self.G)]
            for src, g in zip(perm.tolist(), gid.tolist()):
                buckets[g].append(src)
            M = max(map(len, buckets))
            idx = torch.full((self.G, M), -1, dtype=torch.long)
            valid = torch.zeros((self.G, M), dtype=torch.bool)
            for g, b in enumerate(buckets):
                if b:
                    idx[g, : len(b)] = torch.tensor(b)
                    valid[g, : len(b)] = True
            self.register_buffer("route_idx", idx, persistent=True)
            self.register_buffer("route_valid", valid, persistent=True)
            # remap into view space: old flat slot h held view
            # (h // dim) % views, channel h % dim — the adapter gathers
            # directly from the (B,T,views*dim) one-GEMM views tensor,
            # eliminating the per-layer slot cat entirely
            assert dim > 0 and history_dim % dim == 0
            safe = idx.clamp_min(0)
            mapped = ((safe // dim) % views) * dim + (safe % dim)
            self.register_buffer("route_view_idx", mapped, persistent=True)
            self.v0 = nn.Parameter(torch.zeros(self.G, M, K, dtype=dtype))
            self.v1 = nn.Parameter(torch.zeros(self.G, M, K, dtype=dtype))

    def forward(self, g: torch.Tensor, history: Optional[torch.Tensor]) -> torch.Tensor:
        from .fused import fused_enabled, fused_rhac_forward

        if history is not None and fused_enabled() and g.is_cuda and g.dtype == torch.bfloat16:
            return fused_rhac_forward(g, history, self)
        return self.forward_eager(g, history)

    def forward_eager(self, g: torch.Tensor, history: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, _ = g.shape
        q = torch.sigmoid(g.float() / 1.5).view(B, T, self.G, self.K)
        geo = self.ks2(q)
        loc = self.ks2.out_scale.float().view(1, 1, self.G, 1) * (geo - q)
        if not self.history_dim or history is None:
            return loc.reshape_as(g).to(g.dtype)
        flat = history.reshape(B, T, -1)
        t = q.mean(-1, keepdim=True)
        chunks = []
        for g0 in range(0, self.G, self.group_chunk):
            g1 = min(g0 + self.group_chunk, self.G)
            idx = self.route_view_idx[g0:g1]
            valid = self.route_valid[g0:g1]
            c = flat[:, :, idx].float() * valid
            e0 = torch.einsum("btgm,gmk->btgk", c, self.v0[g0:g1].float())
            e1 = torch.einsum("btgm,gmk->btgk", c, self.v1[g0:g1].float())
            tc = t[:, :, g0:g1]
            chunks.append(loc[:, :, g0:g1] + (1 - tc) * e0 + tc * e1)
        corr = torch.cat(chunks, dim=2)
        return corr.reshape_as(g).to(g.dtype)


class RHACMLP(nn.Module):
    """SwiGLU MLP with the RHAC activation correction grafted in.

    Keeps the base projections as child modules under their original names,
    so base state-dict keys are unchanged. With ctx.enabled False (or no
    adapter) the forward is exactly the base MLP computation.
    """

    def __init__(self, base_mlp: nn.Module, adapter: Optional[RHAC], ctx: "RHACContext", layer_idx: int):
        super().__init__()
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        self.act_fn = getattr(base_mlp, "act_fn", None) or F.silu
        self.adapter = adapter
        self.ctx = ctx
        self.layer_idx = layer_idx

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        a = self.act_fn(g)
        if (
            self.adapter is not None
            and self.ctx.enabled
            and not (self.ctx.skip_prefill and x.shape[1] > 2)
        ):
            a = a + self.adapter(g, self.ctx.history_for(self.layer_idx))
        return self.down_proj(a * u)


class RHACContext:
    """Per-forward state: the detached decoder-history bank and the on/off flag.

    `enabled=False` turns every grafted MLP into the exact base computation
    (used for teacher passes and base-function evaluations).

    The depth-collapse summary is maintained incrementally: rs/rss are the
    running sum and sum-of-squares of the bank, so summary(i) costs O(D)
    instead of iterating bank[:i] (the old O(L^2) per-token/per-prompt cost).
    push() is overwrite-safe for gradient-checkpoint recomputation.
    """

    def __init__(self, dim: int, views: int, mix: int, view_seed: int,
                 history_dtype: torch.dtype = torch.bfloat16):
        self.enabled = True
        # serving knob: apply the adapter only during single-token decode
        # steps (T==1) and skip it on multi-token prefill forwards, where
        # the graft costs ~2.8x wall time. Semantics: the request enters
        # decode with adapter-off context — re-validate quality (PPL /
        # coherence) before using it for scored runs. Default False here so
        # training (full-sequence forwards) is untouched; the server turns
        # it on by default and exposes --adapter_prefill to restore full
        # adapter prefill.
        self.skip_prefill = False
        self.bank: list[torch.Tensor] = []
        self.rs: torch.Tensor | None = None   # (B,T,D) fp32 running sum
        self.rss: torch.Tensor | None = None  # (B,T) fp32 running sum of squares
        self.history_dtype = history_dtype
        # decoder-history encoder seed matches rhac_dual_spread: seed + 99173
        self.encoder = SpreadViews(dim, views, mix, view_seed + 99173)

    def reset(self) -> None:
        self.bank = []
        self.rs = None
        self.rss = None

    def push(self, layer_idx: int, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        xi = x.detach()
        xf = xi.float()
        xsq = xf.square().sum(-1)
        if layer_idx < len(self.bank):
            # gradient-checkpoint recomputation re-fires the hook; subtract
            # the stale entry before adding (values identical -> net zero)
            old = self.bank[layer_idx].float()
            self.rs -= old
            self.rss -= old.square().sum(-1)
            self.bank[layer_idx] = xi
        else:
            self.bank.append(xi)
        if self.rs is None or self.rs.shape != xf.shape:
            # fresh forward (or a harness path without the reset pre-hook):
            # start the running sums from this entry
            self.rs = xf.clone()
            self.rss = xsq.clone()
        else:
            self.rs = self.rs + xf
            self.rss = self.rss + xsq

    def _summary(self, i: int) -> torch.Tensor:
        """RMS-matched depth-mean of bank[:i] from the running sums —
        identical to _collapse() in the artifacts."""
        s = self.rs / i
        target = (self.rss / (i * s.shape[-1])).sqrt().unsqueeze(-1)
        cur = s.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        s = s * (target / cur)
        return s.detach()

    def history_for(self, layer_idx: int) -> Optional[torch.Tensor]:
        L = min(layer_idx, len(self.bank))
        if L == 0:
            return None
        s = self._summary(L)
        # bf16 keeps the prefill GEMM on tensor cores; decode takes the
        # elementwise path inside forward_flat (dtype matches history_dtype)
        if s.shape[1] <= 4:
            return self.encoder.forward_flat(s).to(self.history_dtype)
        return self.encoder.forward_flat(s.to(torch.bfloat16)).to(self.history_dtype)


def _layer_hidden(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    for attr in ("hidden_states", "last_hidden_state"):
        if getattr(output, attr, None) is not None:
            return getattr(output, attr)
    raise TypeError(f"cannot extract hidden state from layer output {type(output)}")


def attach_rhac(
    text_model: nn.Module,
    *,
    K: int = 0,
    views: int = 4,
    mix: int = 4,
    view_seed: int = 4242,
    route_seed_base: int = 1234,
    group_chunk: int = 256,
    geo_steps: int = 2,
    adapter_dtype: torch.dtype = torch.bfloat16,
    history_dtype: torch.dtype = torch.bfloat16,
) -> RHACContext:
    """Graft decoder-history RHAC adapters onto every decoder layer.

    `text_model` is the transformer body exposing `.layers[i].mlp` (e.g.
    Qwen3_5TextModel or the `.model` of a Qwen3_5ForCausalLM). Registers a
    forward hook per layer that stashes the completed decoder-block output
    (detached) into the shared history bank, and a pre-hook on the body that
    resets the bank each forward.
    """
    layers = text_model.layers
    dim = text_model.config.hidden_size
    assert dim % mix == 0, f"hidden {dim} not divisible by mix {mix}"
    ctx = RHACContext(dim, views, mix, view_seed, history_dtype=history_dtype)
    inter = layers[0].mlp.gate_proj.out_features
    K = pick_block_size(inter, K)
    for i, layer in enumerate(layers):
        adapter = RHAC(inter, i * dim, K, seed=route_seed_base + i,
                       group_chunk=group_chunk, geo_steps=geo_steps, dtype=adapter_dtype,
                       dim=dim, views=views)
        layer.mlp = RHACMLP(layer.mlp, adapter, ctx, i)

        def _stash(module, args, output, _i=i):
            ctx.push(_i, _layer_hidden(output))

        layer.register_forward_hook(_stash)
    text_model.register_forward_pre_hook(lambda module, args: ctx.reset())
    return ctx


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".adapter." in k}


def adapter_param_count(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
