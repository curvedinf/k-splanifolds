"""FP32 width-reduced port of modded-nanoGPT record 92; upstream MIT license alongside.

Retains the eleven-slot topology, paired/mixed-width attention, MUDD, value and
n-gram injections. Torch SDPA replaces FA3; FastKS2 uses bounded Horner geometry.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from experiments.common.ks2_stabilized import KS2StabilizedCell, _predicates, _strength

UPSTREAM = '4ea6b937337a4889b8cfe3f38a93d120048d8f71'
NUM_LAYERS = 11
NO_ATTN_LAYERS, NO_MLP_LAYERS = (4, 6, 7, 9), (7,)
ATTN_LAYERS, LONG_WINDOW_LAYERS = (0, 1, 2, 3, 5, 8, 10), (3, 10)
PAIRED_HEAD_LAYERS, HALF_V_LAYERS = (0, 2, 5), (1, 8)
WIDE_QK_LAYERS, NARROW_HEAD_DIM = LONG_WINDOW_LAYERS, 8
ATTN_BANK_ORDER, NUM_QK64_SLOTS = (0, 2, 5, 1, 8, 3, 10), 5
ATTN_GATE_LAYERS, XSA_LAYERS = (3, 10), (1, 3)
CACHE_LAYERS, VALUE_EMBED_LAYERS = (3, 7), (1, 2, 8, 10)
VALUE_EMBED_GATE_LAYERS, VALUE_EMBED_GATE_CHANNELS = (1, 2, 8), 6
X0_INJECT_LAYERS, BIGRAM_INJECT_LAYERS = (0, 1, 2, 4, 5, 7), (0, 1, 4, 5, 9)
POST_GATE_LAYER = 4
PRE_GATE_X0_LAYERS, PRE_GATE_BIGRAM_LAYERS = (0, 1, 2), (0, 1)
POST_GATE_X0_LAYERS, POST_GATE_BIGRAM_LAYERS = (4, 5, 7), (4, 5, 9)
PRE_GATE_ATTN_GATE_LAYERS, POST_GATE_ATTN_GATE_LAYERS = (3,), (10,)
PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, 11
MUDD_GROUPS, LAST_LAYER_MUDD_COEFS = 12, 14
MUDD_GATE_HEAD_LANES, MUDD_GATE_SCALE = 6, .1


def norm(x):
    return F.rms_norm(x, (x.shape[-1],))


def rms_norm_with_head(x, n):
    x = norm(x)
    return x, x[..., :n]


def scale(a, x):
    return a * x


def scale_add(a, x, b, y):
    return a * x + b * y


class FastKS2(KS2StabilizedCell):
    """One geometry step; cubic Horner chains; batched multivariate edge reduction."""
    def __init__(self, width, hidden, rank=8):
        per_chart = 2 * width * rank + 12 * rank * rank + 6 * rank + 4
        charts = (2 * width * hidden // 4) // per_chart
        if charts < 1:
            raise ValueError('Width too small for a quarter-budget KS2 chart')
        super().__init__(width, width, charts, rank, 1)
        nn.init.zeros_(self.up.weight)

    @staticmethod
    def longitudinal(t, z, weights):
        a, b, c, d, e, f = weights.unbind(1)
        return (((2*a-2*b+c+d)*t + (-3*a+3*b-2*c-d))*t+c)*t+a + ((4*f*z+2*e)*z-3*f)*z-e

    def geo_step(self, coords):
        x = coords.float().clamp(-65504, 65504)
        t = 4 * torch.tanh(x.mean(-1, keepdim=True) / 4)
        z = self.warp(t, self.geo_raw_scale[None, :, None], self.geo_raw_shift[None, :, None])
        spine = self.longitudinal(t, z, self.geo_spine)
        edge_t = self.longitudinal(t[..., None], z[..., None], self.geo_t_basis_edge)
        zx = self.warp(x, self.geo_edge_raw_scale[None, :, None], self.geo_edge_raw_shift[None, :, None])[..., None]
        a, b, c, d = self.geo_x_basis_edge.unbind(1)
        edge_x = ((4*d*zx+2*c)*zx+b-3*d)*zx+a-c
        distance, importance = _predicates(x)
        den = 1 + _strength(self.geo_den_delta) * distance[..., None]
        den = den + _strength(self.geo_den_weight) * importance[..., None]
        return spine + (edge_t * edge_x / den).sum(-2)


class MLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.down = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(hidden, width, bias=False)
        nn.init.uniform_(self.down.weight, -math.sqrt(3 / width) * .5, math.sqrt(3 / width) * .5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.relu(self.down(x)).square())


class Ngrams(nn.Module):
    def __init__(self, width, rows):
        super().__init__()
        self.table = nn.Embedding(rows, width)
        nn.init.zeros_(self.table.weight)
        self.register_buffer('signs', torch.randn(1024, width).sign())

    def forward(self, ids):
        p = F.pad(ids[:-1], (1, 0), value=50256)
        pp = F.pad(p[:-1], (1, 0), value=50256)
        # Fixed causal hash and sign pools; intentionally smaller than the record table.
        a, b = p * 5003 + ids, (pp * 5003 + p) * 5003 + ids
        return self.table(a % self.table.num_embeddings) * self.signs[a % 1024] + self.table(b % self.table.num_embeddings) * self.signs[b % 1024]


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False, attn_scale=.13):
        super().__init__()
        self.head_dim, self.paired, self.attn_scale = head_dim, paired, attn_scale
        freq = (1 / 1024) ** torch.linspace(0, 1, NARROW_HEAD_DIM // 2)
        freq = torch.cat((freq.repeat_interleave(2), torch.zeros(head_dim - NARROW_HEAD_DIM)))
        self.register_buffer('freq', freq)
        self.initial_window = None

    def apply_window(self, old, new):
        mix = ((old * self.freq / (2 * math.pi) - 1) / 31).clamp(0, 1)
        self.freq.mul_(old / new + mix * (1 - old / new))
        self.attn_scale *= 1 + .2 * math.log(new / old)

    def rotate(self, x):
        t, h = x.shape[1:3]
        pos = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
        if self.paired:
            pos = 2 * pos + torch.arange(h, device=x.device)[None] % 2
        angle = pos[..., None] * self.freq
        flipped = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
        sign = torch.ones_like(self.freq)
        sign[1::2] = -1
        return x * angle.cos() + flipped * angle.sin() * sign


@dataclass
class AttnArgs:
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor | None
    aux_v: torch.Tensor | None
    xsa_alpha: torch.Tensor | None
    train_max_seq_len: int
    o_gain: torch.Tensor | None = None


class Attention(nn.Module):
    def __init__(self, qk_dim, v_dim, head_dim, paired):
        super().__init__()
        self.qk_dim, self.v_dim, self.head_dim, self.paired = qk_dim, v_dim, head_dim, paired

    def forward(self, x, a, qk_w, v_w, o_w, unused=None):
        b, t, _ = x.shape
        h = 6
        qk = F.linear(x, qk_w * a.sa_lambdas[0]).view(b, t, 2*h, self.qk_dim)
        q, k = norm(qk[:, :, :h]), norm(qk[:, :, h:])
        q, k = a.yarn.rotate(q), a.yarn.rotate(k)
        if a.key_offset:
            k = torch.cat((k[..., :NARROW_HEAD_DIM], torch.cat((k[:, :1, :, NARROW_HEAD_DIM:], k[:, :-1, :, NARROW_HEAD_DIM:]), dim=1)), dim=-1)
        v = F.linear(x, v_w * a.sa_lambdas[0]).view(b, t, h, self.v_dim)
        if a.aux_v is not None:
            v = v + a.aux_v.view(b, t, h, self.head_dim)[..., :self.v_dim]
        documents = a.seqlens
        if self.paired:
            q = q.reshape(b, t*2, h//2, self.qk_dim)
            k = k.reshape(b, t*2, h//2, self.qk_dim)
            v = v.reshape(b, t*2, h//2, self.v_dim)
            documents = documents.repeat_interleave(2)
        length = q.shape[1]
        positions = torch.arange(length, device=x.device)
        delta = positions[:, None] - positions[None, :]
        mask = (delta >= 0) & (delta <= a.bm_size) & (documents[:, None] == documents[None, :])
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=a.yarn.attn_scale)
        y = y.transpose(1, 2).reshape(b, t, h, self.v_dim)
        if a.xsa_alpha is not None and not self.paired:
            y = y - a.xsa_alpha.tanh().view(b, t, h, 1) * (y*v).sum(-1, keepdim=True) / v.square().sum(-1, keepdim=True).clamp_min(1e-8) * v
        if a.attn_gate_w is not None:
            y = y * a.attn_gate_w.view(b, t, h, 1)
        gain = a.sa_lambdas[1] * (1 if a.o_gain is None else a.o_gain)
        return F.linear(y.reshape(b, t, h*self.v_dim), o_w * gain)


class GPT(nn.Module):
    def __init__(self, kind='mlp', width=96, vocab=50257, ngram_rows=4096, rank=8):
        super().__init__()
        global NARROW_HEAD_DIM
        if width % 48:
            raise ValueError('Width must be a multiple of 48')
        NARROW_HEAD_DIM = width // 12
        self.num_layers, self.num_heads, self.head_dim = 11, 6, width // 6
        self.vocab_size, self.ngram_dim = vocab, width
        self.embed = nn.Embedding(vocab, width)
        nn.init.normal_(self.embed.weight, std=.005)
        self.value_embeds = nn.Parameter(.01 * torch.randn(4*vocab, width))
        self.ngrams = Ngrams(width, ngram_rows)
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        self.post_lambdas = nn.Parameter(torch.ones(11, 2))
        self.resid_lambdas = nn.Parameter(torch.full((11, 2), math.sqrt(1.1)))
        self.scalars = nn.Parameter(torch.tensor([.5, 1.] * 11 + [0., -1.5]))
        self.ve_gate_bank = nn.Parameter(torch.zeros(3, 6, 12))
        self._num_qk_groups = 42
        self.qk_bank = nn.Parameter(torch.empty(42, 2*self.head_dim, width))
        self.vo_bank = nn.Parameter(torch.empty(14, width, width))
        bound = math.sqrt(3 / width) * .5
        nn.init.uniform_(self.qk_bank, -bound, bound)
        nn.init.uniform_(self.vo_bank, -bound, bound)
        self.attn = nn.ModuleDict({str(i): Attention(self.attn_qk_dim(i), self.attn_v_dim(i), self.head_dim, i in PAIRED_HEAD_LAYERS) for i in ATTN_LAYERS})
        self.yarn = Yarn(NARROW_HEAD_DIM, 4096, attn_scale=.13 * math.sqrt(64/NARROW_HEAD_DIM))
        self.yarn_paired_head = Yarn(NARROW_HEAD_DIM, 4096, True, attn_scale=self.yarn.attn_scale)
        self.yarn_wide = Yarn(self.head_dim, 4096, attn_scale=.085 * math.sqrt(128/self.head_dim))
        self.init_mudd(11, width)
        self.init_mudd_gate(width)
        # All shared parameters are created before the interchangeable FFNs.
        hidden = width * 11 // 3
        self.ff = nn.ModuleDict({str(i): MLP(width, hidden) if kind == 'mlp' else FastKS2(width, hidden, rank) for i in range(12) if i != 7})

    def attn_qk_dim(self, i):
        return self.head_dim if i in WIDE_QK_LAYERS else NARROW_HEAD_DIM

    def attn_v_dim(self, i):
        return NARROW_HEAD_DIM if i in HALF_V_LAYERS else self.head_dim

    def head_weight(self):
        return self.embed.weight
    def init_mudd(self, num_layers: int, model_dim: int):
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 16
        max_num_coef = LAST_LAYER_MUDD_COEFS
        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))
        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        bs_init[0, 6] = 2.0 / self._mudd_scale
        bs_init[0, 7] = 2.0 / self._mudd_scale
        bs_init[0, 8] = 1.1 ** 0.5 / self._mudd_scale
        bs_init[0, 9] = 1.0 / self._mudd_scale
        bs_init[0, 10] = 0.0
        bs_init[0, 11] = 0.05 / self._mudd_scale
        bs_init[0, 12] = 1.1 ** 0.5 / self._mudd_scale
        bs_init[0, 13] = 1.0 / self._mudd_scale
        bs_init[1, 1] = -0.5 / self._mudd_scale
        self.mudd_b2 = nn.Parameter(bs_init)
        self.mudd_w2g = nn.Parameter(torch.zeros(max_num_coef, MUDD_GROUPS, mudd_dim))

    def forward_mudd(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def forward_mudd_grouped(self, x, id, num_coef):
        h = F.gelu(F.linear(x, self.mudd_w1[id]))
        y = (F.linear(h, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        deltas = torch.einsum('btd,kgd->btkg', h, self.mudd_w2g[:num_coef]) * self._mudd_scale
        return (y.split(1, dim=-1), deltas)

    def init_mudd_gate(self, model_dim: int):
        self._mudd_gate_scale = nn.Parameter(torch.tensor(MUDD_GATE_SCALE))
        mudd_gate_dim = 16
        H = MUDD_GATE_HEAD_LANES
        assert self.num_heads == H
        pre_attn_start = H * len(XSA_LAYERS)
        pre_x0_start = pre_attn_start + H * len(PRE_GATE_ATTN_GATE_LAYERS)
        pre_bigram_start = pre_x0_start + len(PRE_GATE_X0_LAYERS)
        self._mudd_gate_pre_num_coef = pre_bigram_start + len(PRE_GATE_BIGRAM_LAYERS)
        post_x0_start = H * len(POST_GATE_ATTN_GATE_LAYERS)
        post_bigram_start = post_x0_start + len(POST_GATE_X0_LAYERS)
        post_skip_lane = post_bigram_start + len(POST_GATE_BIGRAM_LAYERS)
        self._mudd_gate_post_num_coef = post_skip_lane + 1
        max_num_coef = max(self._mudd_gate_pre_num_coef, self._mudd_gate_post_num_coef)
        self.mudd_gate_w1 = nn.Parameter(torch.empty(2, mudd_gate_dim, model_dim))
        self.mudd_gate_w2 = nn.Parameter(torch.zeros(2, max_num_coef, mudd_gate_dim))
        for j in range(2):
            nn.init.kaiming_uniform_(self.mudd_gate_w1.data[j], a=math.sqrt(5))
        bs_init = torch.zeros(2, max_num_coef)
        attn_gate_bias = 0.25 / MUDD_GATE_SCALE
        bigram_gate_bias = 0.05 / MUDD_GATE_SCALE
        skip_gate_bias = 0.5 / MUDD_GATE_SCALE
        bs_init[0, pre_attn_start:pre_x0_start].fill_(attn_gate_bias)
        bs_init[0, pre_bigram_start:self._mudd_gate_pre_num_coef].fill_(bigram_gate_bias)
        bs_init[1, 0:post_x0_start].fill_(attn_gate_bias)
        bs_init[1, post_bigram_start:post_skip_lane].fill_(bigram_gate_bias)
        bs_init[1, post_skip_lane].fill_(skip_gate_bias)
        self.mudd_gate_b2 = nn.Parameter(bs_init)

    def forward_mudd_gate(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_gate_w1[id]))
        return (F.linear(x, self.mudd_gate_w2[id, :num_coef]) + self.mudd_gate_b2[id, :num_coef]) * self._mudd_gate_scale.type_as(x)

    @staticmethod
    def _unpack_lanes(gate, start, layers, gates, width=1):
        for k, layer in enumerate(layers):
            lo = start + k * width
            gates[layer] = gate[..., lo:lo + width]
        return start + len(layers) * width

    def unpack_pre_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates, bigram_gates):
        lane = self._unpack_lanes(gate, 0, XSA_LAYERS, xsa_alphas, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane == self._mudd_gate_pre_num_coef

    def unpack_post_mudd_gate(self, gate, attn_gates, x0_gates, bigram_gates):
        lane = self._unpack_lanes(gate, 0, POST_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, POST_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, POST_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane + 1 == self._mudd_gate_post_num_coef
        return gate[..., lane:lane + 1]

    def _inject(self, i, x, x0, x0_bigram, x0_gates, bigram_gates):
        if i in X0_INJECT_LAYERS:
            x = x + x0 * x0_gates[i]
        if i in BIGRAM_INJECT_LAYERS and i != 0:
            x = x + F.pad(x0_bigram * bigram_gates[i], (0, x.shape[-1] - self.ngram_dim))
        return x

    def _attn_weights(self):
        H, head_dim, dim = (self.num_heads, self.head_dim, self.qk_bank.shape[-1])
        num_slots = len(ATTN_BANK_ORDER)
        qk_heads = self.qk_bank[:self._num_qk_groups].view(num_slots, 2 * H, head_dim, dim)
        qk_full = qk_heads.flatten(1, 2).unbind(0)
        qk_narrow = qk_heads[:NUM_QK64_SLOTS, :, :NARROW_HEAD_DIM].reshape(NUM_QK64_SLOTS, 2 * H * NARROW_HEAD_DIM, dim).unbind(0)
        vo = self.vo_bank[:2 * num_slots].unbind(0)
        weights = {}
        for slot, layer in enumerate(ATTN_BANK_ORDER):
            qk_w = qk_full[slot] if layer in WIDE_QK_LAYERS else qk_narrow[slot]
            v_w, o_w = (vo[2 * slot], vo[2 * slot + 1])
            v_dim = self.attn_v_dim(layer)
            if v_dim < head_dim:
                v_w = v_w.view(H, head_dim, dim)[:, :v_dim].reshape(H * v_dim, dim)
                o_w = o_w.view(dim, H, head_dim)[:, :, :v_dim].reshape(dim, H * v_dim)
            weights[layer] = (qk_w, v_w, o_w)
        return weights

    def forward(self, input_seq, ws_short, ws_long, seqlens):
        assert input_seq.ndim == 1
        train_max_seq_len = input_seq.numel()
        bm_sizes = [ws_long if i in LONG_WINDOW_LAYERS else ws_short for i in range(self.num_layers)]
        key_offset = [i in LONG_WINDOW_LAYERS for i in range(self.num_layers)]
        sa_lambdas = self.scalars[:2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].float().unbind(0)
        resid_lambdas_mlp = self.resid_lambdas[:, 1].float().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].float().unbind(0)
        post_lambdas_mlp = self.post_lambdas[:, 1].float().unbind(0)
        ve_gates = [None] * self.num_layers
        for layer, gate in zip(VALUE_EMBED_GATE_LAYERS, self.ve_gate_bank.unbind(0)):
            ve_gates[layer] = gate
        attn_gates = [None] * self.num_layers
        xsa_alphas = [None] * self.num_layers
        x0_gates = [None] * self.num_layers
        bigram_gates = [None] * self.num_layers
        attn_weights = self._attn_weights()
        x = self.embed(input_seq)
        x0_bigram = self.ngrams(input_seq)[None]
        ve_planes = self.value_embeds.view(4, self.vocab_size, -1)[:, input_seq]
        ve = [None] * self.num_layers
        for layer, plane in zip(VALUE_EMBED_LAYERS, ve_planes):
            ve[layer] = plane
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])
        pre_gate = self.forward_mudd_gate(x0, id=0, num_coef=self._mudd_gate_pre_num_coef)
        self.unpack_pre_mudd_gate(pre_gate, xsa_alphas, attn_gates, x0_gates, bigram_gates)
        x = x0.clone()
        x = x + F.pad(x0_bigram * bigram_gates[0], (0, x.shape[-1] - self.ngram_dim))
        skip_gate_out = None
        post_skip_gate = None
        cache = {0: x}
        late_attn_in = None
        for i in range(self.num_layers):
            mu = None
            if i == POST_GATE_LAYER:
                post_gate = self.forward_mudd_gate(x, id=1, num_coef=self._mudd_gate_post_num_coef)
                post_skip_gate = self.unpack_post_mudd_gate(post_gate, attn_gates, x0_gates, bigram_gates)
            if i == 6:
                assert post_skip_gate is not None
                skip_gate_out = torch.sigmoid(skip_lambda) * post_skip_gate
                x = x + skip_gate_out * cache[3]
            elif i in NO_ATTN_LAYERS:
                x = self._inject(i, scale(resid_lambdas_attn[i], x), x0, x0_bigram, x0_gates, bigram_gates)
            else:
                ve_gate_head = None
                if late_attn_in is not None:
                    attn_in_normed, attn_x_f8 = late_attn_in
                else:
                    attn_in = cache.get(7, x)
                    if i in VALUE_EMBED_GATE_LAYERS:
                        attn_in_normed, ve_gate_head = rms_norm_with_head(attn_in, VALUE_EMBED_GATE_CHANNELS)
                    else:
                        attn_in_normed = norm(attn_in)
                    attn_x_f8 = None
                    if 7 in cache:
                        late_attn_in = (attn_in_normed, attn_x_f8)
                qkv_fp8 = None
                B, T = (attn_in_normed.size(0), attn_in_normed.size(1))
                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=LAST_LAYER_MUDD_COEFS)
                    v_mudd = mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x
                    v_mudd = v_mudd.view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(self.num_heads // 2, dim=-1).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    gate_in = torch.cat([ve_gate_head, ve[i][None, ..., :VALUE_EMBED_GATE_CHANNELS]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None
                if i in WIDE_QK_LAYERS:
                    yarn = self.yarn_wide
                elif i in PAIRED_HEAD_LAYERS:
                    yarn = self.yarn_paired_head
                else:
                    yarn = self.yarn
                attn_args = AttnArgs(sa_lambdas=sa_lambdas[i], seqlens=seqlens, bm_size=bm_sizes[i], yarn=yarn, key_offset=key_offset[i], attn_gate_w=attn_gates[i] if i in ATTN_GATE_LAYERS else None, aux_v=aux_v, xsa_alpha=xsa_alphas[i], train_max_seq_len=train_max_seq_len, o_gain=post_lambdas_attn[i] if mu is None else None)
                qk_w, v_w, o_w = attn_weights[i]
                attn_out = self.attn[str(i)](attn_in_normed, attn_args, qk_w, v_w, o_w, qkv_fp8)
                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0]
                    x = x + F.pad(mu[11] * x0_bigram, (0, x.shape[-1] - self.ngram_dim))
                else:
                    x = scale(resid_lambdas_attn[i], x) + attn_out
                    x = self._inject(i, x, x0, x0_bigram, x0_gates, bigram_gates)
            if i in NO_MLP_LAYERS:
                x = scale(resid_lambdas_mlp[i], x)
                if i in CACHE_LAYERS:
                    cache[i] = x
                continue
            mlp_in = norm(x)
            mlp_out = self.ff[str(i)](mlp_in)
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            else:
                x = scale_add(resid_lambdas_mlp[i], x, post_lambdas_mlp[i], mlp_out)
            if i == PARALLEL_MLP_LAYER:
                k = PARALLEL_MLP_SLOT
                parallel_out = self.ff[str(k)](mlp_in)
                x = x + scale(post_lambdas_mlp[i], parallel_out)
            if i in CACHE_LAYERS:
                cache[i] = x
        assert late_attn_in is not None, 'norm(cache[7]) is not bound at loop exit'
        sources = [cache[0], cache[7], cache[9], ve[1][None].to(dtype=x.dtype), cache[3], ve[2][None].to(dtype=x.dtype), ve[10][None].to(dtype=x.dtype), ve[8][None].to(dtype=x.dtype), mlp_in, late_attn_in[0]]
        mu, deltas = self.forward_mudd_grouped(x, id=1, num_coef=len(sources))
        grouped = lambda t: t.unflatten(-1, (MUDD_GROUPS, -1))
        mixed = grouped(x)
        for k, src in enumerate(sources):
            mixed = mixed + (mu[k] + deltas[..., k, :]).unsqueeze(-1) * grouped(src)
        x = mixed.flatten(-2)
        return norm(x)[0]
