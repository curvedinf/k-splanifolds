"""RHAC trainer — decoder-history spread-routed activation conditioning with
the KS2 local surface, on a frozen Qwen3_5-family model (Qwen3.8-27B target;
any smaller sibling works).

Locked recipe (RHAC/PLAN.md):
  frozen base + KS2(rank-K) local correction (zero-init out_scale) +
  linear-in-spine external field over 4 fixed views / 4-way Hadamard spread
  of the collapsed decoder-block history, detached history, SGD 3e-3 /
  momentum 0.9 / wd 1e-5, grad-clip 1.0, batch 1, answer-token CE,
  K=32, adapter state bf16 (fp32 math).

Strix Halo memory design:
  - single resident weight copy: the frozen base doubles as its own teacher
    (adapters off == exact base function; zero-init makes this lossless)
  - mmap'd base checkpoint + per-layer weight streaming with eviction
    (--stream_weights): GPU holds one decoder block at a time
  - gradient checkpointing, answer-slice lm_head application (vocab 248k),
    chunked external-field contraction, bf16 history bank
  - hybrid-SSM rule: distillation windows default to seq_len 512 (>= 512
    keeps GDN gradients alive); GDN forward T-chunked
  - no torch.compile anywhere
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parent.parent  # RHAC/src
REPO_ROOT = _SRC.parent  # RHAC/
for p in (REPO_ROOT, _SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import data as rdata
from .rhac import adapter_param_count, adapter_state_dict, attach_rhac, pick_block_size


class LMWithHead(nn.Module):
    """Qwen3_5TextModel + separate lm_head, matching stripped LM-only
    checkpoint keys (layers.N.* / embed_tokens / norm / lm_head)."""

    def __init__(self, text_model: nn.Module, vocab_size: int, logits_chunk: int = 0):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        hidden = text_model.config.hidden_size
        dtype = next(text_model.parameters()).dtype
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False).to(dtype)
        self.logits_chunk = logits_chunk
        self.name_or_path = "rhac-lmwithhead"

    @property
    def device(self):
        return self.lm_head.weight.device

    def tie_weights(self):
        pass  # untied by construction; satisfies lm-eval HFLM init

    def forward(self, input_ids=None, **kw):
        out = self.model(input_ids=input_ids, **kw)
        h = out.last_hidden_state
        if self.logits_chunk > 0 and torch.is_grad_enabled():
            flat = h.reshape(-1, h.shape[-1])
            parts = [self.lm_head(flat[i : i + self.logits_chunk])
                     for i in range(0, flat.shape[0], self.logits_chunk)]
            logits = torch.cat(parts, dim=0).reshape(*h.shape[:-1], -1)
        else:
            logits = self.lm_head(h)
        return SimpleNamespace(logits=logits)

TINY_CFG = dict(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=8,
    full_attention_interval=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    attention_bias=False,
    attn_output_gate=True,
    linear_num_value_heads=4,
    linear_num_key_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=64,
    tie_word_embeddings=False,
)


def install_gdn_t_chunking(step: int) -> bool:
    """Chunk the GatedDeltaNet torch fallback over the sequence dimension:
    FLA chunk kernels stay disabled so the numerically-correct torch fallback
    runs, threaded over T in `step`-token slices to bound the SSM forward
    peak. Must run before model build."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35

    raw = getattr(q35, "torch_chunk_gated_delta_rule", None)
    if raw is None:
        return False
    raw_rec = getattr(q35, "torch_recurrent_gated_delta_rule", None)
    if getattr(raw, "_rhac_raw_torch_chunk", None) is not None:
        return True

    def _t_chunked(q, k, v, g, beta, chunk_size=64, initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        T = q.shape[1]
        if T == 1 and raw_rec is not None:
            # decode step: the chunked fallback pads 1 token to chunk_size and
            # runs a chunk_size-iteration python loop; the recurrent path is O(1)
            return raw_rec(
                q, k, v, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        step = 2048
        outs, state = [], initial_state
        for t0 in range(0, T, step):
            t1 = min(T, t0 + step)
            o, state = raw(
                q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g[:, t0:t1], beta[:, t0:t1],
                chunk_size=chunk_size, initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outs.append(o)
        return torch.cat(outs, dim=1), (state if output_final_state else None)

    _t_chunked._rhac_raw_torch_chunk = raw
    q35.chunk_gated_delta_rule = None
    q35.fused_recurrent_gated_delta_rule = None
    q35.torch_chunk_gated_delta_rule = _t_chunked
    return True


class PinnedPool:
    """One reusable pinned staging buffer for CPU->GPU weight streaming.

    Copying a large PAGEABLE cpu tensor to the GPU makes the HIP runtime
    allocate a full-size pinned staging buffer per copy, and those are anon
    memory the runtime never returns — 64 layers of distinct tensor sizes
    pinned ~50 GB and OOM-killed the host. Routing every copy through a
    single grow-to-fit pinned buffer bounds pinned memory to the largest
    single tensor (~178 MB bf16 at 27B).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.buf: torch.Tensor | None = None

    def to_device(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        if self.buf is None or self.buf.numel() < nbytes:
            self.buf = torch.empty(max(nbytes, 1 << 20), dtype=torch.uint8, pin_memory=True)
        view = self.buf[:nbytes].view(cpu_tensor.dtype).view(cpu_tensor.shape)
        view.copy_(cpu_tensor)
        # non_blocking=False: torch synchronizes the stream so the pinned
        # buffer is not overwritten by the next tensor's copy while the H2D
        # is still in flight
        return view.to(self.device)


class WeightStreamer:
    """Per-module weight streaming for a frozen model on unified memory.

    Registers a forward-pre-hook that copies the module's frozen weights
    (recursively) to the device and a post-hook that re-points .data at the
    CPU masters, so the GPU holds about one decoder block at a time. All
    copies go through a shared PinnedPool. Trainable adapter params are
    never touched (they live on the device permanently), so optimizer state
    stays coherent.
    """

    def __init__(self, modules: list[nn.Module], device: torch.device):
        self.device = device
        self.pool = PinnedPool(device)
        self.ref: dict[int, torch.Tensor] = {}
        self.modules_list: list[nn.Module] = []
        for m in modules:
            self.modules_list.append(m)
            for name, p in m.named_parameters(recurse=True, remove_duplicate=True):
                if p.requires_grad or ".adapter." in name:
                    continue
                self.ref[id(p)] = p.data
            for name, b in m.named_buffers(recurse=True, remove_duplicate=True):
                if b is not None and ".adapter." not in name:
                    self.ref[id(b)] = b.data
            m.register_forward_pre_hook(self._pre)

    def _owned(self, module):
        for name, p in module.named_parameters(recurse=True, remove_duplicate=True):
            if not p.requires_grad and ".adapter." not in name:
                yield p
        for name, b in module.named_buffers(recurse=True, remove_duplicate=True):
            if b is not None and ".adapter." not in name:
                yield b

    def evict_all(self, skip_module: nn.Module | None = None) -> None:
        """Re-point every streamed .data at its CPU master.

        Gradient-checkpoint recomputation (transformers 5.x
        GradientCheckpointingLayer) does not run forward post-hooks, so
        eviction cannot rely on them: each _pre evicts everything else, and
        the trainer evicts explicitly after backward/eval. Still-referenced
        GPU tensors (live graph segments) survive via refcount.
        """
        for m in self.modules_list:
            if m is skip_module:
                continue
            for p in self._owned(m):
                p.data = self.ref[id(p)]

    def _pre(self, module, args):
        if any(p.data.device.type == self.device.type for p in self._owned(module)):
            return  # already resident (recompute of this segment)
        self.evict_all(skip_module=module)
        for p in self._owned(module):
            p.data = self.pool.to_device(self.ref[id(p)])


def resident_to_device(model: nn.Module, device: torch.device) -> PinnedPool:
    """Move the whole model to the device through the bounded pinned pool.

    `nn.Module.to(device)` issues pageable H2D copies per tensor, and the
    HIP runtime pins a full-size staging buffer per distinct tensor size —
    for the 27B that is ~50 GiB of never-returned anon and an OOM kill.
    Routing every copy through one grow-to-fit pinned buffer keeps pinned
    memory at the largest single tensor (~178 MiB).
    """
    pool = PinnedPool(device if not isinstance(device, str) else torch.device(device))
    device = pool.device
    with torch.no_grad():
        for _, p in model.named_parameters(recurse=True, remove_duplicate=True):
            if p.data.device.type != device.type:
                p.data = pool.to_device(p.data)
        for _, b in model.named_buffers(recurse=True, remove_duplicate=True):
            if b is not None and b.data.device.type != device.type:
                b.data = pool.to_device(b.data)
    return pool


def build_model(args, dtype: torch.dtype) -> nn.Module:
    if args.tiny:
        from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

        torch.manual_seed(args.init_seed)
        cfg = Qwen3_5TextConfig(**TINY_CFG)
        cfg._attn_implementation = args.attn_implementation or "eager"
        text = Qwen3_5TextModel(cfg).to(dtype)
        model = LMWithHead(text, TINY_CFG["vocab_size"], logits_chunk=args.logits_chunk)
    elif args.base_checkpoint:
        # Stripped language-model-only checkpoint (Qwen3.8-27B style), built on
        # meta and assign-loaded from an mmap'd .pt; a normal from_pretrained
        # would OOM.
        from transformers.models.qwen3_5 import Qwen3_5TextModel

        Qwen3_5TextModel._supports_flex_attn = True
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        if args.attn_implementation:
            text_cfg._attn_implementation = args.attn_implementation
        with torch.device("meta"):
            text = Qwen3_5TextModel(text_cfg)
            model = LMWithHead(text, text_cfg.vocab_size, logits_chunk=args.logits_chunk)
        ckpt = torch.load(args.base_checkpoint, weights_only=True, map_location="cpu", mmap=True)
        raw = ckpt["model_state_dict"]
        if "layers.0.input_layernorm.weight" in raw:
            load_sd = {"model." + k: v for k, v in raw.items() if k != "lm_head.weight"}
            load_sd["lm_head.weight"] = raw["lm_head.weight"]
        else:
            load_sd = raw
        missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)
        missing_base = [k for k in missing if ".adapter." not in k]
        print(f"  base checkpoint: {len(load_sd) - len(unexpected)} loaded, "
              f"{len(missing_base)} missing base keys, {len(unexpected)} unexpected", flush=True)
        if missing_base:
            print(f"  WARNING missing base keys (first 10): {missing_base[:10]}", flush=True)
        with torch.no_grad():
            fixed = 0
            fixed_names = []
            for name, module in model.named_modules():
                for pname, p in list(module.named_parameters(recurse=False)):
                    if p.is_meta:
                        module._parameters[pname] = torch.empty_like(p, device="cpu").normal_(0, 0.02)
                        fixed += 1
                        fixed_names.append(f"{name}.{pname}")
                for bname, b in list(module.named_buffers(recurse=False)):
                    if b is not None and b.is_meta:
                        newb = torch.zeros_like(b, device="cpu")
                        # rope buffers are non-persistent (never in any
                        # checkpoint) and MUST be the real frequencies —
                        # zeroing them silently kills positional encoding in
                        # every attention layer (found the hard way: thinking
                        # mode never closes, ordering tasks all zero)
                        if bname in ("inv_freq", "original_inv_freq"):
                            head_dim = getattr(model.config, "head_dim", None) or (
                                model.config.hidden_size // model.config.num_attention_heads)
                            prf = 1.0
                            rp = getattr(model.config, "rope_parameters", {}) or {}
                            prf = rp.get("partial_rotary_factor", prf)
                            theta = rp.get("rope_theta", 1000000.0)
                            dim = int(head_dim * prf)
                            newb = 1.0 / (
                                theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                            )
                        module._buffers[bname] = newb
                        fixed += 1
                        fixed_names.append(f"{name}.{bname}")
        if fixed:
            print(f"  WARNING: {fixed} tensors missing from checkpoint; initialized empty: {fixed_names}", flush=True)
        del ckpt, raw, load_sd
        gc.collect()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=dtype,
            attn_implementation=args.attn_implementation or "sdpa",
        )
    model.config.use_cache = False
    model.model.config.use_cache = False
    return model


def text_of(model: nn.Module) -> nn.Module:
    """Descend to the transformer text model (Tuner -> ForCausalLM -> TextModel)."""
    m = model
    while hasattr(m, "model") and not hasattr(m, "layers"):
        m = m.model
    return m


def hidden_forward(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    out = text_of(model)(input_ids=ids)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def answer_loss(model: nn.Module, row: dict, device: torch.device) -> torch.Tensor:
    ids = torch.tensor([row["ids"]], device=device)
    plen = row["plen"]
    h = hidden_forward(model, ids)
    hs = h[:, plen - 1 : -1, :]
    logits = model.lm_head(hs).float()
    tgt = ids[:, plen:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


@torch.inference_mode()
def eval_rows(model, rows, device, max_rows=0) -> dict:
    model.eval()
    total_nll, ntok, exact = 0.0, 0, 0
    by: dict[str, dict] = {}
    rows = rows[: max_rows] if max_rows else rows
    for r in rows:
        ids = torch.tensor([r["ids"]], device=device)
        plen = r["plen"]
        h = hidden_forward(model, ids)
        hs = h[:, plen - 1 : -1, :]
        logits = model.lm_head(hs).float()
        tgt = ids[:, plen:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
        pred = logits.argmax(-1)
        ok = bool(torch.equal(pred, tgt))
        exact += ok
        total_nll += float(ce.sum())
        ntok += tgt.numel()
        d = by.setdefault(r["cat"], {"n": 0, "exact": 0, "nll_sum": 0.0, "tokens": 0})
        d["n"] += 1
        d["exact"] += int(ok)
        d["nll_sum"] += float(ce.sum())
        d["tokens"] += tgt.numel()
    out = {"n": len(rows), "exact": exact,
           "accuracy": exact / max(1, len(rows)),
           "nll": total_nll / max(1, ntok), "categories": {}}
    for c, d in by.items():
        out["categories"][c] = {"n": d["n"], "exact": d["exact"],
                                "accuracy": d["exact"] / d["n"], "nll": d["nll_sum"] / max(1, d["tokens"])}
    return out


def teacher_log_probs(model, ctx, ids, temperature: float, chunk: int, keep_dtype=torch.bfloat16):
    """Base-model (adapters off) chunked log-softmax for distillation."""
    ctx.enabled = False
    saved_bank = ctx.bank
    try:
        with torch.no_grad():
            h = hidden_forward(model, ids)
        parts = []
        for i in range(0, h.shape[1], chunk):
            lg = model.lm_head(h[:, i : i + chunk]).float()
            parts.append(F.log_softmax(lg / temperature, dim=-1).to(keep_dtype))
        return torch.cat(parts, dim=1)
    finally:
        ctx.enabled = True
        ctx.bank = saved_bank


def window_loss(model, ctx, ids, tgt, temperature: float, distill_weight: float, chunk: int):
    """CE over the window plus optional KL(base || student), exact per-token sums.

    The teacher pass must run BEFORE the student forward: it resets the
    history bank (model-level pre-hook), which would otherwise wipe state the
    live student graph still needs for checkpoint recomputation.
    """
    teacher = None
    if distill_weight > 0:
        teacher = teacher_log_probs(model, ctx, ids, temperature, chunk)
    h = hidden_forward(model, ids)
    V = model.lm_head.out_features
    ce = kl = None
    for i in range(0, h.shape[1], chunk):
        lg = model.lm_head(h[:, i : i + chunk]).float()
        ce_i = F.cross_entropy(lg.reshape(-1, V), tgt[:, i : i + chunk].reshape(-1), reduction="sum")
        ce = ce_i if ce is None else ce + ce_i
        if teacher is not None:
            slp = F.log_softmax(lg / temperature, dim=-1)
            tlp = teacher[:, i : i + chunk].float()
            kl_i = F.kl_div(slp, tlp.exp(), reduction="sum") * (temperature * temperature)
            kl = kl_i if kl is None else kl + kl_i
    ntok = tgt.numel()
    return ce / ntok, (kl / ntok if kl is not None else None)


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return -1.0


def save_partial(model, opt, step, output_dir: Path, state_dict_fn=None):
    state_dict_fn = state_dict_fn or adapter_state_dict
    payload = {"adapter": state_dict_fn(model), "opt": opt.state_dict(), "step": step}
    tmp = output_dir / "partial.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, output_dir / "partial.pt")
    (output_dir / "partial.meta.json").write_text(json.dumps({"step": step, "time": time.time()}))
    print(f"  [partial] step {step} saved", flush=True)


def build_tasks(args, tokenizer):
    if args.task == "standard":
        # single standard calibration mix (training side): synthetic rows
        # from 4 categories + retrieval only; sequence+state held out of
        # training entirely (transfer probes). The 80% corpus-token share
        # is interleaved in the train loop from --data_dir windows.
        cats = ["arith", "algebra", "transitive", "deduction"]
        import random as _random

        rng = _random.Random(args.train_seed)
        synth_rows = []
        for c in cats:
            for _ in range(args.train_per_cat):
                p, a = rdata.make_one(rng, c)
                synth_rows.append({"cat": c, "prompt": p, "answer": a})
        synth_rows += rdata.make_retrieval(1000, args.train_per_cat, 6, 12)
        train = rdata.encode_rows(synth_rows, tokenizer)
        test = rdata.encode_rows(rdata.make_set(args.test_seed, args.test_per_cat), tokenizer)
        return train, test
    if args.task == "broad":
        train = rdata.encode_rows(rdata.make_set(args.train_seed, args.train_per_cat), tokenizer)
        test = rdata.encode_rows(rdata.make_set(args.test_seed, args.test_per_cat), tokenizer)
        return train, test
    if args.task == "retrieval":
        train_rows = []
        for gap in (0, 4, 12, 24):
            train_rows += rdata.make_retrieval(1000 + gap, args.retrieval_n, 6, gap)
        train = rdata.encode_rows(train_rows, tokenizer)
        test = rdata.encode_rows(
            [r for gap in (0, 12, 24) for r in rdata.make_retrieval(9000 + gap, 8, 6, gap)], tokenizer
        )
        return train, test
    raise ValueError(args.task)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B",
                    help="HF id or local dir (config source; also tokenizer unless --tokenizer)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--base_checkpoint", default=None,
                    help="stripped language-model-only .pt (see RHAC/scripts/extract_lm_base.py)")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny Qwen3_5 model (path tests)")

    ap.add_argument("--task", default="broad", choices=["broad", "retrieval", "lm", "standard"])
    ap.add_argument("--arch", default="splanifold", choices=["splanifold", "postdown"],
                    help="splanifold: original K-Splanifold RHAC; postdown: dense "
                         "residual conditioner after down_proj (spec)")
    ap.add_argument("--channels", type=int, default=32,
                    help="postdown: sketch channels per depth (C)")
    ap.add_argument("--steps", type=int, default=0, help="0 = per-task default (broad 72, retrieval 24, lm 50)")
    # RHAC locked configuration (K=32; falls back to a divisor if invalid)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--mix", type=int, default=4)
    ap.add_argument("--view_seed", type=int, default=4242)
    ap.add_argument("--route_seed_base", type=int, default=1234)
    ap.add_argument("--group_chunk", type=int, default=256)
    ap.add_argument("--geo_steps", type=int, default=2)
    ap.add_argument("--adapter_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--history_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    # optimizer (best-known: SGD 3e-3 / 0.9 / 1e-5, clip 1.0)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 3e-3 splanifold / 1e-3 postdown (spec dense-sweep recipe)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    # broad data
    ap.add_argument("--train_per_cat", type=int, default=12)
    ap.add_argument("--test_per_cat", type=int, default=20)
    ap.add_argument("--train_seed", type=int, default=44001)
    ap.add_argument("--test_seed", type=int, default=99001)
    ap.add_argument("--order_seed", type=int, default=12345, help="broad epoch shuffle; retrieval uses 77")
    # retrieval data
    ap.add_argument("--retrieval_n", type=int, default=80)
    # lm / distill data
    ap.add_argument("--data_dir", default="data/rhac_mix")
    ap.add_argument("--data_name", default="train_tokens.pt")
    ap.add_argument("--val_name", default="val_tokens.pt")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--distill_weight", type=float, default=0.0,
                    help=">0 adds KL(base||student) on lm windows (distillation vs the frozen base)")
    ap.add_argument("--distill_temperature", type=float, default=1.0)
    # memory (Strix Halo)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--stream_weights", action="store_true",
                    help="per-layer weight streaming with eviction (27B on unified memory)")
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--logits_chunk", type=int, default=128)
    ap.add_argument("--gdn_chunk_tokens", type=int, default=2048,
                    help="GDN fallback T-chunking; 0 disables")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--attn_implementation", default=None,
                    choices=["eager", "sdpa", "flex_attention"])
    # io / eval
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--save_name", default="rhac_final.pt")
    ap.add_argument("--partial_save_interval", type=int, default=10)
    ap.add_argument("--resume", default=None, help="partial.pt or adapter state dict to load")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--skip_pre_eval", action="store_true", help="skip the pre eval (resumes)")
    ap.add_argument("--max_eval_rows", type=int, default=0)
    ap.add_argument("--lm_eval_windows", type=int, default=50)
    ap.add_argument("--init_seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    if not args.steps:
        args.steps = {"broad": 72, "retrieval": 24, "lm": 50}[args.task]
    if args.task == "retrieval" and args.order_seed == 12345:
        args.order_seed = 77

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    history_dtype = getattr(torch, args.history_dtype)
    adapter_dtype = getattr(torch, args.adapter_dtype)

    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "checkpoints" / "RHAC" / f"{model_name}-K{args.K}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evals").mkdir(exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    def log(rec: dict):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"RHAC trainer | task={args.task} model={args.model} output={output_dir}", flush=True)

    if args.gdn_chunk_tokens:
        install_gdn_t_chunking(args.gdn_chunk_tokens)

    t0 = time.time()
    model = build_model(args, dtype)
    print(f"  model built in {time.time()-t0:.1f}s", flush=True)

    if args.arch == "postdown":
        from .postdown import attach_postdown, conditioner_state_dict, trainable_param_count

        if args.lr is None:
            args.lr = 1e-3  # spec §20.5: dense insertion sweep recipe
        ctx = attach_postdown(model.model, args.channels, dtype=adapter_dtype)
        state_dict_fn = conditioner_state_dict
        count_fn = trainable_param_count
        marker = ".residual_conditioner."
        print(f"  post-down residual conditioners attached: C={args.channels} "
              f"trainable={count_fn(model)/1e6:.2f}M", flush=True)
    else:
        from .rhac import adapter_param_count, adapter_state_dict

        if args.lr is None:
            args.lr = 3e-3
        ctx = attach_rhac(
            model.model,
            K=args.K, views=args.views, mix=args.mix, view_seed=args.view_seed,
            route_seed_base=args.route_seed_base, group_chunk=args.group_chunk,
            geo_steps=args.geo_steps, adapter_dtype=adapter_dtype, history_dtype=history_dtype,
        )
        state_dict_fn = adapter_state_dict
        count_fn = adapter_param_count
        marker = ".adapter."
        inter = model.model.layers[0].mlp.gate_proj.out_features
        K = pick_block_size(inter, args.K)
        print(f"  adapters attached: K={K} (inter={inter}) views={args.views} mix={args.mix} "
              f"geo_steps={args.geo_steps} trainable={count_fn(model)/1e6:.2f}M", flush=True)

    trainable = []
    for n, p in model.named_parameters():
        p.requires_grad = marker in n
        if p.requires_grad:
            trainable.append(p)
    print(f"  trainable tensors: {len(trainable)} ({count_fn(model)/1e6:.2f}M params)", flush=True)

    streamer = None
    if args.stream_weights:
        # adapters (params + route buffers) live on device permanently; frozen
        # weights keep CPU masters (mmap pages) and stream per layer
        for name, mod in model.named_modules():
            if name.endswith(".adapter") or name.endswith(".residual_conditioner"):
                mod.to(device)
        text = model.model
        stream_modules = [text.embed_tokens, *text.layers, text.norm, model.lm_head]
        streamer = WeightStreamer(stream_modules, device)
        print("  weight streaming enabled (per-layer eviction)", flush=True)
    else:
        resident_to_device(model, device)

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.resume:
        payload = torch.load(args.resume, weights_only=False, map_location="cpu")
        sd = payload["adapter"] if isinstance(payload, dict) and "adapter" in payload else payload
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  resumed adapter state: {len(sd)} tensors, {len(unexpected)} unexpected", flush=True)

    tok_src = args.tokenizer or ("Qwen/Qwen3.5-0.8B-Base" if args.tiny else args.model)
    base_tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    vocab = model.lm_head.out_features
    if vocab < 32768:
        # tiny fixtures cannot hold real tokenizer ids; clamp into range
        class _ClampedTok:
            def encode(self, s, _base=base_tok, _v=vocab):
                return [t % _v for t in _base.encode(s)]

        tokenizer = _ClampedTok()
    else:
        tokenizer = base_tok

    opt = None
    if not args.eval_only:
        if args.optimizer == "sgd":
            opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        if args.resume and isinstance(payload, dict) and "opt" in payload:
            try:
                opt.load_state_dict(payload["opt"])
                print(f"  optimizer state restored (step {payload.get('step')})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  optimizer state not restored: {e}", flush=True)

    # ---------------- data ----------------
    train_rows = test_rows = None
    lm_loader = None
    val_tokens = None
    if args.task in ("broad", "retrieval", "standard"):
        train_rows, test_rows = build_tasks(args, tokenizer)
        print(f"  task rows: train={len(train_rows)} test={len(test_rows)}", flush=True)
    if args.task in ("lm", "standard"):
        tokens = rdata.load_token_tensor(args.data_dir, args.data_name)
        ds = rdata.TokenWindows(tokens, args.seq_len)
        lm_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, drop_last=True)
        if args.task == "lm":
            val_path = Path(args.data_dir) / args.val_name
            val_tokens = (rdata.load_token_tensor(args.data_dir, args.val_name) if val_path.exists()
                          else tokens[- len(tokens) // 10 :])
            print(f"  lm windows: train={len(ds)} seq_len={args.seq_len} distill_weight={args.distill_weight}", flush=True)
        else:
            print(f"  standard mix: corpus windows={len(ds)} + {len(train_rows)} synthetic rows "
                  f"(80/20 token-interleaved)", flush=True)

    # ---------------- eval ----------------
    def run_eval(tag: str) -> dict:
        if args.task in ("broad", "retrieval", "standard"):
            res = eval_rows(model, test_rows, device, args.max_eval_rows)
        else:
            model.eval()
            ce_tot, kl_tot, ntok = 0.0, 0.0, 0
            with torch.no_grad():
                for wi in range(min(args.lm_eval_windows, max(1, len(val_tokens) // args.seq_len))):
                    x = val_tokens[wi * args.seq_len : (wi + 1) * args.seq_len].unsqueeze(0).to(device)
                    y = val_tokens[wi * args.seq_len + 1 : (wi + 1) * args.seq_len + 1].unsqueeze(0).to(device)
                    ce, kl = window_loss(model, ctx, x, y, args.distill_temperature,
                                         1.0 if args.distill_weight else 0.0, args.logits_chunk)
                    ce_tot += float(ce) * x.numel()
                    kl_tot += float(kl or 0.0) * x.numel()
                    ntok += x.numel()
                    del ce, kl
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            res = {"ppl": float(torch.exp(torch.tensor(ce_tot / max(1, ntok)))), "kl": kl_tot / max(1, ntok)}
        path = output_dir / "evals" / f"{args.task}_{tag}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"{tag.upper()} {json.dumps(res, sort_keys=True)}", flush=True)
        return res

    pre = None if args.skip_pre_eval else (run_eval("pre") if not args.eval_only else run_eval("eval"))
    if streamer is not None:
        streamer.evict_all()

    # ---------------- train ----------------
    losses = []
    if not args.eval_only:
        model.train()
        rng_order = random.Random(args.order_seed)
        stream = []
        if args.task in ("broad", "retrieval", "standard"):
            order = list(range(len(train_rows)))
            while len(stream) < args.steps:
                e = order[:]
                rng_order.shuffle(e)
                stream.extend(e)
            stream = stream[: args.steps]
        lm_iter = iter(lm_loader) if lm_loader is not None else None
        opt_step_time = time.time()
        for st in range(args.steps):
            opt.zero_grad(set_to_none=True)
            if args.task == "standard":
                # 80/20 token-interleave: mostly corpus windows (CE), else a
                # synthetic answer row
                lm_iter = lm_iter if lm_iter is not None else iter(lm_loader)
                if rng_order.random() < 0.8:
                    try:
                        x, y = next(lm_iter)
                    except StopIteration:
                        lm_iter = iter(lm_loader)
                        x, y = next(lm_iter)
                    x, y = x.to(device), y.to(device)
                    ce, _kl = window_loss(model, ctx, x, y, args.distill_temperature,
                                          0.0, args.logits_chunk)
                    loss = ce
                else:
                    loss = answer_loss(model, train_rows[stream[st]], device)
                loss.backward()
                losses.append(float(loss.detach()))
                del loss
            elif args.task in ("broad", "retrieval"):
                loss = answer_loss(model, train_rows[stream[st]], device)
                loss.backward()
                losses.append(float(loss.detach()))
                del loss
            else:
                step_losses = []
                for _ in range(max(1, args.grad_accum_steps)):
                    try:
                        x, y = next(lm_iter)
                    except StopIteration:
                        lm_iter = iter(lm_loader)
                        x, y = next(lm_iter)
                    x, y = x.to(device), y.to(device)
                    ce, kl = window_loss(model, ctx, x, y, args.distill_temperature,
                                         args.distill_weight, args.logits_chunk)
                    total = ce + (args.distill_weight * kl if kl is not None else 0.0)
                    (total / max(1, args.grad_accum_steps)).backward()
                    step_losses.append(float(total.detach()))
                losses.append(sum(step_losses) / len(step_losses))
                del ce, total
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
            if streamer is not None:
                streamer.evict_all()
            gc.collect()
            if st == 0 or (st + 1) % 10 == 0 or st + 1 == args.steps:
                dt = time.time() - opt_step_time
                rec = {"step": st + 1, "loss": round(losses[-1], 5),
                       "avg8": round(sum(losses[-8:]) / min(8, len(losses)), 5),
                       "sec": round(time.time() - t0, 1), "step_sec": round(dt, 2),
                       "rss_gb": round(rss_gb(), 1), "vram_gb": round(vram_gb(), 1)}
                print(f"STEP {rec['step']} loss {rec['loss']} avg8 {rec['avg8']} "
                      f"rss {rec['rss_gb']}G vram {rec['vram_gb']}G (+{rec['step_sec']}s)", flush=True)
                log(rec)
                opt_step_time = time.time()
            if args.partial_save_interval and (st + 1) % args.partial_save_interval == 0 and st + 1 < args.steps:
                save_partial(model, opt, st + 1, output_dir, state_dict_fn)
            if (st + 1) % 10 == 0 and torch.cuda.is_available():
                # release reserved-but-unused allocator segments: the fused
                # backward's varied-size transients fragment the pool and
                # reserved GTT pages hold physical RAM the host cannot spare
                torch.cuda.empty_cache()

        torch.save(state_dict_fn(model), output_dir / args.save_name)
        if args.arch == "postdown":
            cfg_out = {k: v for k, v in vars(args).items()}
            cfg_out.update({"arch": "postdown", "channels": args.channels,
                            "trainable_params": count_fn(model),
                            "train_loss_tail": losses[-12:]})
            (output_dir / "rhac_config.json").write_text(json.dumps(cfg_out, indent=2))
        else:
            try:
                from .peft_integration import export_adapter_repo

                export_adapter_repo(
                    model, output_dir / "peft_adapter", args.model,
                    config_overrides={
                        "K": K, "views": args.views, "mix": args.mix,
                        "view_seed": args.view_seed, "route_seed_base": args.route_seed_base,
                        "geo_steps": args.geo_steps, "group_chunk": args.group_chunk,
                        "adapter_dtype": args.adapter_dtype, "history_dtype": args.history_dtype,
                    },
                )
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: PEFT repo export failed: {e}", flush=True)
            cfg_out = {k: v for k, v in vars(args).items()}
            cfg_out.update({"K_resolved": K, "trainable_params": count_fn(model),
                            "intermediate_size": inter, "train_loss_tail": losses[-12:]})
            (output_dir / "rhac_config.json").write_text(json.dumps(cfg_out, indent=2))
        if args.task == "lm":
            report = {"config": cfg_out, "corpus": rdata.corpus_manifest(args.data_dir)}
            (output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
        print(f"saved adapter checkpoint -> {output_dir / args.save_name}", flush=True)

    post = run_eval("post") if not args.eval_only else None
    if streamer is not None:
        streamer.evict_all()
    log({"pre": pre, "post": post, "seconds": time.time() - t0,
         "trainable_params": count_fn(model)})
    return 0
