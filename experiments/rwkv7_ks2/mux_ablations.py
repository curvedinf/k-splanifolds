"""Independent hypotheses for the raw-mux quality gap; no attention skip."""
import torch
from torch import nn
from torch.nn import functional as F
from raw_mux import RawMuxModel, parameter_breakdown as raw_breakdown
from side_history import HistoryModel
from side_stream import parameter_breakdown as side_breakdown
from spread_projection import FixedInput, FixedOutput, make_codes

DEFAULT_CODE_SEED = 103415
MODES = ('baseline', 'head_skip', 'output_gains', 'persistent_state', 'source_norm', 'demuxed_history')


def normalize_sources(history, raw_width=512):
    pieces = [F.layer_norm(history[..., :96], (96,))]
    for start in range(96, history.shape[-1], 96 + raw_width):
        pieces.append(F.layer_norm(history[..., start:start+96], (96,)))
        pieces.append(F.layer_norm(history[..., start+96:start+96+raw_width], (raw_width,)))
    return torch.cat(pieces, dim=-1)


class AblationBlock(nn.Module):
    def __init__(self, base, mode, raw_width):
        super().__init__()
        self.mode, self.raw_width = mode, raw_width
        self.history_width, self.ffn_input_width = base.history_width, base.ffn_input_width
        for name in ('attention', 'norm1', 'attention_mux', 'ffn', 'norm2', 'mix'):
            setattr(self, name, getattr(base, name))
        if mode == 'output_gains':
            self.raw_gain = nn.Parameter(torch.full((raw_width,), .05))

    def forward(self, history, first_value):
        inputs = normalize_sources(history, self.raw_width) if self.mode == 'source_norm' else history
        view = self.attention_mux(inputs)
        update, first_value = self.attention(self.norm1(view), first_value)
        attention_state = view + update
        current = F.layer_norm(attention_state, (96,)) if self.mode == 'source_norm' else attention_state
        hidden = self.norm2(torch.cat((inputs, current), dim=-1))
        hidden = hidden + (F.pad(hidden[:, :-1], (0,0,1,0)) - hidden) * self.mix
        raw = self.ffn(hidden)
        if self.mode == 'output_gains':
            raw = raw * self.raw_gain
        return attention_state, raw, first_value


class AblationModel(RawMuxModel):
    def __init__(self, seed, charts=64, mode='baseline', code_seed=DEFAULT_CODE_SEED):
        super().__init__(seed, charts)
        self.mode = mode
        if code_seed != DEFAULT_CODE_SEED:
            for b in self.blocks:
                b.attention_mux = FixedInput(make_codes(b.history_width,12,8,code_seed)[0])
                b.ffn.down = FixedInput(make_codes(b.ffn_input_width,charts,8,code_seed)[0])
            self.demux = FixedOutput(make_codes(96,charts,8,code_seed)[0])
            nn.init.ones_(self.demux.weight)
        self.blocks = nn.ModuleList(AblationBlock(b, mode, charts*8) for b in self.blocks)

    def forward(self, tokens):
        history = self.input_norm(self.embedding(tokens))
        first_value, state = None, None
        for block in self.blocks:
            attention_state, raw, first_value = block(history, first_value)
            if self.mode == 'persistent_state':
                state = raw if state is None else state + raw
                raw = state
            history = torch.cat((history, attention_state, raw), dim=-1)
        readout = self.demux(raw)
        if self.mode == 'head_skip':
            readout = readout + attention_state
        return self.head(self.output_norm(readout))


def build_model(seed, charts=64, mode='baseline', code_seed=DEFAULT_CODE_SEED):
    if mode not in MODES:
        raise ValueError(mode)
    if mode != 'demuxed_history':
        return AblationModel(seed, charts, mode, code_seed)
    model = HistoryModel(seed,2048,charts)
    if code_seed != DEFAULT_CODE_SEED:
        output_codes = make_codes(2144,charts,8,code_seed)[0]
        for b in model.blocks:
            b.ffn.down = FixedInput(make_codes(b.input_width,charts,8,code_seed)[0])
            b.ffn.up = FixedOutput(output_codes)
    return model


def parameter_breakdown(model):
    if isinstance(model, HistoryModel):
        return side_breakdown(model)
    result = raw_breakdown(model)
    result['raw_output_gains'] = sum(b.raw_gain.numel() for b in model.blocks if hasattr(b,'raw_gain'))
    result['norms_and_ffn_mix'] -= result['raw_output_gains']
    return result


def make_optimizer(model, control_update_scale=False, ffn_charts=64):
    if isinstance(model,HistoryModel):
        gains = [b.ffn.up.weight for b in model.blocks]
    else:
        gains = [model.demux.weight] + [b.raw_gain for b in model.blocks if hasattr(b,'raw_gain')]
    ids = {id(p) for p in gains}
    scale = 5.5 if control_update_scale else 1.
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in ids],lr_scale=1.,weight_decay=.01,lr=.001),
        dict(params=gains,lr_scale=scale,weight_decay=.01/scale,lr=.001*scale)],lr=.001)
