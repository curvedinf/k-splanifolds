"""Static input/output gains and a rank-8 KS2 dynamic mux-output modulator."""
import torch
from torch import nn
from torch.nn import functional as F
from single_mux import SingleMuxModel, SharedMux, GeometryOnly
from spread_projection import make_codes
from raw_mux import parameter_breakdown as original_breakdown


class GainedMux(SharedMux):
    def __init__(self, original, placement='output'):
        nn.Module.__init__(self)
        if placement not in ('input','output','dynamic'):
            raise ValueError(placement)
        self.placement = placement
        self.main_width = original.weight.shape[0]
        weight = original.weight.clone()
        if placement=='dynamic':
            # Append an independent view without changing the original 608 rows.
            extra = make_codes(weight.shape[1],1,8,seed=103480)[0].flatten(0,1)
            weight = torch.cat((weight,extra),0)
            self.modulator = GeometryOnly(1)
            # Eight learned vectors, each spanning every main mux output.
            self.gain_vectors = nn.Parameter(torch.randn(8,self.main_width)/8**.5)
        else:
            self.gain = nn.Parameter(torch.ones(weight.shape[1 if placement=='input' else 0]))
        self.register_buffer('weight',weight)

    def forward(self,history):
        if self.placement=='input':
            return F.linear(history*self.gain,self.weight)
        joined = F.linear(history,self.weight)
        if self.placement=='dynamic':
            main,control = joined.split((self.main_width,8),dim=-1)
            activations = self.modulator(control)
            gains = activations @ self.gain_vectors
            gains = gains*gains.square().mean(-1,keepdim=True).clamp_min(1e-8).rsqrt()
            return main*gains
        return joined*self.gain


class MuxGainModel(SingleMuxModel):
    def __init__(self,seed,charts=64,placement='output'):
        super().__init__(seed,charts)
        # Construct additions only after all reference weights have been created.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+3000)
            for block in self.blocks:
                block.mux = GainedMux(block.mux,placement)


def parameter_breakdown(model):
    result = original_breakdown(model)
    placement = model.blocks[0].mux.placement
    if placement=='dynamic':
        result['mux_gain_vectors'] = sum(b.mux.gain_vectors.numel() for b in model.blocks)
        result['mux_modulator_geometry'] = sum(p.numel() for b in model.blocks for p in b.mux.modulator.parameters())
        extra = result['mux_gain_vectors']+result['mux_modulator_geometry']
    else:
        key = 'mux_'+placement+'_gains'
        result[key] = sum(b.mux.gain.numel() for b in model.blocks)
        extra = result[key]
    result['norms_and_ffn_mix'] -= extra
    return result


def make_optimizer(model,control_update_scale=False,ffn_charts=64):
    gains = [model.demux.weight]+[b.mux.gain_vectors if b.mux.placement=='dynamic' else b.mux.gain for b in model.blocks]
    gain_ids = {id(p) for p in gains}
    scale = 5.5 if control_update_scale else 1.
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in gain_ids],lr_scale=1.,weight_decay=.01,lr=.001),
        dict(params=gains,lr_scale=scale,weight_decay=.01/scale,lr=.001*scale)],lr=.001)
