"""Replace selected RWKV dense projections on the demuxed-history baseline."""
import torch
from torch import nn
from torch.nn import functional as F
from side_history import HistoryModel
from side_stream import parameter_breakdown as side_breakdown
from spread_projection import make_codes

PROJECTIONS = ('read','key','value','output',
               'decay_control.down','decay_control.up',
               'rate_control.down','rate_control.up',
               'gate_control.down','gate_control.up',
               'value_control.down','value_control.up')
MODES = ('baseline','all',*PROJECTIONS)
CODE_SEED = 203415


class HadamardMux(nn.Module):
    def __init__(self, original, seed):
        super().__init__()
        assert original.bias is None
        self.in_features,self.out_features = original.in_features,original.out_features
        # Every input enters every coordinate, with independent rank-8 views.
        code = make_codes(self.in_features,self.out_features//8,8,seed=seed)[0].flatten(0,1)
        self.register_buffer('matrix',code)
        # Match each original row's L2 norm, including exact zero-init paths.
        self.gain = nn.Parameter(original.weight.detach().square().sum(1).sqrt())

    def forward(self,x):
        return F.linear(x,self.matrix)*self.gain


def selected_modules(model):
    for layer,block in enumerate(model.blocks):
        for index,name in enumerate(PROJECTIONS):
            parts=name.split('.')
            parent=block.attention
            for part in parts[:-1]:parent=getattr(parent,part)
            if parent is not None:
                yield layer,index,name,parent,parts[-1],getattr(parent,parts[-1])


class RWKVMuxModel(HistoryModel):
    def __init__(self,seed,mode='baseline',charts=64):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(seed,2048,charts)
        self.rwkv_mux_mode=mode
        self.replacements=[]
        for layer,index,name,parent,attribute,original in selected_modules(self):
            if mode!='all' and mode!=name:continue
            code_seed=CODE_SEED+100*layer+index
            mux=HadamardMux(original,code_seed)
            setattr(parent,attribute,mux)
            self.replacements.append(dict(layer=layer,projection=name,input_width=mux.in_features,
                output_width=mux.out_features,code_seed=code_seed,
                removed_dense_parameters=original.weight.numel(),added_gains=mux.gain.numel(),
                initial_zero=bool((mux.gain==0).all())))


def parameter_breakdown(model):
    result=side_breakdown(model)
    # Subdivision of RWKV, not extra terms to add to the total.
    muxes=[m for b in model.blocks for m in b.attention.modules() if isinstance(m,HadamardMux)]
    result['rwkv_components']=dict(mux_gains=sum(m.gain.numel() for m in muxes),
        dense_weights=sum(p.numel() for b in model.blocks for m in b.attention.modules()
                          if isinstance(m,nn.Linear) for p in m.parameters()))
    result['rwkv_components']['other']=result['rwkv']-sum(result['rwkv_components'].values())
    return result


def make_optimizer(model,control_update_scale=False,ffn_charts=64):
    # Keep RWKV gains on the SAME base LR/WD as the replaced dense weights.
    decoder_gains=[b.ffn.up.weight for b in model.blocks]
    selected={id(p) for p in decoder_gains}
    scale=5.5 if control_update_scale else 1.
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in selected],
             lr=.001,lr_scale=1.,weight_decay=.01),
        dict(params=decoder_gains,lr=.001*scale,lr_scale=scale,weight_decay=.01/scale)],lr=.001)
