"""RWKV-7 + 1-bit QKV ROSA hybrid on the selected demuxed-history baseline.

Adapted from the official RWKV-v8/251105_reverse_run.py parallel hybrid.
Use 32 ROSA channels and reduced RWKV controller ranks for a matched budget.
"""
import torch
from torch import nn
from torch.nn import functional as F
from side_history import HistoryModel
from side_stream import parameter_breakdown as side_breakdown
from rosa_gpu import rosa
from rwkv_mux import make_optimizer

MODES=('baseline','reduced_control','hybrid_matched','hybrid_full')


class ROSA(nn.Module):
    def __init__(self,width=96,channels=32):
        super().__init__()
        for name in ('q','k','v'):
            self.register_parameter('mix_'+name,nn.Parameter(torch.zeros(width)))
            setattr(self,name,nn.Linear(width,channels))
        self.emb=nn.Parameter(torch.ones(channels))
        self.output=nn.Linear(channels,width)
        # Preserve the initial residual function for a paired training comparison.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        shift=F.pad(x[:,:-1],(0,0,1,0))-x
        q,k,v=[getattr(self,name)(x+shift*getattr(self,'mix_'+name)) for name in ('q','k','v')]
        return self.output(rosa(q,k,v)*self.emb)


def reduce_controllers(attention):
    for name,rank in [('decay_control',16),('rate_control',16),('value_control',16),('gate_control',40)]:
        control=getattr(attention,name)
        if control is None:continue
        down=nn.Linear(96,rank,bias=False);up=nn.Linear(rank,96,bias=False)
        with torch.no_grad():
            down.weight.copy_(control.down.weight[:rank])
            up.weight.copy_(control.up.weight[:,:rank])
        control.down,control.up=down,up


class ROSAHistoryBlock(nn.Module):
    def __init__(self,base,seed):
        super().__init__()
        for name in ('main_width','side_width','output_width','input_width','attention','norm1','norm2','ffn','mix'):
            setattr(self,name,getattr(base,name))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rosa=ROSA(96,32)
            self.rosa_norm=nn.LayerNorm(96)

    def forward(self,main,side,first_value,history):
        update,first_value=self.attention(self.norm1(main),first_value)
        post_attention=main+update+self.rosa(self.rosa_norm(main))
        combined=torch.cat((post_attention,side),dim=-1)
        expanded=torch.cat((combined,*history),dim=-1) if history else combined
        hidden=self.norm2(expanded)
        hidden=hidden+(F.pad(hidden[:,:-1],(0,0,1,0))-hidden)*self.mix
        combined=combined+self.ffn(hidden)
        return combined[...,:self.main_width],combined[...,self.main_width:],first_value,post_attention


class RWKV8Model(HistoryModel):
    def __init__(self,seed,mode='hybrid_matched',charts=64):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(seed,2048,charts)
        self.rwkv8_mode=mode
        if mode in ('reduced_control','hybrid_matched'):
            for b in self.blocks:reduce_controllers(b.attention)
        if mode in ('hybrid_matched','hybrid_full'):
            self.blocks=nn.ModuleList(ROSAHistoryBlock(b,seed+4000+i) for i,b in enumerate(self.blocks))


def parameter_breakdown(model):
    result=side_breakdown(model)
    result['rosa']=sum(p.numel() for b in model.blocks if hasattr(b,'rosa') for p in b.rosa.parameters())
    result['rosa_norm']=sum(p.numel() for b in model.blocks if hasattr(b,'rosa_norm') for p in b.rosa_norm.parameters())
    result['norms_and_ffn_mix']-=result['rosa']+result['rosa_norm']
    return result
