"""Transfer controls: standard causal softmax attention and selective RWKV gate mux."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from study_models_v4 import build_model as build_v4,make_optimizer,parameter_breakdown

class SoftAttention(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width;self.heads=width//32
        self.qkv=nn.Linear(width,3*width,bias=False);self.output=nn.Linear(width,width,bias=False)
        nn.init.normal_(self.qkv.weight,std=.02);nn.init.zeros_(self.output.weight)
        self.register_buffer('inv_frequency',10000.**(-torch.arange(16).float()/16))
    def forward(self,x,first_value=None):
        batch,length,width=x.shape
        q,k,v=self.qkv(x).reshape(batch,length,3,self.heads,32).permute(2,0,3,1,4).unbind(0)
        angles=torch.arange(length,device=x.device).float()[:,None]*self.inv_frequency[None,:]
        cosine,sine=angles.cos(),angles.sin()
        def rotate(t):
            left,right=t.chunk(2,-1)
            return torch.cat((left*cosine-right*sine,left*sine+right*cosine),-1)
        q,k=rotate(q),rotate(k);scores=(q@k.transpose(-1,-2))/math.sqrt(32)
        positions=torch.arange(length,device=x.device);future=positions[None,:]>positions[:,None]
        weights=scores.masked_fill(future,float('-inf')).softmax(-1)
        out=(weights@v).transpose(1,2).reshape(batch,length,width)
        return self.output(out),None

class SoftBlock(nn.Module):
    def __init__(self,base,width):
        super().__init__();self.norm1,self.norm2=base.norm1,base.norm2;self.attention=SoftAttention(width)
        if hasattr(base,'ffn'):self.ffn=base.ffn
        else:self.key,self.value=base.key,base.value
    def forward(self,x,first_value):
        update,_=self.attention(self.norm1(x));x=x+update;hidden=self.norm2(x)
        update=self.ffn(hidden) if hasattr(self,'ffn') else self.value(F.relu(self.key(hidden)).square())
        return x+update,None

def build_model(seed,cfg):
    model=build_v4(seed,cfg)
    if cfg.get('attention')=='softmax':
        assert cfg.get('family') in ('mlp','dense_ks2') and not cfg.get('gate_mux')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+3000)
            model.blocks=nn.ModuleList(SoftBlock(b,cfg.get('width',96)) for b in model.blocks)
    if cfg.get('gate_mux'):
        from rwkv_mux import HadamardMux,CODE_SEED,PROJECTIONS
        index=PROJECTIONS.index('gate_control.up')
        for layer,b in enumerate(model.blocks):b.attention.gate_control.up=HadamardMux(b.attention.gate_control.up,CODE_SEED+100*layer+index)
    return model
