"""Head-sharded RWKV adapter for the harness's 32D attention heads.

The model width and number of heads come from the source block. Ranks without
an attention head still participate in reductions. Recurrent state stays FP32.
"""
import copy,math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
if __package__:
    from .ks3_gpu import Int8Linear,Int8Controller,inference_recurrence
else:
    from ks3_gpu import Int8Linear,Int8Controller,inference_recurrence


class HeadParallelAttention(nn.Module):
    def __init__(self,base,rank,world,reducer=None):
        super().__init__();assert base.width==base.heads*32 and base.heads>0
        heads=base.heads
        count=heads//world+int(rank<heads%world);start=(rank*(heads//world)+min(rank,heads%world))*32;end=start+count*32
        self.head_start=start;self.head_end=end;self.heads=count;self.width=count*32;self.world=world;self.reducer=reducer
        if not count:return
        for name in ('read','decay','key','value','rate','gate'):
            self.register_buffer('mix_'+name,getattr(base,'mix_'+name).detach().clone())
        for name in ('decay_bias','rate_bias','value_bias','key_scale','key_rate'):
            self.register_buffer(name,getattr(base,name).detach()[start:end].clone())
        self.register_buffer('bonus',base.bonus.detach()[start//32:end//32].clone())
        for name in ('decay_control','rate_control','gate_control','value_control'):
            control=getattr(base,name)
            if control is None:setattr(self,name,None)
            else:setattr(self,name,Int8Controller(SimpleNamespace(down=control.down,up=SimpleNamespace(weight=control.up.weight[start:end]),activation=control.activation)))
        for name in ('read','key','value'):setattr(self,name,Int8Linear(getattr(base,name).weight[start:end]))
        self.output=Int8Linear(base.output.weight[:,start:end],group=32)
        self.norm=nn.GroupNorm(count,self.width,eps=base.norm.eps).to(base.norm.weight.device)
        with torch.no_grad():self.norm.weight.copy_(base.norm.weight[start:end]);self.norm.bias.copy_(base.norm.bias[start:end])
    def forward(self,inputs,first_value):
        batch,length,_=inputs.shape
        if self.heads:
            shift=F.pad(inputs[:,:-1],(0,0,1,0))-inputs
            mixed={name:inputs+shift*getattr(self,'mix_'+name) for name in ('read','decay','key','value','rate','gate')}
            read=self.read(mixed['read']);decay=torch.exp(-torch.sigmoid(self.decay_bias+self.decay_control(mixed['decay']))*math.exp(-.5))
            rate=torch.sigmoid(self.rate_bias+self.rate_control(mixed['rate']));key=self.key(mixed['key']);value=self.value(mixed['value'])
            if first_value is None:first_value=value
            else:value=value+(first_value-value)*torch.sigmoid(self.value_bias+self.value_control(mixed['value']))
            shape=(batch,length,self.heads,32);nk=F.normalize((key*self.key_scale).reshape(shape),dim=-1)
            key=key*(1+(rate-1)*self.key_rate)
            read,decay,key,value,rate=[x.reshape(shape) for x in (read,decay,key,value,rate)]
            out=inference_recurrence(read,decay,key,value,nk,rate)
            out=self.norm(out.reshape(batch*length,self.width)).reshape(batch,length,self.width)
            out=out+((read*key*self.bonus).sum(-1,keepdim=True)*value).reshape(batch,length,self.width)
            out=self.output(out*self.gate_control(mixed['gate']))
        else:out=torch.zeros_like(inputs)
        if self.world>1:
            if self.reducer is None:dist.all_reduce(out)
            else:out=self.reducer(out)
        return out,first_value
