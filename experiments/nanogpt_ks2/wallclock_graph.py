"""Capture the unchanged transformer forward/backward for its three windows."""
import math
from types import MethodType

import torch
from torch import nn


class _Window(nn.Module):
    def __init__(self,model,forward,window):
        super().__init__()
        self.model=model
        self.original_forward=forward
        self.window=window

    def forward(self,tokens,documents):
        return self.original_forward(tokens,*self.window,documents)


def capture_model(model,length,*,amp=False):
    """One outstanding training forward per model; evaluation remains eager.

    Token/document values and learned weights stay dynamic. YaRN frequencies
    remain live device buffers; each graph captures the exact Python attention
    scale for its window stage. Unexpected windows/scales/shapes fall back.
    """
    original=model.forward
    def forward(*args):
        if amp:
            # Disable the autocast weight cache: weights change after each replay.
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                output=original(*args)
            return output.float()
        return original(*args)
    yarns=(model.yarn,model.yarn_paired_head,model.yarn_wide)
    initial=[y.attn_scale for y in yarns]
    device=next(model.parameters()).device
    tokens=torch.zeros(length,device=device,dtype=torch.long)
    docs=torch.zeros_like(tokens)
    wrappers={};old=(16,16,48)
    try:
        for window in ((16,48),(48,112),(80,176)):
            new=(window[0],window[0],window[1])
            if new!=old:
                for yarn,a,b in zip(yarns,old,new):
                    yarn.attn_scale*=1+.2*math.log(b/a)
            wrapper=_Window(model,forward,window)
            torch.cuda.make_graphed_callables(wrapper,(tokens,docs),
                                            num_warmup_iters=3,allow_unused_input=True)
            wrappers[(window,tuple(y.attn_scale for y in yarns))]=wrapper
            old=new
    finally:
        for yarn,value in zip(yarns,initial):yarn.attn_scale=value
    stats={'replays':0,'fallbacks':0}
    def dispatch(self,input_seq,ws_short,ws_long,seqlens):
        key=((ws_short,ws_long),tuple(y.attn_scale for y in yarns))
        if (self.training and torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda')
                and input_seq.shape==(length,) and seqlens.shape==(length,)
                and input_seq.device==device and seqlens.device==device
                and input_seq.dtype==torch.long and seqlens.dtype==torch.long and key in wrappers):
            stats['replays']+=1
            return wrappers[key](input_seq,seqlens)
        stats['fallbacks']+=1
        return forward(input_seq,ws_short,ws_long,seqlens)
    model.forward=MethodType(dispatch,model)
    model._whole_graph_stats=stats
    return model
