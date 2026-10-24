"""Remove the final history decoder's unused side output, preserving task logits."""
import types
import torch
from torch import nn
from torch.nn import functional as F


def final_forward(self,main,side,first_value,history):
    update,first_value=self.attention(self.norm1(main),first_value)
    post_attention=main+update
    combined=torch.cat((post_attention,side),dim=-1)
    expanded=torch.cat((combined,*history),dim=-1) if history else combined
    hidden=self.norm2(expanded)
    hidden=hidden+(F.pad(hidden[:,:-1],(0,0,1,0))-hidden)*self.mix
    # Side output has no consumer after this block; all decoder inputs are retained.
    main=post_attention+self.ffn(hidden)
    return main,side,first_value,post_attention


def prune_final_side(model):
    block=model.blocks[-1]
    assert block.side_width>0 and block.ffn.up.matrix.shape[0]==block.output_width
    width=block.main_width
    block.ffn.up.matrix=block.ffn.up.matrix[:width].clone()
    block.ffn.up.weight=nn.Parameter(block.ffn.up.weight[:width].detach().clone())
    block.forward=types.MethodType(final_forward,block)
    return model
