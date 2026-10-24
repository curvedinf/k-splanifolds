"""One layer-entry mux: 64x8 KS2 coordinates plus one 96D RWKV view."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from side_stream import BaseModel
from experiments.common.ks2_stabilized import KS2StabilizedCell, _coordinates
from spread_projection import make_codes, FixedOutput


def hadamard96():
    # A 48x48 +/-1 Hadamard matrix from quadratic residues modulo 47,
    # doubled to 96x96. Verification is exact in integer-valued arithmetic.
    q = 47
    residues = {i*i % q for i in range(1,q)}
    character = torch.tensor([0.] + [1. if i in residues else -1. for i in range(1,q)])
    indices = torch.arange(q)
    core = character[(indices[:,None]-indices[None,:]) % q] - torch.eye(q)
    h = torch.ones(q+1,q+1); h[1:,1:] = core
    h = torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    assert torch.equal(h @ h.T,96*torch.eye(96))
    return h / math.sqrt(96)


def attention_code(width, seed=103415+64):
    """One 96D signed/permuted Hadamard view, padding the last group if needed."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(width,generator=gen)
    padded = ((width+95)//96)*96
    signs = torch.randint(0,2,(padded,),generator=gen).float()*2-1
    identity = F.pad(torch.eye(width)[:,perm],(0,padded-width))
    mixed = (identity.reshape(width,-1,96) @ hadamard96().T).reshape(width,padded)
    code = ((mixed*signs).reshape(width,-1,96).sum(1)).T.contiguous()
    return code / code.square().sum(1,keepdim=True).sqrt()


class SharedMux(nn.Module):
    def __init__(self,width,charts=64):
        super().__init__()
        ks2 = make_codes(width,charts,8)[0].flatten(0,1)
        self.register_buffer('weight',torch.cat((ks2,attention_code(width)),0))

    def forward(self,history):
        return F.linear(history,self.weight)


class GeometryOnly(KS2StabilizedCell):
    def __init__(self,charts):
        super().__init__(96,96,num_subspaces=charts,subspace_rank=8,geo_steps=1)
        del self.down,self.up

    def forward(self,coords):
        shape = coords.shape
        coords = coords.float().reshape(-1,self.num_subspaces,self.subspace_rank)
        for _ in range(self.geo_steps):
            coords = _coordinates(coords)
            coords = coords + self.geo_step(coords)
        return coords.reshape(shape)


class SingleMuxBlock(nn.Module):
    def __init__(self,base,cell,layer,charts):
        super().__init__()
        self.ks2_width = charts*8
        self.history_width = 96 + layer*(96+self.ks2_width)
        self.input_norm = nn.LayerNorm(self.history_width)
        self.mux = SharedMux(self.history_width,charts)
        self.attention,self.norm1 = base.attention,base.norm1
        self.ffn = cell
        self.mix = nn.Parameter(1-(torch.arange(self.ks2_width).float()/self.ks2_width).pow((1-layer/4)**4))

    def forward(self,history,first_value):
        joined = self.mux(self.input_norm(history))
        coords,view = joined.split((self.ks2_width,96),dim=-1)
        update,first_value = self.attention(self.norm1(view),first_value)
        attention_state = view + update
        # Coordinate-wise causal mixing remains, with no decoder projection.
        coords = coords + (F.pad(coords[:,:-1],(0,0,1,0))-coords)*self.mix
        raw = self.ffn(coords)
        # The attention channels pass through the decoder as residual state.
        return attention_state,raw,first_value


class SingleMuxModel(nn.Module):
    def __init__(self,seed,charts=64):
        super().__init__()
        base = BaseModel(seed,'mlp',width=96,layers=4,fused_cpu=False)
        self.embedding,self.input_norm = base.embedding,base.input_norm
        self.output_norm,self.head = base.output_norm,base.head
        torch.manual_seed(seed+2000)
        self.blocks = nn.ModuleList(SingleMuxBlock(b,GeometryOnly(charts),i,charts)
                                   for i,b in enumerate(base.blocks))
        # Full final residual: 512 KS2 coordinates + 96 attention coordinates.
        self.demux = FixedOutput(make_codes(96,charts+12,8)[0])
        nn.init.ones_(self.demux.weight)

    def forward(self,tokens):
        history = self.input_norm(self.embedding(tokens))
        first_value = None
        for block in self.blocks:
            attention_state,raw,first_value = block(history,first_value)
            history = torch.cat((history,attention_state,raw),dim=-1)
        residual = torch.cat((raw,attention_state),dim=-1)
        return self.head(self.output_norm(self.demux(residual)))
