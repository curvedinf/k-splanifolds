"""Read separate post-attention and post-FFN main residuals from earlier layers."""
import torch
from torch import nn
from torch.nn import functional as F

from side_stream import SideModel
from spread_projection import FixedInput, make_codes


class HistoryBlock(nn.Module):
    def __init__(self, base, layer):
        super().__init__()
        self.main_width = base.main_width
        self.side_width = base.side_width
        self.output_width = self.main_width + self.side_width
        self.input_width = self.output_width + 2 * self.main_width * layer
        self.attention, self.norm1 = base.attention, base.norm1
        self.ffn = base.ffn
        self.norm2 = nn.LayerNorm(self.input_width)
        with torch.no_grad():
            self.norm2.weight[:self.output_width].copy_(base.norm2.weight)
            self.norm2.bias[:self.output_width].copy_(base.norm2.bias)
        mix = 1 - (torch.arange(self.input_width).float() / self.input_width).pow((1 - layer / 4) ** 4)
        mix[:self.output_width] = base.mix.detach()
        self.mix = nn.Parameter(mix)
        if layer:
            codes, _, _ = make_codes(self.input_width, self.ffn.num_subspaces, 8)
            self.ffn.down = FixedInput(codes)
        # Keep the original 2144D fixed output matrix/gains and all geometry.

    def forward(self, main, side, first_value, history):
        update, first_value = self.attention(self.norm1(main), first_value)
        post_attention = main + update
        combined = torch.cat((post_attention, side), dim=-1)
        expanded = torch.cat((combined, *history), dim=-1) if history else combined
        hidden = self.norm2(expanded)
        hidden = hidden + (F.pad(hidden[:, :-1], (0, 0, 1, 0)) - hidden) * self.mix
        combined = combined + self.ffn(hidden)
        return combined[..., :self.main_width], combined[..., self.main_width:], first_value, post_attention


class HistoryModel(SideModel):
    def __init__(self, seed, side_width=2048, charts=64):
        super().__init__(seed, side_width, charts)
        self.blocks = nn.ModuleList(HistoryBlock(block, i) for i, block in enumerate(self.blocks))

    def forward(self, tokens, initial_side=None):
        main = self.input_norm(self.embedding(tokens))
        side = main.new_zeros(*main.shape[:-1], self.side_width) if initial_side is None else initial_side
        first_value = None
        history = []
        for block in self.blocks:
            main, side, first_value, post_attention = block(main, side, first_value, history)
            history.extend((post_attention, main))
        return self.head(self.output_norm(main))
