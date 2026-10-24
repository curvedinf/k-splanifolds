"""Add the previous post-attention state to each new 96D Hadamard view."""
from raw_mux import RawMuxModel
from spread_projection import FixedInput


class AttentionSkipMux(FixedInput):
    def __init__(self, original, raw_width):
        super().__init__(original.weight.reshape(12, 8, -1))
        self.raw_width = raw_width
        self.skip_scale = 1.0

    def forward(self, history):
        view = super().forward(history)
        # History ends with [previous post-attention 96D, previous raw KS2].
        previous_attention = history[..., -(self.raw_width + 96):-self.raw_width]
        return view + self.skip_scale * previous_attention


class RawSkipModel(RawMuxModel):
    def __init__(self, seed, charts=64):
        super().__init__(seed, charts)
        # Layer 1 has no preceding attention state and remains unchanged.
        for block in self.blocks[1:]:
            block.attention_mux = AttentionSkipMux(block.attention_mux, charts * 8)
