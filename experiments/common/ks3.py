"""KS3: learned subspace charts with a bounded, warped Hermite spine.

A standalone, differentiable PyTorch reference for the winning RWKV FFN.
The optimized W8A8 implementation lives alongside this file in ks3_int8.py.
This reference uses ordinary floating-point tensors, without quantization.

Example:
    >>> cell = KS3(96, num_subspaces=368)
    >>> cell(torch.randn(2, 64, 96)).shape
    torch.Size([2, 64, 96])
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["KS3"]


class KS3(nn.Module):
    """Project into subspace charts, transform each chart, then project back.

    Each chart learns six spine coefficients per output coordinate and two warp
    parameters. Its bounded spine coordinate is the mean of the chart inputs. The
    first four coefficients stay in the Hermite basis; Horner evaluation
    changes the arithmetic, not the trainable parameterization.

    Input:  ``(..., in_features)``.
    Output: ``(..., out_features)``; defaults to the input width.
    Parameters: ``num_subspaces * (subspace_rank * (in_features + out_features + 6) + 2)``.

    Normalization, time mixing, and the outer residual connection belong to
    the enclosing model. The coordinate bypass inside this cell is included.
    Geometry runs in FP32 for FP16/BF16 inputs, or FP64 for a double model;
    projections follow PyTorch's normal dtype/autocast behavior. The fused
    W8A8 export separately defaults to FP16 nonlinearities and FP32 state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        num_subspaces: int = 368,
        subspace_rank: int = 1,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (in_features, out_features, num_subspaces, subspace_rank)):
            raise ValueError("Feature widths, subspace count and rank must be positive integers.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_subspaces = num_subspaces
        self.subspace_rank = subspace_rank
        self.features = num_subspaces * subspace_rank
        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(in_features, self.features, bias=False, **factory)
        self.up = nn.Linear(self.features, out_features, bias=False, **factory)
        coefficient_shape = (num_subspaces, 6) if subspace_rank == 1 else (num_subspaces, subspace_rank, 6)
        self.coefficients = nn.Parameter(torch.empty(coefficient_shape, **factory))
        self.raw_scale = nn.Parameter(torch.full((num_subspaces,), 0.75, **factory))
        self.raw_shift = nn.Parameter(torch.zeros(num_subspaces, **factory))
        nn.init.normal_(self.coefficients, std=0.05)

    def forward(self, inputs: Tensor) -> Tensor:
        coordinates = self.down(inputs)
        geometry_dtype = (
            torch.float64 if coordinates.dtype == torch.float64 else torch.float32
        )
        x = coordinates.to(geometry_dtype).clamp(-65504.0, 65504.0)

        x = x.reshape(*inputs.shape[:-1], self.num_subspaces, self.subspace_rank)

        # Bound the cubic coordinate and learn a warp for the two extra terms.
        t = 4.0 * torch.tanh(x.mean(-1, keepdim=True) / 4.0)
        scale = F.softplus(self.raw_scale.to(geometry_dtype))[:, None] + 1e-4
        shift = 0.5 * torch.tanh(self.raw_shift.to(geometry_dtype))[:, None]
        z = torch.tanh(scale * (t + shift))

        # Cubic Hermite spine, evaluated in Horner form.
        p0, p1, p2, p3, p4, p5 = self.coefficients.to(geometry_dtype).reshape(self.num_subspaces, self.subspace_rank, 6).unbind(-1)
        cubic = 2.0 * p0 - 2.0 * p1 + p2 + p3
        quadratic = -3.0 * p0 + 3.0 * p1 - 2.0 * p2 - p3
        spine = ((cubic * t + quadratic) * t + p2) * t + p0

        # p4 * T2(z) + p5 * T3(z), also in Horner form.
        warped = ((4.0 * p5 * z + 2.0 * p4) * z - 3.0 * p5) * z - p4
        features = F.relu(x + spine + warped).square()
        return self.up(features.flatten(-2).to(self.up.weight.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_subspaces={self.num_subspaces}, subspace_rank={self.subspace_rank}"
        )
