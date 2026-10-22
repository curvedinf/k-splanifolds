"""Custom GS128 ternary: 26 base-three bytes and one FP16 absmax scale."""
import numpy as np
import torch


def decode(packed):
    p=np.ascontiguousarray(packed)
    scales=p[:,26:].copy().view('<f2').astype(np.float32)
    digits=np.stack([(p[:,:26].astype(np.uint16)//3**i)%3 for i in range(5)],axis=-1)
    return (digits.reshape(-1,130)[:,:128].astype(np.float32)-1)*scales


def roundtrip(weight):
    if weight.dtype!=torch.float32 or weight.device.type!='cpu' or weight.ndim<2 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite CPU FP32 weights with a leading row/chart axis')
    flat=weight.detach().reshape(weight.shape[0],-1).numpy()
    width=flat.shape[1]
    padded=np.pad(flat,((0,0),(0,(-width)%128)))
    blocks=padded.reshape(-1,128)
    scale=np.abs(blocks).max(axis=1,keepdims=True)
    inv=np.divide(np.float32(1),scale,out=np.zeros_like(scale),where=scale!=0)
    z=blocks*inv
    ternary=np.copysign(np.floor(np.abs(z)+np.float32(.5)),z)
    digits=np.pad((ternary+1).astype(np.uint16),((0,0),(0,2)),constant_values=1)
    packed=np.empty((len(blocks),28),dtype=np.uint8)
    packed[:,:26]=(digits.reshape(-1,26,5)*np.array([1,3,9,27,81],dtype=np.uint16)).sum(axis=-1)
    half=scale.astype('<f2')
    packed[:,26:]=half.view(np.uint8)
    result=decode(packed)
    np.testing.assert_array_equal(result,ternary*half.astype(np.float32))
    out=torch.from_numpy(result.reshape(padded.shape)[:,:width].copy()).reshape_as(weight)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite GS128 result')
    return out,dict(weights=weight.numel(),packed_bytes=packed.nbytes,padded_values=padded.size,
                    rms_error=(out-weight).square().mean().sqrt().item(),max_error=(out-weight).abs().max().item())
