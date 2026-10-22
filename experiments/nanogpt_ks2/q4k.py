"""llama.cpp reference Q4_K bytes and dequantized FP32 values; no activation quantization."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np
import torch

_LIB=None


def library():
    global _LIB
    if _LIB is None:
        root=Path(__file__).parent;src=root/'q4k_ref.c';so=root/'.build/q4k_ref.so'
        if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
            so.parent.mkdir(exist_ok=True)
            subprocess.run(shlex.split(os.environ.get('CC','gcc'))+['-O3','-std=gnu11','-shared','-fPIC','-ffp-contract=off',str(src),'-lm','-o',str(so)],check=True)
        _LIB=ctypes.CDLL(str(so))
        for name in ('quantize_row_q4_K_ref','dequantize_row_q4_K'):
            fn=getattr(_LIB,name);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];fn.restype=None
    return _LIB


def roundtrip(weight):
    if weight.dtype!=torch.float32 or weight.device.type!='cpu' or weight.ndim<2 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite CPU FP32 weights with a leading row/chart axis')
    flat=weight.detach().reshape(weight.shape[0],-1).numpy()
    width=flat.shape[1];padded=np.pad(flat,((0,0),(0,(-width)%256)))
    packed=np.zeros((padded.size//256,144),dtype=np.uint8)
    result=np.empty_like(padded)
    library().quantize_row_q4_K_ref(padded.ctypes.data,packed.ctypes.data,padded.size)
    library().dequantize_row_q4_K(packed.ctypes.data,result.ctypes.data,padded.size)
    decoded=decode(packed).reshape(padded.shape)
    np.testing.assert_array_equal(decoded,result)
    out=torch.from_numpy(result[:,:width].copy()).reshape_as(weight)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite Q4_K result')
    return out,dict(weights=weight.numel(),packed_bytes=packed.nbytes,padded_values=padded.size,
                    rms_error=(out-weight).square().mean().sqrt().item(),max_error=(out-weight).abs().max().item())


def decode(packed):
    """Independent NumPy decoder checks the packed scales/minima and nibble layout."""
    p=np.ascontiguousarray(packed);d=p[:,:2].copy().view('<f2').astype(np.float32).ravel()
    dm=p[:,2:4].copy().view('<f2').astype(np.float32).ravel()
    s=p[:,4:16];out=np.empty((len(p),256),dtype=np.float32)
    for k in range(8):
        scale=(s[:,k]&63) if k<4 else ((s[:,k+4]&15)|((s[:,k-4]>>6)<<4))
        minimum=(s[:,k+4]&63) if k<4 else ((s[:,k+4]>>4)|((s[:,k]>>6)<<4))
        q=p[:,16+(k//2)*32:16+(k//2+1)*32]
        q=(q&15) if k%2==0 else (q>>4)
        out[:,k*32:(k+1)*32]=(d*scale)[:,None]*q.astype(np.float32)-(dm*minimum)[:,None]
    return out
