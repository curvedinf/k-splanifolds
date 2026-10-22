"""llama.cpp reference TQ1_0 bytes and dequantized FP32 values; no activation quantization."""
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
        root=Path(__file__).parent;src=root/'tq1_ref.c';so=root/'.build/tq1_ref.so'
        if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
            so.parent.mkdir(exist_ok=True)
            subprocess.run(shlex.split(os.environ.get('CC','gcc'))+['-O3','-std=gnu11','-shared','-fPIC','-ffp-contract=off',str(src),'-lm','-o',str(so)],check=True)
        _LIB=ctypes.CDLL(str(so))
        for name in ('quantize_row_tq1_0_ref','dequantize_row_tq1_0'):
            fn=getattr(_LIB,name);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];fn.restype=None
    return _LIB


def roundtrip(weight):
    if weight.dtype!=torch.float32 or weight.device.type!='cpu' or weight.ndim<2 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite CPU FP32 weights with a leading row/chart axis')
    flat=weight.detach().reshape(weight.shape[0],-1).numpy()
    width=flat.shape[1];padded=np.pad(flat,((0,0),(0,(-width)%256)))
    packed=np.zeros((padded.size//256,54),dtype=np.uint8)
    result=np.empty_like(padded)
    library().quantize_row_tq1_0_ref(padded.ctypes.data,packed.ctypes.data,padded.size)
    library().dequantize_row_tq1_0(packed.ctypes.data,result.ctypes.data,padded.size)
    decoded=decode(packed).reshape(padded.shape)
    np.testing.assert_array_equal(decoded,result)
    out=torch.from_numpy(result[:,:width].copy()).reshape_as(weight)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite TQ1_0 result')
    return out,dict(weights=weight.numel(),packed_bytes=packed.nbytes,padded_values=padded.size,
                    rms_error=(out-weight).square().mean().sqrt().item(),max_error=(out-weight).abs().max().item())


def decode(packed):
    """Independent NumPy decoder of the packed base-three digits."""
    p=np.ascontiguousarray(packed)
    d=p[:,52:54].copy().view('<f2').astype(np.float32).ravel()
    chunks=[]
    for offset,width,count in [(0,32,5),(32,16,5),(48,4,4)]:
        raw=p[:,offset:offset+width].astype(np.uint16)
        for n in range(count):
            trits=(((raw*(3**n))&255)*3)>>8
            chunks.append((trits.astype(np.float32)-np.float32(1))*d[:,None])
    return np.concatenate(chunks,axis=1)
