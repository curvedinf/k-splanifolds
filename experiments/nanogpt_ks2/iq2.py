"""Actual IQ2_XXS reference quantization with explicit uniform importance weights."""
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np
import torch

_LIB=None
_TABLES=None


def library():
    global _LIB
    if _LIB is None:
        root=Path(__file__).parent;src=root/'iq2_ref.c';so=root/'.build/iq2_ref.so'
        if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
            so.parent.mkdir(exist_ok=True)
            subprocess.run(shlex.split(os.environ.get('CC','gcc'))+['-O3','-std=gnu11','-shared','-fPIC','-fopenmp','-DGGML_USE_OPENMP','-ffp-contract=off',str(src),'-lm','-o',str(so)],check=True)
        _LIB=ctypes.CDLL(str(so))
        _LIB.iq2_init.argtypes=[ctypes.c_int];_LIB.iq2_init.restype=None
        _LIB.iq2_init(min(8,torch.get_num_threads()))
        _LIB.quantize_iq2_xxs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
        _LIB.quantize_iq2_xxs.restype=ctypes.c_size_t
        _LIB.dequantize_row_iq2_xxs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
        _LIB.dequantize_row_iq2_xxs.restype=None
    return _LIB


def roundtrip(weight):
    if weight.dtype!=torch.float32 or weight.device.type!='cpu' or weight.ndim<2 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite CPU FP32 weights with leading row/chart axis')
    flat=weight.detach().reshape(weight.shape[0],-1).numpy();width=flat.shape[1]
    padded=np.pad(flat,((0,0),(0,(-width)%256)))
    importance=np.ones(padded.shape[1],dtype=np.float32)
    packed=np.zeros((padded.size//256,66),dtype=np.uint8);result=np.empty_like(padded)
    n=library().quantize_iq2_xxs(padded.ctypes.data,packed.ctypes.data,len(padded),padded.shape[1],importance.ctypes.data)
    assert n==packed.nbytes
    library().dequantize_row_iq2_xxs(packed.ctypes.data,result.ctypes.data,padded.size)
    np.testing.assert_array_equal(decode(packed).reshape(padded.shape),result)
    out=torch.from_numpy(result[:,:width].copy()).reshape_as(weight)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite IQ2_XXS result')
    return out,dict(weights=weight.numel(),packed_bytes=packed.nbytes,padded_values=padded.size,
                    rms_error=(out-weight).square().mean().sqrt().item(),max_error=(out-weight).abs().max().item())


def decode(packed):
    global _TABLES
    if _TABLES is None:_TABLES=json.loads(Path(__file__).with_name('iq2_tables.json').read_text())
    grid=np.array(_TABLES['iq2xxs_grid'],dtype='<u8').view(np.uint8).reshape(256,8)
    signs=np.array(_TABLES['ksigns_iq2xs'],dtype=np.uint8)
    masks=np.array(_TABLES['kmask_iq2xs'],dtype=np.uint8)
    p=np.ascontiguousarray(packed);d=p[:,:2].copy().view('<f2').astype(np.float32).ravel()
    out=np.empty((len(p),256),dtype=np.float32)
    for k in range(8):
        words=p[:,2+k*8:2+(k+1)*8].copy().view('<u4')
        scale=d*(np.float32(.5)+(words[:,1]>>28).astype(np.float32))*np.float32(.25)
        for j in range(4):
            code=(words[:,0]>>(j*8))&255
            sign=signs[(words[:,1]>>(j*7))&127]
            direction=np.where((sign[:,None]&masks)!=0,np.float32(-1),np.float32(1))
            out[:,k*32+j*8:k*32+(j+1)*8]=(scale[:,None]*grid[code].astype(np.float32))*direction
    return out
