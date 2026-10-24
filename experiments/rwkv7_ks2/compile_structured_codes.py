"""Compile MI100 kernels without running GPU work or disturbing training timings."""
import argparse,json,math
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from structured_codes_gpu import _mux,_transpose
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for width in (96,2144,2336,2528,2720):
 for name,fn,signatures,constants in [
  ('mux',_mux,{'X':'*fp32','P':'*i16','S':'*i8','Y':'*fp32'},dict(W=width,C=64,SCALE=1/math.sqrt(width),G=triton.next_power_of_2(width//8))),
  ('transpose',_transpose,{'X':'*fp32','I':'*i16','S':'*i8','Y':'*fp32'},dict(W=width,C=64,SCALE=1/math.sqrt(width),CH=64,B=16))]:
  kernel=triton.compile(ASTSource(fn,signature=signatures,constexprs=constants),target=GPUTarget('hip','gfx908',64),options={'num_warps':4,'enable_fp_fusion':False})
  row=dict(width=width,kernel=name,shared=kernel.metadata.shared,hash=kernel.hash);rows.append(row);print(json.dumps(row),flush=True)
a.out.write_text(json.dumps(dict(status='PASS',note='Compile only; no GPU execution or numerical validation.',rows=rows),indent=2)+'\n')
