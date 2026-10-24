"""Offline MI100 compilation for generic-rank/basis KS2 analytic kernels."""
import argparse,json,hashlib
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import ks2_general_gpu as k
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for r in (1,2,4,8,16):
 for kind in (0,1,2):
  constants=dict(N=17,S=3,B=128//r,R=r,KIND=kind)
  for name,ptrs in [('_geometry_fwd',['X','P','Y']),('_geometry_bwd',['X','P','DY','DX','DP']),('_spine_fwd',['X','P','RS','RH','Y']),('_spine_bwd',['X','P','RS','RH','DY','DX','DP'])]:
   for mode in ((0,1,2) if name.endswith('bwd') else (None,)):
    const=dict(constants)
    if mode is not None:const['MODE']=mode
    kernel=triton.compile(ASTSource(getattr(k,name),signature={s:'*fp32' for s in ptrs},constexprs=const),target=GPUTarget('hip','gfx908',64),options={'num_warps':4,'enable_fp_fusion':False})
    rows.append(dict(rank=r,basis=kind,gradient_mode=mode,kernel=name,shared=kernel.metadata.shared,hash=kernel.hash))
  print(json.dumps(dict(rank=r,basis=kind,status='COMPILED')),flush=True)
a.out.write_text(json.dumps(dict(status='PASS',source_sha256=hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),note='Compilation only, not GPU execution.',rows=rows),indent=2)+'\n')
