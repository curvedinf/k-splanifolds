import argparse,json,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from spread_projection import make_codes
from structured_codes import compact_codes,reference_mux,reference_demux

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(6543);rows=[]
for width in (96,2144,2336,2528,2720):
 charts=64;codes,expected_perm,expected_signs=make_codes(width,charts)
 perm,inv,signs=compact_codes(width,charts)
 assert torch.equal(perm,expected_perm) and torch.equal(signs,expected_signs)
 matrix=codes.flatten(0,1);scale=float(matrix[0,0].abs())
 x=torch.randn(3,width,requires_grad=True);u=torch.randn(3,charts*8)
 y=reference_mux(x,perm,signs,scale);target=F.linear(x,matrix)
 torch.testing.assert_close(y,target,atol=6e-6,rtol=2e-5)
 g=torch.autograd.grad((y*u).sum(),x)[0];gt=F.linear(u,matrix.T)
 torch.testing.assert_close(g,gt,atol=6e-6,rtol=2e-5)
 z=reference_demux(u,inv,signs,scale)
 torch.testing.assert_close(z,gt,atol=6e-6,rtol=2e-5)
 row=dict(width=width,charts=charts,mux_error=float((y-target).abs().max()),transpose_error=float((z-gt).abs().max()),gradient_error=float((g-gt).abs().max()),dense_buffer_bytes=matrix.numel()*4,compact_buffer_bytes=perm.numel()*5)
 rows.append(row);print(json.dumps(row),flush=True)
a.out.write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
