"""Independent operator checks for the W8A8, Horner, and INT8-LUT implementation."""
import argparse,json
from pathlib import Path
import torch
import triton as tr
import run
from study_models_v19 import build_model
from rank1_int8_tp import Int8Linear,Int8Rank1FFN,quantize,_gemm


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 checks=[]
 for m,n,k,g in [(1,96,96,128),(256,368,96,128),(256,96,368,92),(7,32,96,128)]:
  torch.manual_seed(8);x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda');linear=Int8Linear(w,g);q,s=quantize(x,g);actual=linear(x);expected=torch.zeros(m,n,device='cuda')
  for i in range(tr.cdiv(k,g)):expected+=(q[:,i*g:(i+1)*g].float()@linear.qweight[:,i*g:(i+1)*g].float().T)*s[:,i,None]*linear.scales[:,i][None,:]
  torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5);checks.append(dict(kind='W8A8',shape=[m,n,k,g],max_error=float((actual-expected).abs().max())))
 meta=json.loads((a.checkpoint_run/'completed.json').read_text());model=build_model(meta['arguments']['seed'],meta['study_config']).cuda();model.load_state_dict(torch.load(a.checkpoint_run/'checkpoint.pt',weights_only=True,map_location='cuda')['model'])
 with torch.inference_mode():
  for layer,block in enumerate(model.blocks):
   cell=Int8Rank1FFN(block.ffn);torch.manual_seed(900+layer);x=torch.randn(17,96,device='cuda')*.25;q,s=quantize(x);coordinates=cell.down(x).clamp(-65504,65504)
   for nl in [0,2]:
    h=torch.empty(17,368,device='cuda',dtype=torch.int8);hs=torch.empty(17,4,device='cuda')
    compiled=_gemm[(tr.cdiv(17,16),4)](q,s,cell.down.qweight,cell.down.scales,h,hs,cell.spine,cell.spine_scale,cell.warp_scale,cell.warp_shift,cell.tanh_lut,17,368,96,128,16,128,128,True,nl,0,num_warps=4)
    c=cell.spine.float();c0,c1,c2,c3,c4,c5=c.unbind(1)
    if nl==0:
     t=4*torch.tanh(coordinates*.25);z=torch.tanh(cell.warp_scale*(t+cell.warp_shift))
     # Basis evaluation, independently of the fused Horner expression.
     g=(2*t**3-3*t*t+1)*c0+(-2*t**3+3*t*t)*c1+(t**3-2*t*t+t)*c2+(t**3-t*t)*c3+(2*z*z-1)*c4+(4*z**3-3*z)*c5
     ref=torch.relu(coordinates+g*cell.spine_scale).square()
    else:
     ci=cell.spine.int();c0,c1,c2,c3,c4,c5=ci.unbind(1)
     ti=((coordinates*.25).clamp(-3.2,3.2)*(127/3.2)).round().long();tq=cell.tanh_lut[ti+128].int()
     warp=cell.warp_scale*(tq.float()*(4/127)+cell.warp_shift);zi=(warp.clamp(-3.2,3.2)*(127/3.2)).round().long();zq=cell.tanh_lut[zi+128].int()
     div=lambda n:torch.div(n+63,127,rounding_mode='floor')
     poly=(2*c0-2*c1+c2+c3)*64
     poly=div(poly*tq*4)+(-3*c0+3*c1-2*c2-c3)*64
     poly=div(poly*tq*4)+c2*64;poly=div(poly*tq*4)+c0*64
     warped=4*c5*64;warped=div(warped*zq)+2*c4*64;warped=div(warped*zq)-3*c5*64;warped=div(warped*zq)-c4*64
     pre=(coordinates+(poly+warped).float()*(cell.spine_scale/64)).reshape(17,4,92)
     scale=pre.abs().amax(-1).clamp_min(1.27e-10)/127
     v=(pre/scale[...,None]).round().int().clamp_min(0);ref=(v*v).float()*scale[...,None].square();ref=ref.reshape(17,368)
    rq,rs=quantize(ref,92);difference=(h.int()-rq.int()).abs();print(layer,nl,float((hs-rs).abs().max()),int(difference.max()),flush=True);torch.testing.assert_close(hs,rs,atol=1e-6,rtol=2e-5)
    assert int(difference.max())<=1 and float((difference!=0).float().mean())<.002
    asm=compiled.asm['amdgcn'];native_int8=('mfma_i32' in asm and 'i8' in asm) or 'dot4' in asm
    assert native_int8,'Expected native INT8 dot instructions'
    checks.append(dict(kind='fused_Horner' if nl==0 else 'INT8_LUT_integer_Horner',layer=layer,max_code_difference=int(difference.max()),different_codes=int((difference!=0).sum()),native_int8_instructions=native_int8))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n');print('PASS',len(checks),'operator checks')
if __name__=='__main__':main()
