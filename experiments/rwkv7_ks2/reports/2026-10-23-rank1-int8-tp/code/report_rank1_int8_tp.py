"""Package measured INT8/Horner/fusion/TP results without data or checkpoint weights."""
import argparse,hashlib,io,json,re,shutil,statistics as st,tarfile,time
from pathlib import Path


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;code=Path(__file__).resolve().parent;repo=code.parents[1]
 read=lambda f:json.loads((r/f).read_text())
 for name in ['final/final_complete.json','operator_checks.json','export_reload_check.json','export_complete.json']:assert read(name)['status']=='PASS',name
 results={p.parent.name:json.loads(p.read_text()) for p in (r/'final').glob('*/results.json')};assert len(results)==8
 modes=['baseline','fp32','fp16','int8'];labels={'baseline':'Original FP32 KS2','fp32':'W8A8 + FP32 nonlinearities','fp16':'W8A8 + FP16 nonlinearities','int8':'W8A8 + INT8 nonlinearities'}
 get=lambda name,mode:next(x for x in results[name]['cases'] if x['nonlinear']==mode)
 mlp=[1.4623653888702393,1.4636157751083374,1.463143229484558];mlpmean=st.mean(mlp)
 quality=[]
 for mode in modes:
  vals=[get(f'confirm_tp4_seed{seed}',mode)['loss'] for seed in (11,22,33)];base=[get(f'confirm_tp4_seed{seed}','baseline')['loss'] for seed in (11,22,33)]
  quality.append(dict(mode=mode,loss=st.mean(vals),sd=st.stdev(vals),delta_from_ks2=st.mean(x-y for x,y in zip(vals,base)),delta_from_mlp=st.mean(vals)-mlpmean,mlp_seed_wins=sum(x<y for x,y in zip(vals,mlp)),values=vals))
 speeds=[]
 for batch in (4,64,256):
  single=f'verified_tp1_b{batch}';multi='confirm_tp4_seed11' if batch==64 else f'verified_tp4_b{batch}'
  row=dict(batch=batch,context=64,baseline=st.median(get(single,'baseline')['ms']),tp1=st.median(get(single,'fp32')['ms']),tp4_fp32=st.median(get(multi,'fp32')['ms']),tp4_fp16=st.median(get(multi,'fp16')['ms']),tp4_int8=st.median(get(multi,'int8')['ms']))
  row['tp4_speedup_vs_original']=row['baseline']/row['tp4_fp32'];row['tp4_speedup_vs_tp1']=row['tp1']/row['tp4_fp32'];speeds.append(row)
 manifest=read('export_tp4_seed11/manifest.json')
 for d in results.values():
  for name,digest in d['source_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
  assert all(c['status']=='PASS' for c in d['cases']) and d['state_dtype']=='float32'
 summary=dict(status='PASS',quality=quality,mlp_loss=mlpmean,speeds=speeds,export=manifest,epoch=time.time(),scope='Inference/export only; no training update timings or new training runs. All three precisions share the same W8A8/Horner/fusion/head+FFN TP implementation.')
 (r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 text=['# Rank-1 KS2: integrated INT8, Horner, fusion and tensor parallelism\n',
 'One integrated inference export of the strongest replicated KS2 model: four layers, a 96D residual stream, and 368 rank-1 spine charts per FFN. The original trained budget remains **73,600 parameters per FFN**. The three precision settings below change KS2 nonlinear evaluation, not the architecture or training. All learned linear maps, embeddings and KS2 coefficients are INT8. **Recurrent state stays FP32 in every setting.**\n',
 '## Quality on the earlier reserved interval\n',
 'The three original 100k checkpoints (seeds 11, 22, 33; larger training pool) are exported without retraining. Loss is nats/byte, using four overlapping T64 tilings of the same reserved interval used by the previous report. The unchanged matched MLP mean is **1.463041**.\n',
 table(['Implementation','Mean loss','Seed SD','Change vs original KS2','Change vs MLP','Wins vs MLP'],[[labels[q['mode']],f"{q['loss']:.6f}",f"{q['sd']:.6f}",f"{q['delta_from_ks2']:+.6f}",f"{q['delta_from_mlp']:+.6f}",f"{q['mlp_seed_wins']}/3"] for q in quality]),
 'INT8 nonlinearities retain the sub-MLP loss result. FP16 is effectively tied with FP32 here; INT8 has a small additional loss cost. FP32 nonlinearities are the default because they offer the best measured speed/quality tradeoff. This default still uses INT8 weights, activations and integer matrix dots.\n',
 '## Measured inference latency\n',
 table(['Batch × context','Original FP32, TP1 ms','W8A8 TP1 ms','W8A8 TP4, FP32 nonlinear ms','TP4, FP16 nonlinear ms','TP4, INT8 nonlinear ms','TP4 / original speedup'],[[f"{v['batch']} × 64",f"{v['baseline']:.3f}",f"{v['tp1']:.3f}",f"{v['tp4_fp32']:.3f}",f"{v['tp4_fp16']:.3f}",f"{v['tp4_int8']:.3f}",f"{v['tp4_speedup_vs_original']:.2f}×"] for v in speeds]),
 'These are captured **forward + cross-entropy evaluation** times, not optimizer steps, autoregressive serving/decode latency, or the earlier ~8 ms training timings. Each sequence starts with zero recurrent state. Five consecutive rounds of 100 graph replays exclude compilation, batch copies and process startup; TP4 records the slowest rank. Cases are not randomized and clocks are not locked, so small differences should not be overinterpreted. Raw rounds remain in the JSON records.\n',
 'The optimized single-GPU path remains preferable for small batches. Head sharding makes TP4 useful at larger batches; communication and replicated work prevent ideal fourfold scaling. Four GPUs accelerate one model here, rather than running independent experiments.\n',
 '## What is fused and quantized\n',
 '- Input projection uses native INT8 × INT8 → INT32 matrix dots. Its epilogue evaluates the rank-1 Hermite polynomial in Horner form, adds the coordinate bypass, applies ReLU², and directly requantizes for the output projection.\n- Hermite coefficients are quantized from the trained checkpoint. Horner coefficient combinations happen inside the kernel; this is not a retrained power-basis model.\n- Small RWKV controllers fuse their input projection, tanh/sigmoid (or identity), and activation quantization.\n- Linear weights use symmetric per-output/group INT8 scales. Activation quantization uses per-token/group scales and round-to-nearest. FFN output groups are 92 features so TP1 and TP4 partition the same quantization groups. Attention output groups are 32 features, aligned with heads.\n- RWKV inference omits the backward-only saved-state history while keeping the live recurrent matrix and arithmetic FP32.\n',
 'The design draws on the repository’s [VNNI KS2 implementation](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks2_cpu/vnni.cpp): prepacked quantized coefficients, integer accumulation, explicit scales, and floating-point nonlinear/state exceptions. The MI100 round-to-nearest and custom-all-reduce patterns come from `../int8-vllm`; its installed gfx908 collective uses uncached staging for coherent graph replay. No serving configuration or sibling repository was modified.\n',
 '## Nonlinear precision scope\n',
 '- **FP32:** fused Horner evaluation and bounded tanh in FP32.\n- **FP16:** the KS2 nonlinear inputs, Horner stages and ReLU² intermediates round to FP16; tanh itself uses an FP32 transcendental internally.\n- **INT8:** quantized tanh inputs/outputs with a 256-entry table, INT8 spine coefficients, signed INT32 Horner accumulation with six fractional guard bits, and integer ReLU² before requantization. The tanh table covers ±3.2 because its INT8 output already saturates near that range.\n',
 'The nonlinear precision switch covers **KS2 geometry and ReLU²**. RWKV controller nonlinearities, normalization, time mixing, scales, residual streams, FP32 collective partials, and recurrent state retain FP32. This is the VNNI-style INT8 contract requested by the user, not an integer-only recurrent network.\n',
 '## Tensor-parallel layout\n',
 'Each GPU owns **92 of the 368 FFN subspaces** in every layer. The unchanged model has three RWKV heads, so attention heads are assigned **[1, 1, 1, 0]** across ranks. The fourth rank still owns its FFN shard and participates in reductions. Per-head controller output rows and the attention output projection are sharded. Embeddings, head, layer norms, and some small controller input projections are replicated. Both attention and FFN output partials are summed into the 96D residual.\n',
 'RCCL is a supported fallback. The reported TP4 timings use the existing vLLM MI100 custom all-reduce with an 8 MiB envelope. The initial FFN-only TP split was slower than optimized TP1 at B64; adding head sharding resolved that limitation.\n',
 '## Checks and artifacts\n',
 'Independent operator checks compare W8A8 products against explicit integer-operand reference products and fused Horner against basis evaluation. The integer nonlinear path matches its independent signed-integer reference. Compiled assembly contains native INT8 dot instructions. Attention/FFN partition checks, graph-versus-eager checks, all three trained checkpoints and all four ranks pass.\n',
 'Floating-point reduction order can change an INT8 rounding decision downstream, so complete TP1 and TP4 logits are not bitwise identical. The component checks bound the summation error, and the full held-out scores evaluate the resulting model behavior. Export reload checks require exact state tensors and exact logits for every rank and all three nonlinear settings.\n',
 '- [Summary and raw timing/quality values](summary.json)\n- [Independent operator checks](operator_checks.json)\n- [Four-shard reload checks](export_reload_check.json)\n- [Export manifest, sizes and hashes](export_manifest.json)\n- [Combined kernel implementation](code/rank1_int8_tp.py)\n- [Head-sharded attention](code/rank1_int8_attention_tp.py)\n- [Export/load helper](code/export_rank1_int8_tp.py)\n- [Exact current implementation sources](source_bundle.tar.gz)\n',
 'The loadable weights remain in the shared workspace at `experiments/rwkv7_ks2/gpu_runs/rank1_int8_tp/export_tp4_seed11/`. Checkpoints and validation text are not published in this report. The source archive extracts into a fresh directory with the original package paths and includes the model builder dependencies. Use the existing ROCm environment; custom collective timings additionally require the tested vLLM MI100 extension.\n',
 '```bash\n# Export the trained checkpoint into four INT8 shards.\npython experiments/rwkv7_ks2/export_rank1_int8_tp.py \\\n  --checkpoint-run PATH_TO_TRAINED_RUN --out NEW_EXPORT_DIRECTORY --tp 4\n\n# Check the integrated implementation against a trained run.\nTORCH_NCCL_ASYNC_ERROR_HANDLING=0 torchrun --standalone --nproc-per-node=4 \\\n  experiments/rwkv7_ks2/check_rank1_int8_tp.py \\\n  --checkpoint-run PATH_TO_TRAINED_RUN --out NEW_RESULTS_DIRECTORY \\\n  --batch 64 --collective custom --full-eval --validation VALIDATION_BYTES\n```\n']
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'REPORT.md').write_text('\n'.join(text));(a.out/'README.md').write_text('[Read the integrated INT8/Horner/fusion/TP report](REPORT.md).\n')
 for f in ['summary.json','operator_checks.json','export_reload_check.json']:shutil.copyfile(r/f,a.out/f)
 shutil.copyfile(r/'export_tp4_seed11/manifest.json',a.out/'export_manifest.json');shutil.copytree(r/'final',a.out/'measurements',ignore=shutil.ignore_patterns('*.log'))
 names=['rank1_int8_tp.py','rank1_int8_attention_tp.py','export_rank1_int8_tp.py','check_rank1_int8_tp.py','check_rank1_int8_export.py','validate_rank1_int8.py','report_rank1_int8_tp.py'];(a.out/'code').mkdir()
 for name in names:shutil.copyfile(code/name,a.out/'code'/name)
 sources=list(code.glob('*.py'))+[p for p in (repo/'experiments/common').rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h')]
 index={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 with tarfile.open(a.out/'source_bundle.tar.gz','w:gz') as archive:
  for p in sources:archive.add(p,arcname=str(p.relative_to(repo)))
  data=json.dumps(index,indent=2).encode();info=tarfile.TarInfo('source_index.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
 with tarfile.open(a.out/'source_bundle.tar.gz') as archive:
  for name,digest in index.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
 for md in a.out.rglob('*.md'):
  for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md.read_text()):
   if not link.startswith(('https://','http://','#')):assert (md.parent/link.split('#')[0]).is_file(),link
 records=[dict(path=str(p.relative_to(a.out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.out.rglob('*')) if p.is_file()]
 (a.out/'package_check.json').write_text(json.dumps(dict(status='PASS',source_files=len(index),files=records,excluded='Checkpoint weights, optimizer states and validation/training text.'),indent=2)+'\n')
 print(json.dumps(dict(status='PASS',report=str(a.out/'REPORT.md'),quality=quality,speeds=speeds)))
if __name__=='__main__':main()
