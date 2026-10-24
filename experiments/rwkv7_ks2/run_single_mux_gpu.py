"""Train the corrected one-mux architecture with the established GPU protocol."""
import hashlib
import json
from pathlib import Path
import sys
import run_raw_mux_gpu
from single_mux import SingleMuxModel


def main():
    out = Path(sys.argv[sys.argv.index('--out')+1])
    run_raw_mux_gpu.RawMuxModel = SingleMuxModel
    run_raw_mux_gpu.main()
    path = out/'results.json'
    if not path.exists():return
    result = json.loads(path.read_text())
    result['architecture'] = 'One shared layer-entry mux; 64x8 geometry-only KS2 branch plus one 96D RWKV branch; final full-residual demux'
    result['arguments']['muxes_per_layer'] = 1
    result['coding_protocol'] = dict(charts=64,rank=8,muxes_per_layer=1,
        input_widths=[96,704,1312,1920],mux_output_width=608,ks2_coordinates=512,rwkv_view=96,
        final_demux_input=608,final_demux_output=96,view_seed=103415,attention_view_seed=103479,
        attention_view='One signed/permuted 96-way Hadamard view. Pad final group to 96 when needed; fold and normalize each row. All actual input features enter every coordinate; padded groups can break exact row orthogonality.',
        input='Embedding plus all previous 96D post-attention and raw 512D KS2 states; no detach. Latest previous residual is already included once in this history.',
        branches='One shared LayerNorm then one 608-row fixed matrix multiply. First 512 coordinates undergo coordinate-wise causal time mix then geometry only; final 96 feed normalized RWKV with view+update residual. No same-layer attention-to-KS2 projection.',
        output='The 96 attention channels pass through the decoder unchanged alongside the 512 KS2 channels. Only the final full 608D residual demuxes to 96D, then output norm/head. No intermediate demux or attention pass-through between layers.')
    for name in ('single_mux.py','run_single_mux_gpu.py'):
        result['source_hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
