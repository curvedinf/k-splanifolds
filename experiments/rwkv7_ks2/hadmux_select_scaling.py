"""Select complete three-seed transfer cohorts, then test capacity scaling."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    code = Path(__file__).resolve().parent
    manifest = root/'wave1_transfer_manifest.json'
    spec = json.loads(manifest.read_text())
    deadline = spec['deadline_epoch']
    print(json.dumps(dict(event='waiting_for_all_transfer_cohorts', epoch=time.time())), flush=True)
    while True:
        paths = [Path(spec['out'])/j['id'] for j in spec['jobs']]
        assert not any((p/'failure.json').exists() for p in paths), 'Resolve actual training failure before selection'
        if all((p/'completed.json').exists() for p in paths):
            break
        if time.time() >= deadline:
            raise RuntimeError('Transfer screen did not finish before exploration deadline')
        time.sleep(5)
    groups = []
    records = [json.loads((p/'completed.json').read_text()) for p in paths]
    for name in sorted({r['study_config']['name'] for r in records}):
        cohort = [r for r in records if r['study_config']['name']==name]
        assert sorted(r['arguments']['seed'] for r in cohort)==[11,22,33]
        groups.append(dict(name=name, config=cohort[0]['study_config'],
                           mean=st.mean(r['tiled_evaluation']['loss'] for r in cohort),
                           mean_ms=st.mean(r['training_seconds']*1000/r['arguments']['steps'] for r in cohort),
                           values=[r['tiled_evaluation']['loss'] for r in cohort]))
    groups.sort(key=lambda x:x['mean'])
    full = min((g for g in groups if not g['config'].get('geometry_terms')), key=lambda x:x['mean'])
    rank1 = min((g for g in groups if g['config'].get('geometry_rank')==1 and not g['config'].get('frozen_geometry')), key=lambda x:x['mean'])
    rank8 = min((g for g in groups if g['config'].get('geometry_rank')==8 and g['config'].get('geometry_terms')=='spine' and not g['config'].get('frozen_geometry')), key=lambda x:x['mean'])
    coefficient_cost = 8 + (2 if rank1['config'].get('mux_affine') else 0)
    matched_features = 52480 // coefficient_cost
    assert matched_features % 8 == 0 and matched_features * coefficient_cost == 52480
    definitions = [
        ('scale_full_f1024_s2048', full, 1024, 2048),
        ('scale_r1_f2048_s2048', rank1, 2048, 2048),
        ('scale_r1_matched_s2048', rank1, matched_features, 2048),
        ('scale_r8spine_f2048_s2048', rank8, 2048, 2048),
        ('scale_full_f512_s8192', full, 512, 8192),
        ('scale_r8spine_f2048_s8192', rank8, 2048, 8192),
    ]
    configs = [dict(parent['config'], name=name, features=features, side_width=side)
               for name, parent, features, side in definitions]
    configs.append(dict(name='original_mlp_repeat', family='mlp'))
    selection = dict(epoch=time.time(), criterion='Complete three-seed exploratory mean within full, learned-R1-spine and learned-R8-spine families; no fresh reserved scores.',
                     all_groups=groups, parents=dict(full=full, rank1=rank1, rank8_spine=rank8),
                     matched_rank1_features=matched_features,
                     matched_learned_per_ffn=54624,
                     note='Matched R1 scale exactly matches the full 512-feature R8 FFN learned budget, including per-output gains. The external norms/time mixing are held at the same side/history widths. Wider side/feature tests are architecture changes; code directions may change. Original MLP repeat is identical, not retuned.',
                     definitions=[dict(name=name,parent=parent['name'],features=features,side_width=side) for name,parent,features,side in definitions])
    (root/'scaling_selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    next_spec = dict(out=str(root/'wave2_scaling'), deadline_epoch=deadline,
                     jobs=[dict(id=f'{cfg["name"]}_{seed}',config=cfg,seed=seed,steps=15000)
                           for seed in (11,22,33) for cfg in configs])
    next_manifest = root/'wave2_scaling_manifest.json'
    next_manifest.write_text(json.dumps(next_spec, indent=2)+'\n')
    with (root/'wave2_cpu_check.log').open('w') as log:
        subprocess.run([sys.executable,str(code/'check_hadmux_v2.py'),'--manifest',str(next_manifest),'--cpu'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(event='scaling_selected',epoch=time.time(),parents={k:v['name'] for k,v in selection['parents'].items()},jobs=len(next_spec['jobs']),matched_rank1_features=matched_features)),flush=True)
    command = [sys.executable,str(code/'hadmux_wave_v2.py'),'--manifest',str(next_manifest),
               '--requires-check',str(root.parent/'goal_20261022_232213/compact_code_checks/general_geometry_check.json')]
    with (root/'wave2_dispatch.jsonl').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps(dict(event='scaling_wave_finished',epoch=time.time())),flush=True)


if __name__ == '__main__':
    main()
