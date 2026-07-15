#!/usr/bin/env python3
"""Extract verl metric lines from rlfsx training logs -> JSON."""
import re, json, os

LOGS = {
    'start':  ['/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/rlfsx_q35_inst_start_w2-10987150.out'],
    'wd03':   ['/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/rlfsx_cl_wd03_a10_w2-10987156.out'],
    'nom_a5': ['/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/rlfsx_cl_nom_a5_r1-11061396.out'],
    'newmt':  ['/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/rlfsx_cl_newmt_a10_r1-11061398.out'],
    # old wave? base run July-06 (different data wave, kept separately for reference)
    'start_OLD0706': ['/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/rlfsx_q35_inst_start-10728651.out'],
}

KEYS = ['training/global_step','critic/rewards/mean','critic/rewards/max','critic/score/mean',
        'actor/entropy','actor/kl_loss','actor/ppo_kl','actor/kl_coef','actor/grad_norm',
        'actor/pg_loss','actor/pg_clipfrac',
        'response_length/mean','response_length/max','response_length/min','response_length/clip_ratio',
        'critic/advantages/max','critic/advantages/min']

pat = re.compile(r'([a-zA-Z_/@0-9]+):(-?[0-9.]+(?:e-?[0-9]+)?)')

out = {}
for arm, files in LOGS.items():
    rows = {}
    for f in files:
        if not os.path.exists(f): continue
        for line in open(f, errors='replace'):
            if 'critic/rewards/mean' not in line: continue
            kv = dict(pat.findall(line))
            try: step = int(float(kv['training/global_step']))
            except Exception: continue
            rows[step] = {k: float(kv[k]) for k in KEYS if k in kv}
    out[arm] = rows

print(json.dumps(out))
