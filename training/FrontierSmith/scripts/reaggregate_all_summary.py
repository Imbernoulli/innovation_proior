#!/usr/bin/env python3
# Recover FCS (and real ALE) into summary_all.json from fcsale/summary.json, regardless of
# the fcsale process exit code (ALE infra-fail must not discard a good FCS). Idempotent.
import json, glob, pathlib
for sa in sorted(glob.glob('/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/cc_eval_all_*/summary_all.json')):
    base = pathlib.Path(sa).parent
    try: d = json.load(open(sa))
    except Exception: continue
    fp = base/'fcsale'/'summary.json'
    if not fp.is_file(): continue
    try: fd = json.load(open(fp))
    except Exception: continue
    m = fd.get('metrics', {})
    fcs = (m.get('frontiercs') or {}).get('reward', {}).get('mean@5')
    ale = (m.get('alebench') or {}).get('performance', {}).get('mean@5')
    changed = False
    if fcs is not None and d.get('frontiercs') != fcs:
        d['frontiercs'] = round(fcs, 4); changed = True
    if ale is not None and ale != 0.0 and d.get('alebench') != ale:
        d['alebench'] = round(ale, 4); changed = True
    if changed:
        json.dump(d, open(sa, 'w'), indent=2)
        print(f"  {base.name}: FCS={d.get('frontiercs')} ALE={d.get('alebench')}")
