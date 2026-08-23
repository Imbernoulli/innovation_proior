#!/usr/bin/env python3
"""Split the build_sft.py output (+ v4) into the LF training datasets, with the `_u` normalization.

build_sft.py writes ONE sft/innovation_sft.jsonl + /tmp/sr_build/_sft_examples.json (carries _kind).
This splits by _kind into method / method+traj / all, concatenates v4, and -- crucially -- ensures
EVERY record has a `system` and a `tools` field (default ""). LF's dataset_info declares a `tools`
column; a record missing the key makes datasets' converter raise KeyError: 'tools' (the failure of
jobs 10244357/8). The old `_u` files carried `tools:""` on every record; this reproduces that.

Writes sft/r1/*.jsonl. Copy to LF-innov/data/ and register (innovation_*_r1) in dataset_info.json.
"""
import json, os
REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

def norm(ex):
    """Strip _-prefixed build keys; reproduce the `_u` shape so HF datasets can build a uniform schema:
      - record level: guarantee `system` and `tools` (default "").
      - turn level:   guarantee EVERY conversation turn has a `loss` key. build_sft only sets `loss` on
        FOLDED turns (False) and the current trained turn (True); unmarked turns (method, traj_full,
        human/observation) get loss=True -- matching the old working innovation_method_traj_u.jsonl
        (gpt:False=folded/True=trained, human:True, observation:True). Without this, methodtraj records
        mix ('from','value') and ('from','loss','value') turn-structs -> Arrow "Couldn't cast array"."""
    r = {k: v for k, v in ex.items() if not k.startswith('_')}
    r.setdefault('system', '')
    r.setdefault('tools', '')
    for t in r.get('conversations', []):
        t.setdefault('loss', True)
    return r

exs = json.load(open('/tmp/sr_build/_sft_examples.json'))
v4  = [norm(json.loads(l)) for l in open('sft/innovation_v4_sft.jsonl')]

method = [norm(e) for e in exs if e['_kind'] == 'method']
traj   = [norm(e) for e in exs if e['_kind'] in ('method', 'traj_full', 'traj_folded')]
allset = [norm(e) for e in exs]

out = {
    'innovation_method_r1.jsonl'        : method,
    'innovation_method_traj_r1.jsonl'   : traj,
    'innovation_sft_r1.jsonl'           : allset,
    'innovation_v4.jsonl'               : v4,
    'innovation_methodv4_r1.jsonl'      : method + v4,
    'innovation_method_traj_v4_r1.jsonl': traj + v4,
}
os.makedirs('sft/r1', exist_ok=True)
for fn, rows in out.items():
    assert all('tools' in r and 'system' in r for r in rows), fn
    with open(f'sft/r1/{fn}', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{fn:38s} {len(rows):5d} rows  (tools+system on all: OK)")
