#!/usr/bin/env python3
# Split build_sft output into r2 training datasets, EXCLUDING agentic (user: agentic still unusable).
# Normalizes every record/turn to the `_u` shape (system, tools, per-turn loss) so HF Arrow is happy.
import json, os
REPO=os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
def norm(ex):
    r={k:v for k,v in ex.items() if not k.startswith('_')}
    r.setdefault('system',''); r.setdefault('tools','')
    for t in r.get('conversations',[]): t.setdefault('loss',True)
    return r
exs=json.load(open('/tmp/sr_build/_sft_examples.json'))
method=[norm(e) for e in exs if e['_kind']=='method']
traj  =[norm(e) for e in exs if e['_kind'] in ('method','traj_full','traj_folded')]
v4    =[norm(e) for e in exs if e['_kind']=='v4']
out={
 'innovation_method_r2.jsonl'        : method,
 'innovation_method_traj_r2.jsonl'   : traj,
 'innovation_methodv4_r2.jsonl'      : method+v4,
 'innovation_method_traj_v4_r2.jsonl': traj+v4,
 'innovation_v4_r2.jsonl'            : v4,
}
os.makedirs('sft/r2',exist_ok=True)
for fn,rows in out.items():
    assert all('tools' in r and 'system' in r for r in rows), fn
    with open(f'sft/r2/{fn}','w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    cpp=sum(1 for r in rows if 'cpp' in "".join(t['value'] for t in r['conversations'] if t['from']=='gpt').lower())
    print(f"{fn:38s} {len(rows):5d} rows")
print("(agentic EXCLUDED)")
