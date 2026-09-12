"""Per MLS run: n_scored, agent_failed (context overflow) tasks, zero tasks split into crash (settings empty) vs clamp."""
import json,re,sys,glob,os
D="/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
def classify(tag):
    d=json.load(open(f"{D}/cc_mls21_{tag}/summary.json")); t={x['task']:x for x in d['tasks']}
    af=[k for k,v in t.items() if v['status']=='agent_failed']
    crash=[];clamp=[]
    for k,v in t.items():
        if v['status']=='agent_failed' or (v.get('score') or 0)>0: continue
        log=f"{D}/cc_mls21_{tag}/task_logs/{k}.log"
        txt=open(log,errors='ignore').read() if os.path.exists(log) else ''
        m=re.search(r'### SCORE\n.*?\n(\{.*?\n\})\n',txt,re.S)
        empty=True
        if m:
            try:
                j=json.loads(m.group(1)); empty=(j[k][0].get('settings')==[])
            except Exception: empty=None
        (crash if empty else clamp).append(k)
    ov=[k for k in af if 'maximum context length' in open(f"{D}/cc_mls21_{tag}/task_logs/{k}.log",errors='ignore').read()]
    return dict(tag=tag,n_scored=d['n_scored'],mean=d['mean_score'],agent_failed=af,overflow=len(ov),crash=crash,clamp=clamp)
for tag in sys.argv[1:]:
    r=classify(tag); print(f"{r['tag']:32s} scored={r['n_scored']:2d} mean={r['mean']:.4f} agent_failed={len(r['agent_failed'])}(overflow {r['overflow']}) crash0={len(r['crash'])} clamp0={len(r['clamp'])}")
    print("    agent_failed:",r['agent_failed']); print("    crash0:",r['crash']); print("    clamp0:",r['clamp'])
