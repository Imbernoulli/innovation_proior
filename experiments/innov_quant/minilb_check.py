"""校验:活排行榜里还留着完整 final 行的格子,用最小排行榜能否复现 as-run 分数。"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minilb as M
D="/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
ARMS=["base9b_v2c","ft01mix_a10","rlv5_base_s20","rlv5_ft01mix_a10_s20",
      "base4b","4b_ft01mix_a10","rlv5_4b_base_s20","rlv5_4b_ft01mix_a10_s20"]
def load(a):
    s={}
    for d in (f"cc_mls21_{a}_al1", f"cc_mls21_{a}_al1-fix"):
        p=f"{D}/{d}/summary.json"
        if not os.path.exists(p): continue
        j=json.load(open(p)); ts=j.get("tasks",j); ts=list(ts.values()) if isinstance(ts,dict) else ts
        for t in ts: s.setdefault(t["task"],{})[d[-3:]]=t.get("score")
    return s
M.build(force="--rebuild" in sys.argv)
ok=bad=skip=0; diffs=[]
for a in ARMS:
    model=f"vllm/{a}_al1"
    for task,sc in load(a).items():
        rows=[r for r in M.live_rows(task,model) if str(r.get("is_final","")).lower()=="true"]
        rows=[r for r in rows if M.metrics_of(r)]
        if not rows: skip+=1; continue
        r=max(rows,key=lambda x:(len(M.metrics_of(x)), x.get("timestamp","")))
        got=M.score_row(task,model,M.metrics_of(r))
        want=sc.get("fix", sc.get("al1"))
        if want is None: skip+=1; continue
        if got is not None and abs(got-want)<1e-9: ok+=1
        else: bad+=1; diffs.append((a,task,want,got))
print(f"可校验格子: 复现 {ok} / 不符 {bad} / 无完整 final 行跳过 {skip}")
for a,t,w,g in diffs[:25]: print(f"  {a:<24}{t:<42} as-run={w} minilb={g}")
