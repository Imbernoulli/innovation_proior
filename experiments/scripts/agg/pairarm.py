# pairarm.py -- OUR arm minus THE BASE arm, paired on one identical sample set.
# Comparing each arm's own CI against the anchor is the wrong test: those two CIs
# share the anchor's noise and overlap even when the paired difference is decisive.
# This bootstraps the per-sample difference directly.
#   python3 pairarm.py <ours> <theirs> <frontiercs|alebench|frontiercs_research>
import json, glob, sys, random
R = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
def load(tag, src):
    sub = "research_thinking_32k_vllm" if src == "frontiercs_research" else "thinking_32k_both_vllm"
    ok = {}
    for f in glob.glob(f"{R}/cc_eval_{tag}_{sub}/shard_*/samples.jsonl"):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("data_source") != src or r.get("error"): continue
            s = (r.get("metrics") or {}).get("score", r.get("score"))
            if s is None: continue
            ok[(str(r["ground_truth"]), r.get("sample_idx"))] = float(s)
    return ok
a, b, src = sys.argv[1], sys.argv[2], sys.argv[3]
A, B = load(a, src), load(b, src)
common = sorted(set(A) & set(B), key=str)
d = [A[k] - B[k] for k in common]
random.seed(0); N = len(d); bs = []
for _ in range(20000):
    bs.append(sum(d[random.randrange(N)] for _ in range(N)) / N)
bs.sort()
m = sum(d) / N
win = sum(1 for x in d if x > 0); tie = sum(1 for x in d if x == 0)
print(f"{src}: n={N}  {a} mean={sum(A[k] for k in common)/N:.3f}  {b} mean={sum(B[k] for k in common)/N:.3f}")
print(f"  paired diff ({a} - {b}) = {m:+.3f}   95% CI [{bs[500]:+.3f}, {bs[19500]:+.3f}]")
print(f"  per-sample: ours wins {win}, ties {tie}, theirs wins {N-win-tie}")
print(f"  P(diff>0) over bootstrap = {sum(1 for x in bs if x>0)/len(bs):.3f}")
