"""Blind labeling corpus v2. For each (bench, arm, control) pair in wins.json: up to CAP problems per side (largest best@5 gap first),
each folder = statement.md + <letter>.txt (best COMPLETE sample of each side: reasoning head/tail excerpt + full final answer) + meta.
Replicate pairs ([rep]) are included as the negative control: same model, two runs. Topology-invalid ALE pairs are skipped.
blind/key.json maps folder -> letter -> arm (never show the key to labelers)."""
import json, os, random, re, collections, shutil, sys
import pandas as pd
from dump2 import load, ARMS, BENCHES
OUT = os.path.dirname(os.path.abspath(__file__)); B = f"{OUT}/blind"; CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data"
wins = json.load(open(f"{OUT}/wins.json"))
samples = collections.defaultdict(list)
for l in open(f"{OUT}/samples.jsonl"):
    s = json.loads(l); s.pop("code", None); samples[(s["bench"], s["arm"], s["problem"])].append(s)
texts = {}
def text_of(bench, arm, prob, si):
    if (bench, arm) not in texts: texts[(bench, arm)] = load(arm, bench)[0]
    return texts[(bench, arm)].get((prob, si), {}).get("text", "")
stm = {}
for bench, path in [("frontiercs", "frontiercs/full.parquet"), ("frontiercs_research", "frontiercs/research.parquet"), ("alebench", "alebench/full40.parquet")]:
    df = pd.read_parquet(f"{FS}/{path}")
    for _, r in df.iterrows(): stm[(bench, str(r["reward_model"]["ground_truth"]))] = r["prompt"][0]["content"]
def excerpt(text, head=2500, tail=2500):
    i = text.rfind("</think>")
    if i < 0: return "(TRUNCATED: no final answer)\n" + text[:head] + "\n...\n" + text[-tail:]
    th, final = text[:i], text[i + 8:]
    th = th if len(th) <= head + tail + 200 else th[:head] + f"\n\n[... {len(th) - head - tail} chars of reasoning omitted ...]\n\n" + th[-tail:]
    return "=== REASONING (excerpt) ===\n" + th + "\n\n=== FINAL ANSWER (full) ===\n" + final
random.seed(20260914); key = {}
if os.path.exists(B): shutil.rmtree(B)
counts = collections.Counter()
for k, w in wins.items():
    bench, arm, ctrl = k.split("|")
    if not w.get("topo_ok", True): continue
    for side, probs in (("arm", w["arm_wins"]), ("ctrl", w["ctrl_wins"])):
        def gap(p):
            ba = max((s["score"] for s in samples[(bench, arm, p)] if s["in_common"]), default=0); bc = max((s["score"] for s in samples[(bench, ctrl, p)] if s["in_common"]), default=0)
            return abs(ba - bc)
        for p in sorted(probs, key=gap, reverse=True)[:CAP]:
            d = f"{B}/{bench}/{arm}__vs__{ctrl}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', p)}"; os.makedirs(d, exist_ok=True)
            open(f"{d}/statement.md", "w").write(stm.get((bench, p), "(statement not found)"))
            codes = random.sample("ABCDEFGH", 2); m = {}
            for code, a in zip(codes, (arm, ctrl)):
                rs = sorted([s for s in samples[(bench, a, p)] if s["in_common"]], key=lambda s: (-s["score"], not s["complete"]))
                if not rs: continue
                best = rs[0]
                open(f"{d}/{code}.txt", "w").write(f"[score={best['score']:.2f} complete={best['complete']}]\n\n" + excerpt(text_of(bench, a, p, best["sample_idx"])))
                m[code] = dict(arm=a, best_score=best["score"], best_complete=best["complete"], scores=[s["score"] for s in rs], n_complete=sum(s["complete"] for s in rs))
            key[d.replace(B + "/", "")] = dict(winner_side=side, kind=w.get("kind", "main"), **m); counts[(bench, w.get("kind", "main"))] += 1
json.dump(key, open(f"{B}/key.json", "w"), indent=1)
print("problem folders", len(key)); print(dict(counts))
