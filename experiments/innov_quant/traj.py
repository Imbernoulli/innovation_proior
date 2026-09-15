# -*- coding: utf-8 -*-
"""RL 轨迹:step 0(SFT/base 起点)→ s5 → s10 → s15 → s20。
中途检查点的评测已经在盘上,不用重新跑 GPU。
每个 (臂, step, bench) 算:完成率、mean@draw、自多样性(同题 5 次抽样代码的平均两两 4-gram Jaccard)、
技术家族数、代码 LOC。假设:创新先验让探索在 RL 过程中衰减得更慢;base 起点的臂坍塌得更快。"""
import json, glob, os, re, sys, itertools
import numpy as np
from collections import defaultdict
from dump2 import D, sub_of, extract_code, toks, grams, TRUNC_TOK, BENCHES
from recomb import PAT

TRAJ = {   # (family, lineage): {step: arm}
 ("9B","base"):    {0:"base9b_v2c", 10:"rlv5_base_s10", 15:"rlv5_base_s15", 20:"rlv5_base_s20"},
 ("9B","ft01mix"): {0:"ft01mix_a10", 10:"rlv5_ft01mix_a10_s10", 15:"rlv5_ft01mix_a10_s15", 20:"rlv5_ft01mix_a10_s20"},
 ("9B","ft03nm"):  {0:"ft03nm_a20", 10:"rlv5_ft03nm_a20_s10", 15:"rlv5_ft03nm_a20_s15", 20:"rlv5_ft03nm_a20_s20"},
 ("9B","lo32nm"):  {0:"lo32nm_a10", 5:"rlv5_lo32nm_a10_s5", 10:"rlv5_lo32nm_a10_s10", 15:"rlv5_lo32nm_a10_s15", 20:"rlv5_lo32nm_a10_s20"},
 ("4B","base"):    {0:"base4b", 15:"rlv5_4b_base_s15", 20:"rlv5_4b_base_s20"},
 ("4B","ft01mix"): {0:"4b_ft01mix_a10", 15:"rlv5_4b_ft01mix_a10_s15", 20:"rlv5_4b_ft01mix_a10_s20"},
 ("4B","lo32nm"):  {0:"4b_lo32nm_a10", 15:"rlv5_4b_lo32nm_a10_s15", 20:"rlv5_4b_lo32nm_a10_s20"},
}

def load(arm, bench):
    ok = {}
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub_of(bench)}/shard_*/samples.jsonl")):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("data_source") != bench: continue
            if r.get("error"): continue
            m = r.get("metrics") or {}; s = m.get("score", r.get("score"))
            if s is None: continue
            ok[(str(r["ground_truth"]), int(r.get("sample_idx", -1)))] = (
                float(s), r.get("text") or "", r.get("completion_tokens") or 0)
    return ok

rows = []
for (fam, lin), steps in TRAJ.items():
    for step, arm in sorted(steps.items()):
        for bench in BENCHES:
            ok = load(arm, bench)
            if not ok:
                sys.stderr.write("MISSING %s %s\n" % (arm, bench)); continue
            bycode = defaultdict(list)
            n = comp = 0; scores = []; ctoks = []; ntech = []; locs = []
            for (prob, si), (sc, txt, ct) in ok.items():
                n += 1; scores.append(sc); ctoks.append(ct)
                i = txt.rfind("</think>")
                if i < 0: continue
                comp += 1
                code, lang = extract_code(txt[i+8:], bench)
                if not code.strip(): continue
                bycode[prob].append(code)
                ntech.append(sum(1 for p in PAT.values() if p.search(code[:60000])))
                locs.append(sum(1 for l in code.splitlines() if l.strip()))
            sims = []
            for prob, cs in bycode.items():
                if len(cs) < 2: continue
                g = [grams(toks(c)[:4000]) for c in cs]
                v = [len(a & b)/len(a | b) for a, b in itertools.combinations(g, 2) if (a | b)]
                if v: sims.append(float(np.mean(v)))
            rows.append(dict(fam=fam, lineage=lin, step=step, arm=arm, bench=bench, n=n,
                             p_complete=comp/max(1,n), mean_score=float(np.mean(scores)),
                             med_ctok=float(np.median(ctoks)),
                             self_sim=(float(np.mean(sims)) if sims else float("nan")), n_probs_sim=len(sims),
                             mean_ntech=(float(np.mean(ntech)) if ntech else float("nan")),
                             med_loc=(float(np.median(locs)) if locs else float("nan"))))
            sys.stderr.write("%s %s %s done\n" % (lin, step, bench))
import csv
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "traj_metrics.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("rows", len(rows))
