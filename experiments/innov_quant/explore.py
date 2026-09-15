# -*- coding: utf-8 -*-
"""创新如果不在最终代码里,就可能在**搜索过程**里:模型在定稿前考虑了几种不同的路子、
放弃了几种、多久换一次思路。前面所有指标都只看 </think> 之后的代码,这里只看 </think> 之前的推理。
逐样本:
  n_reason      推理里点到的不同技术家族数
  n_code        最终代码里的技术家族数
  n_abandon     推理里点到但最终代码里没有的家族数(考虑过又放弃)
  explore_ratio n_reason / max(1, n_code)
  n_alt         “换个思路”类标记词出现次数(Alternatively / Another approach / Instead of / What if / 或者)
  每 10k 字符归一版本各一份(推理长度各臂差很多,必须归一)
"""
import os, re, json, sys
import numpy as np
from collections import defaultdict
from dump2 import ARMS, load, BENCHES, normalize
from recomb import PAT

HERE = os.path.dirname(os.path.abspath(__file__))
ALT = re.compile(r"\balternativ|\banother (approach|idea|way|option|strategy)|\binstead of\b|"
                 r"\bwhat if\b|\blet me (try|instead|reconsider|rethink)|\bon second thought\b|"
                 r"\bbetter (idea|approach)\b|\bdifferent (approach|idea|way)\b", re.I)

import csv as _csv
PARTS = os.path.join(HERE, "explore_parts")
FIELDS = ["bench","fam","arm","stage","problem","sample_idx","score","rlen","n_reason","n_code",
          "n_abandon","explore_ratio","n_alt","n_reason_10k","n_alt_10k","n_abandon_10k"]
JOBS = [(b, a) for b in BENCHES for a in ARMS]
_rot = int(os.environ.get("ROT", "0")) % max(1, len(JOBS))
JOBS = JOBS[_rot:] + JOBS[:_rot]
if os.environ.get("REV"): JOBS = JOBS[::-1]
for bench, arm in JOBS:
        part = os.path.join(PARTS, "%s__%s.csv" % (bench, arm))
        if os.path.exists(part):
            sys.stderr.write("skip %s %s (done)\n" % (bench, arm)); continue
        rows = []
        try: ok, _seen = load(arm, bench)
        except Exception as e:
            sys.stderr.write("skip %s %s: %s\n" % (arm, bench, e)); continue
        if not ok: continue
        fam, stage, ctrl, start = ARMS[arm]
        for k, v in ok.items():
            txt = v.get("text") or ""
            i = txt.rfind("</think>")
            if i < 0: continue                       # 截断样本没有定稿,排除
            reason = txt[:i]
            s = normalize(bench, arm, k, dict(v))
            code = s.get("code") or ""
            if not code.strip(): continue
            tr = set(k2 for k2, p in PAT.items() if p.search(reason[:200000]))
            tc = set(k2 for k2, p in PAT.items() if p.search(code[:60000]))
            L = max(len(reason), 1)
            rows.append(dict(bench=bench, fam=fam, arm=arm, stage=stage,
                             problem=str(k[0]), sample_idx=k[1], score=s["score"],
                             rlen=len(reason), n_reason=len(tr), n_code=len(tc),
                             n_abandon=len(tr - tc), explore_ratio=len(tr)/max(1, len(tc)),
                             n_alt=len(ALT.findall(reason)),
                             n_reason_10k=1e4*len(tr)/L, n_alt_10k=1e4*len(ALT.findall(reason))/L,
                             n_abandon_10k=1e4*len(tr - tc)/L))
        tmp = part + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        os.replace(tmp, part)
        sys.stderr.write("%s %s done (%d)\n" % (bench, arm, len(rows)))

import glob as _g
allr = []
for f in sorted(_g.glob(os.path.join(PARTS, "*.csv"))):
    with open(f) as fh:
        allr.extend(list(_csv.DictReader(fh)))
with open(os.path.join(HERE, "explore_metrics.csv"), "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(allr)
sys.stderr.write("ALLDONE rows %d\n" % len(allr))
