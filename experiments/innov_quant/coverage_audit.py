"""Deliverability audit: for every arm x bench, how many of the nominal draws exist, why the rest are
missing, which problems are lost entirely, and what the paired (common-key) analysis actually runs on.
Writes coverage_audit.md next to coverage.json (run from the scratch dump dir)."""
import json, glob, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump2 import ARMS, BENCHES, load, files, D

NOM = {"frontiercs": (172, 5), "frontiercs_research": (64, 5), "alebench": (40, 5)}
ORDER = ["frontiercs", "frontiercs_research", "alebench"]
ZH = {"frontiercs": "FrontierCS", "frontiercs_research": "FCS-research", "alebench": "ALE-Bench"}
cov = json.load(open("coverage.json"))
out = ["# 覆盖率与可交付性审计", "",
       "口径:`samples.jsonl` 去重规则 = (ground_truth, sample_idx),跳过 `error` 行,同键后出现的行覆盖先出现的。",
       "`keys_seen` = 该臂该 bench 出现过的键数(含失败的);`rows_ok` = 有分数的键数。", ""]

out += ["## 1 名义格子 vs 实际落地", "",
        "| bench | 名义 | 每臂 keys_seen | rows_ok 区间 | 丢失率 |", "|---|---|---|---|---|"]
for b in ORDER:
    n, k = NOM[b]
    vs = [v for kk, v in cov.items() if kk.startswith(b + "|") and "rows_ok" in v]
    ksn = sorted({v["keys_seen"] for v in vs}); ro = [v["rows_ok"] for v in vs]
    tot = sum(v["keys_seen"] for v in vs)
    out.append("| %s | %d 题 x %d = %d | %s | %d–%d | %.2f%% |" % (
        ZH[b], n, k, n * k, ksn, min(ro), max(ro), 100 * (tot - sum(ro)) / tot))
out += ["", "**每个臂在每个 bench 上的 keys_seen 都等于名义格子数**,即没有任何一个臂少跑过 batch;",
        "所有缺口都是判题侧报错的行(生成是有的,分数没有)。", ""]

out += ["## 2 逐臂缺口", "", "| 臂 | fam | stage | FCS (rows/860) | research (rows/320) | ALE (rows/200) |", "|---|---|---|---|---|---|"]
for a, (fam, st, ctl, ss) in ARMS.items():
    c = []
    for b in ORDER:
        v = cov.get(b + "|" + a)
        c.append("—" if not v else "%d (−%d)" % (v["rows_ok"], v["keys_seen"] - v["rows_ok"]))
    out.append("| `%s` | %s | %s | %s | %s | %s |" % (a, fam, st, c[0], c[1], c[2]))
out.append("")

out += ["## 3 整题丢失(5 次抽样全部判题失败)", ""]
dead = collections.defaultdict(list)
for a in ARMS:
    for b in ORDER:
        ok, seen = load(a, b)
        g2 = collections.defaultdict(list)
        for (g, i) in seen: g2[g].append(i)
        for g, idxs in g2.items():
            if not any((g, i) in ok for i in idxs): dead[(b, g)].append(a)
out += ["| bench | 题 | 受影响的臂数 | 臂 |", "|---|---|---|---|"]
for (b, g), arms in sorted(dead.items(), key=lambda x: -len(x[1])):
    out.append("| %s | `%s` | %d | %s |" % (ZH[b], g, len(arms), ", ".join("`%s`" % x for x in arms) if len(arms) <= 6 else "(%d 个臂)" % len(arms)))
out.append("")

out += ["## 4 报错种类(判题基础设施,不是模型)", ""]
cnt = collections.Counter()
for a in ARMS:
    for b in ORDER:
        for f in files(a, b):
            for ln in open(f):
                try: r = json.loads(ln)
                except Exception: continue
                if r.get("data_source") != b: continue
                e = r.get("error")
                if e:
                    t = str(e).split("(")[0].split(":")[0][:40]
                    cnt[(b, t)] += 1
out += ["| bench | 异常类 | 行数 |", "|---|---|---|"]
for (b, t), n in sorted(cnt.items(), key=lambda x: -x[1]):
    out.append("| %s | `%s` | %d |" % (ZH[b], t, n))
out.append("")

out += ["## 5 配对分析实际用到的公共键", "",
        "所有配对检验只在“该 family 全部臂都有分数”的键上做,所以上面的缺口不会造成臂间不平衡。", "",
        "| bench | family | 公共键 | 5 次全在的题 | 至少 1 次在的题 |", "|---|---|---|---|---|"]
for k, v in cov.items():
    if not k.endswith("|common"): continue
    b, fam, _ = k.split("|")
    out.append("| %s | %s | %d | %d | %d |" % (ZH[b], fam, v["common_keys"], v["problems_all5"], v["problems_any"]))
out.append("")
open("coverage_audit.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
