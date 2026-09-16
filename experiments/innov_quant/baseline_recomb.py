# -*- coding: utf-8 -*-
"""MLS-Bench:模型提出的方法,是不是题目给的 baselines 的拼接?

和 recomb.py 测的完全不是一回事:
  recomb.py  : FrontierCS/ALE 代码里出现了几个「算法技术家族」(sim_anneal / beam / dp ...),
               参照系是 frontier 解池,跟题目给没给 baseline 无关。
  这个脚本   : MLS 每道题在 task_description 里明写 Reference baselines,并在
               tasks/<t>/edits/*.edit.py 里放了每条 baseline 的参考实现。
               问题是:模型写进 custom_* 的那段代码,是不是就是这些 baseline 的调用/拼接。

指标(每个 (arm, task) 一格):
  B0 acted        : 这一格模型到底动没动手(有没有 edit)。没动手 = 直接交默认脚手架 = 方法就是那条默认 baseline。
  B1 n_base       : 模型写的代码里点到了几条**本题给定**的 baseline
  B2 base_ge1/ge2 : 点到 >=1 / >=2 条(>=2 = 字面意义的「baseline 组合」)
  B3 sim_max      : 模型写的代码 与 每条 baseline 参考实现 的 token Jaccard 的最大值
  B4 novel_frac   : 写的代码里,不含任何给定 baseline 名字的实义行占比

数据源:outputs/cc_mls21_<arm>/task_logs/<task>.log 里的 `Step N  edit` 块,
      取 `+  k | ` 开头的新增行 = 模型写进去的代码。
"""
import os, re, sys, glob, json, itertools
from collections import defaultdict, Counter

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
TASKS = ("causal-discovery-discrete causal-observational-linear-gaussian "
         "causal-observational-linear-non-gaussian causal-observational-nonlinear "
         "causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration "
         "ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting "
         "ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift "
         "ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy "
         "optimization-hyperparameter-search optimization-multi-objective optimization-nas "
         "optimization-online-bandit").split()

# 这两题在所有臂上都是 agent.__init__ 就崩(缺 causal-learn / deap),模型一个 token 没被问过
INFRA_DEAD = {"causal-observational-linear-gaussian", "optimization-multi-objective"}

ARMS = [("base9b_v2c_p1", "9B base"), ("ft01mix_a10_p1", "9B SFT"),
        ("rlv5_base_s20_p1", "9B RL(base)"), ("rlv5_ft01mix_a10_s20_p1", "9B RL(先验)"),
        ("base4b_p1", "4B base"), ("4b_ft01mix_a10_p1", "4B SFT"),
        ("rlv5_4b_base_s20_p1", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20_p1", "4B RL(先验)")]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"^Step\s+(\d+)\s+(\w+)\s*$")
ADD  = re.compile(r"^\+\s*\d+\s\|\s?(.*)$")

# 太短/太常见的 stem,必须以大写形式出现才算命中,否则会把 'de'、'random' 之类全打上
UPPER_ONLY = {"de", "hc", "pc", "ga", "sa", "es", "lp", "mi", "kl"}
GENERIC = {"random", "custom", "base", "default", "mean", "median", "linear", "template"}


class Pat:
    def __init__(self, ci, cs):
        self.ci, self.cs = ci, cs

    def search(self, text):
        if self.ci is not None and self.ci.search(text):
            return True
        if self.cs is not None and self.cs.search(text):
            return True
        return False


def baseline_vocab(task):
    """每条 baseline -> 一组正则。来源:文件名 stem / docstring 首行的名字 / 参考实现里的 CamelCase 导入。"""
    out = {}
    for p in sorted(glob.glob(f"{D}/mlsroot/tasks/{task}/edits/*.edit.py")):
        stem = os.path.basename(p)[:-len(".edit.py")]
        if stem in GENERIC:
            continue
        ci, cs = [], []
        src = open(p, encoding="utf-8", errors="replace").read()
        # 1) 文件名 stem(下划线也允许写成连字符或直接连写)
        st = re.escape(stem).replace(r"\_", r"[ _\-]?")
        if stem.lower() in UPPER_ONLY:
            cs.append(r"\b" + stem.upper() + r"\b")
        elif len(stem) >= 3:
            ci.append(r"\b" + st + r"\b")
        # 2) docstring 首行 "<NAME> baseline for ..."
        m = re.match(r'"""\s*([A-Za-z0-9_\-\+ ]{2,40}?)\s+baseline', src)
        if m:
            nm = m.group(1).strip()
            if nm.lower() not in GENERIC and len(nm) >= 3:
                ci.append(r"\b" + re.escape(nm).replace(r"\ ", r"[ _\-]?") + r"\b")
        # 3) 参考实现里 from ... import CamelCase 的类名
        for mm in re.finditer(r"from\s+[\w\.]+\s+import\s+([A-Za-z_, ]+)", src):
            for nm in mm.group(1).split(","):
                nm = nm.strip()
                if re.match(r"^[A-Z][A-Za-z]{3,}$", nm) and nm.lower() not in GENERIC:
                    cs.append(r"\b" + re.escape(nm) + r"\b")
        if ci or cs:
            out[stem] = Pat(re.compile("|".join(sorted(set(ci))), re.I) if ci else None,
                            re.compile("|".join(sorted(set(cs)))) if cs else None)
    return out


def ref_tokens(task):
    out = {}
    for p in sorted(glob.glob(f"{D}/mlsroot/tasks/{task}/edits/*.edit.py")):
        stem = os.path.basename(p)[:-len(".edit.py")]
        src = open(p, encoding="utf-8", errors="replace").read()
        out[stem] = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", src))
    return out


def tmpl_tokens(task):
    """题目自带的空脚手架。跟它的相似度 = 「写得像不像这道题的惯用法」,
    是 sim_max 的对照项:只有 sim_max 明显高过 sim_tmpl,才谈得上「贴着某条 baseline 写」。"""
    p = f"{D}/mlsroot/tasks/{task}/edits/custom_template.py"
    if not os.path.exists(p):
        return set()
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}",
                          open(p, encoding="utf-8", errors="replace").read()))


def added_code(logpath):
    """返回 (是否出现过 edit 动作, 模型新增的代码行 list)"""
    try:
        raw = open(logpath, encoding="utf-8", errors="replace").read()
    except OSError:
        return False, []
    lines = [ANSI.sub("", l) for l in raw.splitlines()]
    cur, acted, adds = None, False, []
    for l in lines:
        m = STEP.match(l.strip())
        if m:
            cur = m.group(2)
            if cur == "edit":
                acted = True
            continue
        if cur == "edit":
            a = ADD.match(l)
            if a:
                adds.append(a.group(1))
    return acted, adds


SIG = re.compile(r"^\s*(#|\"\"\"|'''|$)")


def main():
    vocab = {t: baseline_vocab(t) for t in TASKS}
    refs  = {t: ref_tokens(t) for t in TASKS}
    tmpl  = {t: tmpl_tokens(t) for t in TASKS}

    print("## 0. 每道题给了几条 baseline(= tasks/<t>/edits/*.edit.py)\n")
    print("| task | n_baseline | baselines |")
    print("|---|---|---|")
    for t in TASKS:
        v = vocab[t]
        print(f"| {t}{' ⚠缺包' if t in INFRA_DEAD else ''} | {len(v)} | {', '.join(sorted(v))} |")
    print()

    rows = {}
    hitc = Counter()
    for arm, lab in ARMS:
        for t in TASKS:
            acted, adds = added_code(f"{D}/outputs/cc_mls21_{arm}/task_logs/{t}.log")
            code = "\n".join(adds)
            sig = [l for l in adds if not SIG.match(l)]
            # 只留代码行,并把行尾 # 注释砍掉 —— 注释里写 "unlike DBSCAN, we..." 不算用了 DBSCAN
            codeonly = "\n".join(re.sub(r"#.*$", "", l) for l in sig)
            # 强形态:真的 import 了 / 真的当函数调了
            hits, hard = set(), set()
            for name, rx in vocab[t].items():
                if rx.search(code):
                    hits.add(name); hitc[(t, name)] += 1
                if rx.search(codeonly):
                    for l in codeonly.splitlines():
                        if not rx.search(l):
                            continue
                        if re.search(r"^\s*(from|import)\s", l) or re.search(r"\w\s*\(", l):
                            hard.add(name); break
            # 与每条参考实现的 token Jaccard
            ct = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", code))
            sim = 0.0
            if ct:
                for name, rt in refs[t].items():
                    j = len(ct & rt) / max(1, len(ct | rt))
                    sim = max(sim, j)
            simt = (len(ct & tmpl[t]) / max(1, len(ct | tmpl[t]))) if (ct and tmpl[t]) else 0.0
            nb = len(hits)
            nov = 0.0
            if sig:
                bad = sum(1 for l in sig if any(rx.search(l) for rx in vocab[t].values()))
                nov = 1.0 - bad / len(sig)
            rows[(arm, t)] = dict(acted=acted, n_add=len(sig), n_base=nb,
                                  n_hard=len(hard), sim=sim, sim_tmpl=simt,
                                  sim_excess=sim - simt, ntok=len(ct),
                                  novel=nov, hits=sorted(hits), hard=sorted(hard))
    json.dump({f"{a}|{t}": v for (a, t), v in rows.items()},
              open("/scratch/gpfs/CHIJ/ziran/.tmp/claude-374317/"
                   "-scratch-gpfs-CHIJ-bohan-1-innovation-proior/"
                   "20154e8e-f2e8-4272-a550-4f0c059fe5b0/scratchpad/baseline_recomb.json", "w"),
              ensure_ascii=False, indent=1)
    return rows, vocab, hitc


if __name__ == "__main__":
    rows, vocab, hitc = main()
    print("## 1. 词表命中自检(哪条 baseline 被认出来过几次,8 臂 x 21 题 = 168 格)\n")
    print("| task | baseline | 命中格数 |")
    print("|---|---|---|")
    for (t, n), c in hitc.most_common(40):
        print(f"| {t} | {n} | {c} |")


# ============================ 聚合 ============================
import numpy as np
from scipy import stats


def status_map(arm):
    try:
        d = json.load(open(f"{D}/outputs/cc_mls21_{arm}/summary.json"))
    except OSError:
        return {}
    ts = d.get("tasks", d)
    if isinstance(ts, dict):
        ts = list(ts.values())
    out = {}
    for t in ts:
        n = t.get("task") or t.get("task_name") or t.get("name")
        if n:
            out[n] = t.get("status")
    return out


def report(rows, vocab):
    st = {a: status_map(a) for a, _ in ARMS}
    VALID = {}   # (arm,task) -> bool,agent 根本没被跑起来的格子剔掉
    for a, _ in ARMS:
        for t in TASKS:
            VALID[(a, t)] = st[a].get(t) != "agent_failed"

    # sim 的长度混淆:Jaccard 的分母是并集,写得少 => 分母小 => sim 天然偏高。
    # 在所有「动手了」的格子上把 sim 对 log(1+ntok) 做一元回归,用残差。
    pool = [(a, t) for a, _ in ARMS for t in TASKS
            if VALID[(a, t)] and rows[(a, t)]["acted"] and rows[(a, t)]["ntok"] > 0]
    X = np.log1p([rows[k]["ntok"] for k in pool])
    Y = np.array([rows[k]["sim"] for k in pool])
    b, a0 = np.polyfit(X, Y, 1)
    rr = stats.pearsonr(X, Y)
    print(f"\n> sim 对 log(1+ntok) 的拟合:sim = {a0:+.4f} {b:+.4f}·log(1+ntok),"
          f"n={len(pool)},r={rr[0]:+.3f},p={rr[1]:.2g}。下表 `sim_resid` = sim − 拟合值。\n")
    for k in pool:
        rows[k]["sim_resid"] = rows[k]["sim"] - (a0 + b * np.log1p(rows[k]["ntok"]))
    for a, _ in ARMS:
        for t in TASKS:
            rows[(a, t)].setdefault("sim_resid", float("nan"))

    print("\n## 2. 逐臂:模型写的方法里,点到了几条**本题给定**的 baseline\n")
    print("| arm | 有效题 | 动手率 | n_base(含注释) | n_hard(真调用) | P(n_hard≥1) | P(n_hard≥2) | sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    for a, lab in ARMS:
        ts = [t for t in TASKS if VALID[(a, t)]]
        r = [rows[(a, t)] for t in ts]
        act = [x for x in r if x["acted"]]
        per[a] = dict(ts=ts, r=r)
        f = lambda k, src: (np.nanmean([x[k] for x in src]) if src else float("nan"))
        print(f"| {lab} | {len(ts)} | {len(act)/max(1,len(ts)):.2f} | "
              f"{f('n_base',act):.2f} | {f('n_hard',act):.2f} | "
              f"{np.mean([x['n_hard']>=1 for x in act]) if act else float('nan'):.2f} | "
              f"{np.mean([x['n_hard']>=2 for x in act]) if act else float('nan'):.2f} | "
              f"{f('sim',act):.3f} | {f('sim_tmpl',act):.3f} | {f('sim_resid',act):+.3f} | {f('novel',act):.3f} | {f('n_add',act):.0f} |")
    print("\n注:`动手率`=21 题里有 edit 动作的比例;没动手 = 直接交默认脚手架,方法**就是**那条默认 baseline。"
          "后面各列只在「动手了」的格子上算。")

    CONTR = [("rlv5_ft01mix_a10_s20_p1", "base9b_v2c_p1", "9B RL(先验) − base"),
             ("rlv5_ft01mix_a10_s20_p1", "rlv5_base_s20_p1", "9B RL(先验) − RL(base)"),
             ("rlv5_ft01mix_a10_s20_p1", "ft01mix_a10_p1", "9B RL(先验) − SFT"),
             ("rlv5_4b_ft01mix_a10_s20_p1", "base4b_p1", "4B RL(先验) − base"),
             ("rlv5_4b_ft01mix_a10_s20_p1", "rlv5_4b_base_s20_p1", "4B RL(先验) − RL(base)"),
             ("rlv5_4b_ft01mix_a10_s20_p1", "4b_ft01mix_a10_p1", "4B RL(先验) − SFT")]
    METRICS = [("n_hard", "真 import/调用的 baseline 条数(低=好)"),
               ("n_base", "点到的 baseline 条数,含注释(低=好)"),
               ("sim", "与参考实现的最大 Jaccard(低=好)"),
               ("sim_tmpl", "与空脚手架模板的 Jaccard(对照项,不是结论)"),
               ("sim_excess", "sim − sim_tmpl:超出惯用法的那部分相似(低=好)"),
               ("sim_resid", "sim 对代码长度做回归后的残差(低=好)★这条才是干净的"),
               ("ntok", "写进去的不同标识符数(对照项)"),
               ("novel", "不含 baseline 名的实义行占比(高=好)")]
    print("\n## 3. 配对对照(同题配对,两臂都动手的题才进)\n")
    for key, desc in METRICS:
        print(f"\n### {key} — {desc}\n")
        print("| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |")
        print("|---|---|---|---|---|---|")
        for A, B, lab in CONTR:
            pa, pb = [], []
            for t in TASKS:
                if not (VALID[(A, t)] and VALID[(B, t)]):
                    continue
                ra, rb = rows[(A, t)], rows[(B, t)]
                if not (ra["acted"] and rb["acted"]):
                    continue
                va, vb = ra[key], rb[key]
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                pa.append(va); pb.append(vb)
            n = len(pa)
            if n < 3:
                print(f"| {lab} | {n} | — | — | — | ⚠样本不足 |")
                continue
            d = np.array(pa) - np.array(pb)
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
            try:
                wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except Exception:
                wp = float("nan")
            print(f"| {lab} | {n} | {d.mean():+.3f} | {pos}/{neg} | {sp:.4f} | {wp:.4f} |")

    print("\n## 4. 「方法就是 baseline」的最强形态:一行没改就提交\n")
    print("| arm | 有效题 | 没动手题数 | 没动手的题 |")
    print("|---|---|---|---|")
    for a, lab in ARMS:
        ts = per[a]["ts"]
        nz = [t for t in ts if not rows[(a, t)]["acted"]]
        print(f"| {lab} | {len(ts)} | {len(nz)} | {', '.join(nz) if nz else '—'} |")


if __name__ == "__main__":
    report(rows, vocab)
