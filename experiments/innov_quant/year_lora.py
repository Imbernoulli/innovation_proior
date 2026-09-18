#!/usr/bin/env python3
"""lora 线的年份扫描 —— FCS-research + MLS-21,**年份线本来就该跟的那两把尺子**。

起因(用户 2026-09-18:「一个是在 2026 年测 Judgment/Taste/Idea;还有一个是使用不同年份的
System Prompt 来测这个 Research 和 MLS」)。年份 × Research/MLS 的数据其实 09-11 就跑完了,
但 `year_research.py` 与 `mls_year.py` 的 ARMS 都只写了 **ft01mix 线**,`year_grid.py` 的
年份曲线也只列八条主臂 —— **lora 线的年份点在盘上是最密的(9B lora 有 11 个年份),
却从来没进过任何一张年份表**。这份把它补上(用户铁律 o:「lora 线要给全」)。

口径逐条抄 `year_research.py` / `mls_lora.py`,免得两张表同一个数对不上(§27):
  - research:`cc_eval_<arm>_y<Y>_research_thinking_32k_vllm/shard_*/samples.jsonl`,
    按 (ground_truth, sample_idx) 去重,后出现的覆盖先出现的(9B 的分片有重复);
    完成率 = `text.rfind("</think>") >= 0`,与 dump2.py 一致。
  - MLS:`cc_mls21_<arm>_y<Y>`(并 `-fix` 补跑,同名题以 -fix 为准),
    **分母恒为 21**,真 0 记 0、缺题也记 0(用户铁律 h)。
  - 对照:NEAR = 2026,FAR = mean(2000, 2100) —— 这两个远年是两条 lora 臂**都有**的点,
    所以 9B 与 4B 用的是同一个对照定义。噪声底 = 同臂 2000 vs 2100(两个都是远年,
    倒 U 假设不预测它们有差)。
  - 曲线一节把每条臂**自己有的**年份点全列出来,并限制到该臂所有年份点的公共题。

★ 半成品防护(§39):任何 `ysweep-`/`mls21-` 作业还在 squeue 里,对应格子整格跳过。
用法:year_lora.py  → 打印并写 year_lora.md
"""
import collections
import glob
import json
import os
import subprocess

import numpy as np

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
HERE = os.path.dirname(os.path.abspath(__file__))
N_MLS = 21
BOOT, SEED = 10000, 0

ARMS = [("lo32nm_a10", "9B SFT lora"),
        ("rlv5_lo32nm_a10_s20", "9B RL lora"),
        ("4b_lo32nm_a10", "4B SFT lora"),
        ("rlv5_4b_lo32nm_a10_s20", "4B RL lora")]
ALL_YEARS = [1800, 1900, 1950, 1975, 2000, 2010, 2025, 2026, 2050, 2075, 2100]
NEAR, FAR = 2026, (2000, 2100)

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


# ---------- research ----------
def load_res(arm, year):
    d = os.path.join(D, f"cc_eval_{arm}_y{year}_research_thinking_32k_vllm")
    if not os.path.isdir(d):
        return None
    seen = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*", "samples.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            seen[(str(r.get("ground_truth")), r.get("sample_idx"))] = r
    if not seen:
        return None
    out = collections.defaultdict(lambda: {"score": [], "done": []})
    for (gt, _), r in seen.items():
        out[gt]["score"].append(float((r.get("metrics") or {}).get("score") or 0.0))
        out[gt]["done"].append(1.0 if (r.get("text") or "").rfind("</think>") >= 0 else 0.0)
    return {k: {m: float(np.mean(v)) for m, v in d2.items()} for k, d2 in out.items()}


# ---------- MLS ----------
def load_mls(arm, year):
    """-> {task: score};分母恒 21 由调用方保证(缺题记 0)。"""
    got = {}
    for d in (f"{D}/cc_mls21_{arm}_y{year}", f"{D}/cc_mls21_{arm}_y{year}-fix"):
        p = f"{d}/summary.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        ts = j.get("tasks", j)
        ts = ts if isinstance(ts, list) else list(ts.values())
        for t in ts:
            got[t["task"]] = t.get("score")
    return got or None


def running():
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    return {ln.strip() for ln in out.splitlines() if ln.strip()}


def boot_ci(d, rng):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    bs = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    se = bs.std(ddof=1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            float(d.mean() / se) if se > 0 else float("nan"))


def paired(store, a_year, b_years, key, rng):
    """a_year − mean(b_years),按题配对。key=None 表示 store 的值就是标量。"""
    if a_year not in store or any(y not in store or store[y] is None for y in b_years):
        return None
    if store[a_year] is None:
        return None
    common = set(store[a_year])
    for y in b_years:
        common &= set(store[y])
    common = sorted(common)
    if len(common) < 5:
        return None
    def val(y, p):
        v = store[y][p]
        return v if key is None else v[key]
    d = [val(a_year, p) - float(np.mean([val(y, p) for y in b_years])) for p in common]
    m, lo, hi, z = boot_ci(d, rng)
    return dict(n=len(common), mean=m, lo=lo, hi=hi, z=z,
                pos=sum(1 for x in d if x > 0), neg=sum(1 for x in d if x < 0))


def row(lbl, r):
    if not r:
        return f"| {lbl} | — | — | — | — | — |"
    star = " ★" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return (f"| {lbl} | {r['n']} | {r['mean']:+.4f}{star} | "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['pos']}/{r['neg']} | {r['z']:+.2f} |")


def main():
    rng = np.random.default_rng(SEED)
    live = running()
    busy = any(j.startswith(("ysweep-", "mls21-")) for j in live)
    emit("# lora 线的年份扫描:FCS-research + MLS-21\n")
    emit("年份只改系统提示里的一个数字(`It is now year <Y>.`),其余一切不变。")
    emit("**这两把尺子才是年份线该跟的**(用户铁律 c);数据 09-11 就在盘上,"
         "只是此前的年份脚本 ARMS 里没有 lora 线。\n")
    if busy:
        emit("> ⚠ 现在有 `ysweep-`/`mls21-` 作业在队列里,相关格子可能是半成品。\n")

    res, mls = {}, {}
    for arm, lbl in ARMS:
        res[arm] = {y: load_res(arm, y) for y in ALL_YEARS}
        res[arm] = {y: v for y, v in res[arm].items() if v}
        mls[arm] = {y: load_mls(arm, y) for y in ALL_YEARS}
        mls[arm] = {y: v for y, v in mls[arm].items() if v}

    emit("## 0. 每条臂有哪些年份点\n")
    emit("| 臂 | bench | " + " | ".join(str(y) for y in ALL_YEARS) + " | 点数 |")
    emit("|---|---|" + "---:|" * (len(ALL_YEARS) + 1))
    for arm, lbl in ARMS:
        cells = [str(len(res[arm][y])) + "题" if y in res[arm] else "—" for y in ALL_YEARS]
        emit(f"| {lbl} `{arm}` | research | " + " | ".join(cells) + f" | **{len(res[arm])}** |")
        cells = [f"{len(mls[arm][y])}/21" if y in mls[arm] else "—" for y in ALL_YEARS]
        emit(f"| {lbl} `{arm}` | MLS | " + " | ".join(cells) + f" | **{len(mls[arm])}** |")
    emit()

    emit("## 1. 逐年原始值(每条臂限制到它自己所有年份点的公共题)\n")
    emit("research = 均分;MLS = 总分/21(**分母恒 21**,缺题记 0)。\n")
    emit("| 臂 | bench | 指标 | " + " | ".join(str(y) for y in ALL_YEARS) + " | 峰值 |")
    emit("|---|---|---|" + "---:|" * (len(ALL_YEARS) + 1))
    for arm, lbl in ARMS:
        if res[arm]:
            common = set.intersection(*[set(v) for v in res[arm].values()])
            for key, name in (("score", "分数"), ("done", "完成率")):
                vals = {y: float(np.mean([res[arm][y][p][key] for p in common])) for y in res[arm]}
                pk = max(vals, key=vals.get)
                cells = [f"{vals[y]:.3f}" if y in vals else "" for y in ALL_YEARS]
                emit(f"| {lbl} | research | {name}(公共 {len(common)} 题) | "
                     + " | ".join(cells) + f" | **{pk}** |")
        if mls[arm]:
            vals = {y: sum(x for x in mls[arm][y].values() if x) / N_MLS for y in mls[arm]}
            pk = max(vals, key=vals.get)
            cells = [(f"{vals[y]:.3f}" + ("" if len(mls[arm][y]) >= N_MLS
                                           else f" ⚠{len(mls[arm][y])}/21"))
                     if y in vals else "" for y in ALL_YEARS]
            emit(f"| {lbl} | MLS | 总分/21 | " + " | ".join(cells) + f" | **{pk}** |")
    emit()

    emit(f"## 2. NEAR({NEAR}) − FAR(mean{FAR}):倒 U 假设预测这一列为正\n")
    emit(f"`{FAR[0]}` 与 `{FAR[1]}` 是两条 lora 臂**都有**的远年点,所以 9B 与 4B 用同一个定义。\n")
    emit("| 臂 | bench / 指标 | n题 | Δ | 95% CI | +/− | Z |")
    emit("|---|---|---:|---:|---|:---:|---:|")
    for arm, lbl in ARMS:
        for key, name in (("score", "research 分数"), ("done", "research 完成率")):
            emit(row(f"{lbl} | {name}", paired(res[arm], NEAR, FAR, key, rng)))
        m = {y: {t: (v or 0.0) for t, v in mls[arm][y].items()} for y in mls[arm]}
        emit(row(f"{lbl} | MLS 分数", paired(m, NEAR, FAR, None, rng)))
    emit()

    emit(f"## 3. 噪声底:同臂 {FAR[0]} vs {FAR[1]}(两个都是远年,倒 U 不预测有差)\n")
    emit("| 臂 | bench / 指标 | n题 | Δ | 95% CI | +/− | Z |")
    emit("|---|---|---:|---:|---|:---:|---:|")
    for arm, lbl in ARMS:
        for key, name in (("score", "research 分数"), ("done", "research 完成率")):
            emit(row(f"{lbl} | {name}", paired(res[arm], FAR[1], (FAR[0],), key, rng)))
        m = {y: {t: (v or 0.0) for t, v in mls[arm][y].items()} for y in mls[arm]}
        emit(row(f"{lbl} | MLS 分数", paired(m, FAR[1], (FAR[0],), None, rng)))
    emit()
    emit("★ = 该格 95% CI 不含 0。第 3 节任何一格出现 ★,说明第 2 节同量级的效应"
         "**不能**归给年份。\n")

    # ---- §4:lora 比对照臂好在哪 ----
    # 用户 2026-09-18:「现在就是我们全面 shift 到这个 arm,要找到这个 arm 相比起
    # base 的好的指标」。年份只改系统提示里的一个数字,对「哪条臂更强」是无关扰动,
    # 所以同一对臂在各个年份点上的对比 ≈ 若干次准复跑(与 year_research.md §5 同法)。
    # 这些点不是独立复跑,所以看的是**方向一致性**,不是把 Z 叠起来那个 p。
    CMP = [("rlv5_lo32nm_a10_s20", "rlv5_base_s20", "9B RL lora − RL(base)"),
           ("rlv5_lo32nm_a10_s20", "base9b_v2c", "9B RL lora − 预训练 base"),
           ("rlv5_4b_lo32nm_a10_s20", "rlv5_4b_base_s20", "4B RL lora − RL(base)"),
           ("rlv5_4b_lo32nm_a10_s20", "base4b", "4B RL lora − 预训练 base")]
    for a, b in {(x[1], None) for x in CMP}:
        if a not in res:
            res[a] = {y: v for y in ALL_YEARS if (v := load_res(a, y))}
            mls[a] = {y: v for y in ALL_YEARS if (v := load_mls(a, y))}
    emit("## 4. ★lora 比对照臂好在哪★:逐年份点(年份点当准复跑)\n")
    emit("年份只改系统提示里的一个数字,对「哪条臂更强」是无关扰动 —— 所以同一对臂在")
    emit("各个年份点上的对比近似若干次复跑。**看方向一致性,不要把这些 Z 叠成一个 p**")
    emit("(同模型、同题集,不是独立复跑)。\n")
    emit("| 对比 | bench / 指标 | 共同年份点 | 各点 Δ | **正/负** | 均值 Δ |")
    emit("|---|---|---:|---|:---:|---:|")
    for hi, lo, lbl in CMP:
        for store, key, name in ((res, "score", "research 分数"),
                                 (res, "done", "research 完成率"),
                                 (mls, None, "MLS 分数")):
            ys = sorted(set(store.get(hi, {})) & set(store.get(lo, {})))
            ds = []
            for y in ys:
                A, B = store[hi][y], store[lo][y]
                if key is None:
                    A = {t: (v or 0.0) for t, v in A.items()}
                    B = {t: (v or 0.0) for t, v in B.items()}
                    common = sorted(set(A) & set(B))
                    if len(common) < 5:
                        continue
                    ds.append((y, float(np.mean([A[t] - B[t] for t in common]))))
                else:
                    common = sorted(set(A) & set(B))
                    if len(common) < 5:
                        continue
                    ds.append((y, float(np.mean([A[p][key] - B[p][key] for p in common]))))
            if not ds:
                emit(f"| {lbl} | {name} | 0 | — | — | — |"); continue
            pos = sum(1 for _, d in ds if d > 0)
            emit(f"| {lbl} | {name} | {len(ds)} | "
                 + " ".join(f"{y}:{d:+.3f}" for y, d in ds)
                 + f" | **{pos}/{len(ds)-pos}** | **{np.mean([d for _, d in ds]):+.4f}** |")
    emit()


main()
with open(os.path.join(HERE, "year_lora.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
