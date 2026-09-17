"""「最终文件状态」这把尺子:as-run 与 file-state 并排,永不合并。

as-run     = MLS 原样口径,agent 不主动 finalize 就记 0(summary.json 当时算出来的)。
file-state = 用户裁决口径:agent 把方法写进 workspace 却从没测过,我们替它测一遍
             (rescore_cell.py 的产物),再评分。

评分走 minilb(最小排行榜:baseline 行 + 一条待评行)。不能直接读活的
leaderboard.csv —— 它在被别的作业并发读改写,09-16 那批 al1 的行已经有被抹掉的。
minilb 在 54 个「final 行还活着」的格子上逐位复现了 as-run,0 个不符。

file-state 只动 as-run 记 0 的格子,所以它只会往上抬,不会往下压;零分多的臂抬得多。
这对我们是不利的方向(9B base 有 17 个这样的格子,我们只有 8 个),照报。
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minilb as M

# 批次后缀。用户 2026-09-17 定:MLS 一律用 p1 —— al1 换了采样,和年份点不是
# 同一套协议,混在一起年份扫描就没意义了。al1 那份留着当采样 A/B,不进主表。
SUF = next((a for a in sys.argv[1:] if not a.startswith("-")), "p1")
D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
RES = D / "outputs" / f"rescore_{SUF}"

ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]

PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("ft01mix_a10", "rlv5_ft01mix_a10_s20", "9B 我们 − SFT"),
         ("base9b_v2c", "rlv5_ft01mix_a10_s20", "9B 我们 − base"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
         ("4b_ft01mix_a10", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − SFT"),
         ("base4b", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − base")]


def sign_counts(lo, hi):
    """逐题配对:hi 更高 / 更低 / 相同。两条线都要报,不然看不出抬升是不是只发生在少数题上。"""
    w = sum(1 for a, b in zip(lo, hi) if b > a)
    l = sum(1 for a, b in zip(lo, hi) if b < a)
    return w, l, len(lo) - w - l


def sign_p(w, l):
    """双边符号检验(平局剔除)。21 题,用精确二项。"""
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    return min(1.0, 2 * sum(c[:min(w, l) + 1]) / float(sum(c)))


def asrun():
    s = {}
    for tag, _ in ARMS:
        cur = {}
        for d in (f"cc_mls21_{tag}_{SUF}", f"cc_mls21_{tag}_{SUF}-fix"):
            p = D / "outputs" / d / "summary.json"
            if not p.exists():
                continue
            j = json.load(open(p))
            ts = j.get("tasks", j)
            ts = list(ts.values()) if isinstance(ts, dict) else ts
            for t in ts:
                cur[t["task"]] = t.get("score")
        s[tag] = cur
    return s


HERE = os.path.dirname(os.path.abspath(__file__))
_BUF = []


def emit(line=""):
    """同时打屏和落盘。只打 stdout 的脚本,表一转手就丢了(第 18 号)。"""
    sys.stdout.write(line + "\n")
    _BUF.append(line)


def _flush(name):
    with open(os.path.join(HERE, f"{name}_{SUF}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(_BUF) + "\n")


def main():
    M.build(force="--rebuild" in sys.argv)
    A = asrun()
    T21 = sorted({t for tag, _ in ARMS for t in A[tag]})
    assert len(T21) == 21, f"题数 {len(T21)},应为 21"

    fs, moved, nometric, tried = {}, {}, {}, {}
    for tag, _ in ARMS:
        fs[tag] = {t: (A[tag].get(t) or 0.0) for t in T21}
        moved[tag], nometric[tag], tried[tag] = [], [], []
        d = RES / tag
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            j = json.load(open(f))
            t = j["task"]
            if t not in fs[tag]:
                continue
            if (A[tag].get(t) or 0.0) != 0.0:
                continue                      # 只补 as-run 记 0 的格子
            tried[tag].append(t)              # 补测过的格子,不管结果如何
            m = j.get("metrics") or {}
            if not m:
                nometric[tag].append(t)
                continue
            s = M.score_row(t, f"vllm/{tag}_{SUF}", m)
            if s is None:
                nometric[tag].append(t)
                continue
            fs[tag][t] = s
            if s > 0:
                moved[tag].append((t, s))

    emit(f"## 1. 逐臂:as-run vs file-state({SUF},分母 21)\n")
    emit("| 臂 | 补测格子 | 跑出指标 | 补出非零 | as-run 均分 | file-state 均分 | Δ | as-run 非零 | file-state 非零 |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    store = {}
    for tag, name in ARMS:
        a = [A[tag].get(t) or 0.0 for t in T21]
        b = [fs[tag][t] for t in T21]
        store[tag] = (a, b)
        # 三个数不是同一件事:补测过的格子 ⊋ 跑出指标的 ⊋ 评分为正的。
        # 中间那一层以前没打印,于是这一列比 rescore_why.py 的 71 少了 10 格 ——
        # 「跑出指标但分数被钳到 0」既不在 moved 里也不在 nometric 里。
        n_try = len(tried[tag])
        n_met = n_try - len(nometric[tag])
        emit(f"| {name} | {n_try} | {n_met} | {len(moved[tag])} | {sum(a)/21:.4f} | {sum(b)/21:.4f} | "
              f"{sum(b)/21-sum(a)/21:+.4f} | {sum(1 for x in a if x>0)}/21 | "
              f"{sum(1 for x in b if x>0)}/21 |")

    emit("\n## 2. 对照\n")
    emit("| 对照 | as-run Δ | as-run 胜/负/平 | as-run p | file-state Δ | file-state 胜/负/平 | file-state p |")
    emit("|---|---:|:---:|---:|---:|:---:|---:|")
    for lo, hi, lbl in PAIRS:
        da = (sum(store[hi][0]) - sum(store[lo][0])) / 21
        db = (sum(store[hi][1]) - sum(store[lo][1])) / 21
        wa = sign_counts(store[lo][0], store[hi][0])
        wb = sign_counts(store[lo][1], store[hi][1])
        emit(f"| {lbl} | {da:+.4f} | {wa[0]}/{wa[1]}/{wa[2]} | {sign_p(*wa[:2]):.4f} | "
              f"{db:+.4f} | {wb[0]}/{wb[1]}/{wb[2]} | {sign_p(*wb[:2]):.4f} |")

    emit("\n## 3. 补出非零的格子\n")
    emit("| 臂 | 题 | file-state 分 |")
    emit("|---|---|---:|")
    for tag, name in ARMS:
        for t, s in sorted(moved[tag]):
            emit(f"| {name} | `{t}` | {s:.4f} |")

    json.dump({"tasks": T21,
               "asrun": {t: store[t][0] for t, _ in ARMS},
               "filestate": {t: store[t][1] for t, _ in ARMS},
               "moved": moved, "nometric": nometric, "tried": tried},
              open(RES / "rescore_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
    _flush("rescore_score")
