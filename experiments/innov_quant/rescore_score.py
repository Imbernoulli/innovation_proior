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
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minilb as M

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
RES = D / "outputs" / "rescore_al1"

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


def asrun():
    s = {}
    for tag, _ in ARMS:
        cur = {}
        for d in (f"cc_mls21_{tag}_al1", f"cc_mls21_{tag}_al1-fix"):
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


def main():
    M.build(force="--rebuild" in sys.argv)
    A = asrun()
    T21 = sorted({t for tag, _ in ARMS for t in A[tag]})
    assert len(T21) == 21, f"题数 {len(T21)},应为 21"

    fs, moved, nometric = {}, {}, {}
    for tag, _ in ARMS:
        fs[tag] = {t: (A[tag].get(t) or 0.0) for t in T21}
        moved[tag], nometric[tag] = [], []
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
            m = j.get("metrics") or {}
            if not m:
                nometric[tag].append(t)
                continue
            s = M.score_row(t, f"vllm/{tag}_al1", m)
            if s is None:
                nometric[tag].append(t)
                continue
            fs[tag][t] = s
            if s > 0:
                moved[tag].append((t, s))

    print("## 1. 逐臂:as-run vs file-state(al1,分母 21)\n")
    print("| 臂 | 补测格子 | 补出非零 | as-run 均分 | file-state 均分 | Δ | as-run 非零 | file-state 非零 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    store = {}
    for tag, name in ARMS:
        a = [A[tag].get(t) or 0.0 for t in T21]
        b = [fs[tag][t] for t in T21]
        store[tag] = (a, b)
        n_try = len(moved[tag]) + len(nometric[tag])
        print(f"| {name} | {n_try} | {len(moved[tag])} | {sum(a)/21:.4f} | {sum(b)/21:.4f} | "
              f"{sum(b)/21-sum(a)/21:+.4f} | {sum(1 for x in a if x>0)}/21 | "
              f"{sum(1 for x in b if x>0)}/21 |")

    print("\n## 2. 对照\n")
    print("| 对照 | as-run Δ | file-state Δ |")
    print("|---|---:|---:|")
    for lo, hi, lbl in PAIRS:
        da = (sum(store[hi][0]) - sum(store[lo][0])) / 21
        db = (sum(store[hi][1]) - sum(store[lo][1])) / 21
        print(f"| {lbl} | {da:+.4f} | {db:+.4f} |")

    print("\n## 3. 补出非零的格子\n")
    print("| 臂 | 题 | file-state 分 |")
    print("|---|---|---:|")
    for tag, name in ARMS:
        for t, s in sorted(moved[tag]):
            print(f"| {name} | `{t}` | {s:.4f} |")

    json.dump({"tasks": T21,
               "asrun": {t: store[t][0] for t, _ in ARMS},
               "filestate": {t: store[t][1] for t, _ in ARMS},
               "moved": moved, "nometric": nometric},
              open(RES / "rescore_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
