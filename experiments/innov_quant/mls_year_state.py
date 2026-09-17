"""年份线的 MLS 两把尺子:as-run 与 file-state,分开两张表,永不合并。

as-run     = MLS 原样口径,agent 不主动 finalize 就记 0(summary.json 当时算出来的)。
file-state = 用户裁决口径:方法写进了 workspace 却从没测过,我们替它测一遍再评分
             (rescore_cell.py 的产物)。只动 as-run 记 0 的格子,所以只会往上抬。

年份点一律 p1 协议:2026 这一列取 `<arm>_p1`,不取 al1 —— al1 换了采样
(temp/top_p/top_k/min_p/presence_penalty),年份点一个都没换,混进同一条曲线
就等于让 2026 单独换了套协议。al1 留作采样 A/B,不进这张表。

分母钉死 21:跑出分的进分子,没跑出分的记 0,不缩分母(第 h 条 + 第 j 条)。
`-fix` 合并、`-alfix` 不并。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minilb as M

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B 我们"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B 我们")]
# 2026 用 p1 那批;其余是年份点。键是列名,值是 (目录 tag, 重评目录)。
COLS = [("2000", "y2000", "rescore_year"), ("2025", "y2025", "rescore_year"),
        ("2026", "p1", "rescore_p1"), ("2050", "y2050", "rescore_year"),
        ("2075", "y2075", "rescore_year"), ("2100", "y2100", "rescore_year")]

# 用户裁决 2026-09-17:**只看 RL 之后,不看 RL 之前**。RL 是最终 shape 出来的模型,
# 所以主对照只有「我们的 RL − baseline 的 RL」。SFT / base 那些对照降为附录,
# 留着是为了不丢历史,不进主表、不进论文正文。
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)")]
PAIRS_PRE = [("ft01mix_a10", "rlv5_ft01mix_a10_s20", "9B 我们 − SFT"),
             ("base9b_v2c", "rlv5_ft01mix_a10_s20", "9B 我们 − base"),
             ("4b_ft01mix_a10", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − SFT"),
             ("base4b", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − base")]

_BUF = []


def emit(line=""):
    """同时打屏和落盘。只打 stdout 的脚本,表一转手就丢了(第 18 号)。"""
    sys.stdout.write(line + "\n")
    _BUF.append(line)


def asrun(tag):
    """<tag> 与 <tag>-fix 合并,同名题后写覆盖;-alfix 不并(第 17 号)。"""
    cur = {}
    for d in (f"cc_mls21_{tag}", f"cc_mls21_{tag}-fix"):
        p = D / "outputs" / d / "summary.json"
        if not p.exists():
            continue
        j = json.load(open(p))
        ts = j.get("tasks", j)
        ts = list(ts.values()) if isinstance(ts, dict) else ts
        for t in ts:
            cur[t["task"]] = t.get("score")
    return cur


def main():
    M.build()
    T21 = sorted(asrun(f"{ARMS[0][0]}_p1"))
    assert len(T21) == 21, f"题数 {len(T21)},应为 21"

    A, F, cov = {}, {}, {}
    for tag, _ in ARMS:
        for col, suf, rdir in COLS:
            full = f"{tag}_{suf}"
            if not (D / "outputs" / f"cc_mls21_{full}").is_dir():
                continue
            a = asrun(full)
            A[(tag, col)] = {t: (a.get(t) or 0.0) for t in T21}
            f = dict(A[(tag, col)])
            n_try = n_hit = 0
            # rescore_p1 那批当时是按「臂名 + SUF 参数」调的,落盘目录只有臂名;
            # 年份这批 tag 自己就是全名。两边目录名不一样,别按一个规则去拼。
            rd = D / "outputs" / rdir / (tag if suf == "p1" else full)
            if rd.is_dir():
                for p in sorted(rd.glob("*.json")):
                    j = json.load(open(p))
                    t = j["task"]
                    if t not in f or f[t] != 0.0:
                        continue
                    n_try += 1
                    m = j.get("metrics") or {}
                    if not m:
                        continue
                    s = M.score_row(t, f"vllm/{full}", m)
                    if s is None:
                        continue
                    f[t] = s
                    if s > 0:
                        n_hit += 1
            F[(tag, col)] = f
            cov[(tag, col)] = (n_try, n_hit, sum(1 for t in T21 if not A[(tag, col)][t]))

    cols = [c for c, _, _ in COLS]

    def incomplete(tag, c):
        """这一格的 0 分格子还没补完 —— file-state 值只是「补了一部分」的中间态。

        y2050 是后落地的,它的 0 分格子当时不在重评计划里,于是同一列上
        四个臂全补、四个臂一格没补。不标出来,这一列会被当成可比的数读(第 6 号)。
        """
        n_try, _, n_zero = cov.get((tag, c), (0, 0, 0))
        return n_zero and n_try < n_zero

    def grid(store, title, mark=False):
        emit(f"## {title}\n")
        emit("| 臂 | " + " | ".join(cols) + " | 峰值 |")
        emit("|---|" + "---:|" * (len(cols) + 1))
        for tag, name in ARMS:
            cells, vals = [], {}
            for c in cols:
                if (tag, c) not in store:
                    cells.append("—"); continue
                v = sum(store[(tag, c)].values()) / 21
                vals[c] = v
                cells.append(f"{v:.3f}" + ("⚠" if mark and incomplete(tag, c) else ""))
            pk = max(vals, key=vals.get) if vals else "—"
            emit(f"| {name} | " + " | ".join(cells) + f" | {pk} |")
        emit()

    grid(A, "1. as-run(MLS 原样口径,总分/21)")
    grid(F, "2. file-state(补测「有方法但没 finalize」的格子后,总分/21)", mark=True)
    emit("> ⚠ = 这一格的 0 分格子**还没补完**,值是中间态,不要和同列其它格比。\n")

    emit("## 3. 补测覆盖(记 0 的格子里,已补测 / 补出非零)\n")
    emit("| 臂 | " + " | ".join(cols) + " |")
    emit("|---|" + "---:|" * len(cols))
    for tag, name in ARMS:
        cs = []
        for c in cols:
            if (tag, c) not in cov:
                cs.append("—"); continue
            n_try, n_hit, n_zero = cov[(tag, c)]
            cs.append(f"{n_try}/{n_zero} → +{n_hit}")
        emit(f"| {name} | " + " | ".join(cs) + " |")
    emit()

    def contrast(pairs):
        emit("| 对照 | 尺子 | " + " | ".join(cols) + " |")
        emit("|---|---|" + "---:|" * len(cols))
        for lo, hi, lbl in pairs:
            for store, sname in ((A, "as-run"), (F, "file-state")):
                cs = []
                for c in cols:
                    if (lo, c) not in store or (hi, c) not in store:
                        cs.append("—"); continue
                    d = (sum(store[(hi, c)].values()) - sum(store[(lo, c)].values())) / 21
                    bad = sname == "file-state" and (incomplete(lo, c) or incomplete(hi, c))
                    cs.append(f"{d:+.3f}" + ("⚠" if bad else ""))
                emit(f"| {lbl} | {sname} | " + " | ".join(cs) + " |")
        emit()

    emit("## 4. 主对照:RL 之后(Δ = 我们的 RL − baseline 的 RL)\n")
    emit("**主尺是 as-run**;file-state 行只作稳健性旁证。\n")
    contrast(PAIRS)

    emit("## 5. 附录:RL 之前的对照(不进主表)\n")
    emit("留档用。RL 是最终 shape 出来的模型,论文正文只引第 4 节。\n")
    contrast(PAIRS_PRE)

    json.dump({"asrun": {f"{t}|{c}": v for (t, c), v in A.items()},
               "filestate": {f"{t}|{c}": v for (t, c), v in F.items()},
               "cov": {f"{t}|{c}": v for (t, c), v in cov.items()}},
              open(os.path.join(HERE, "mls_year_state.json"), "w"),
              ensure_ascii=False, indent=1)


main()
with open(os.path.join(HERE, "mls_year_state.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(_BUF) + "\n")
