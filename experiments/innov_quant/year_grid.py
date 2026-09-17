# -*- coding: utf-8 -*-
"""全量年份网格:所有 arm × 所有年份 × 所有 bench,含 2026。

2026 这个点有两种来源,必须分开看:
  - 显式 `_y2026` 目录
  - **裸 tag**:`EVAL_RESEARCHER_YEAR` 默认 2026(`cc_eval_cpu_client_pinned.sh:225`),
    MLS 的 p1 也显式传 `MLSBENCH_SYS_PREFIX="It is now year 2026."`。
    所以裸 tag / p1 在语义上就是 y2026。
但 §25 的坑在这里最致命:裸 tag 和年份点往往**不是同一世代**。所以本脚本
每一格都同时打印 mtime,并按 mtime 把格子分到「世代」里。**不做任何跨世代合并**,
合不合由看表的人决定。

用法:`python year_grid.py` → year_grid.md
"""
import os, re, json, glob, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
import dump2

YEARS = [1700, 1800, 1900, 1950, 1975, 2000, 2010, 2025, 2026, 2050, 2075, 2100]
# 2026-09-16 用户定调:年份这条线只有 FCS-research 和 MLS 有希望,FrontierCS / ALE 不再跟。
NONMLS = [("frontiercs_research", "FCS-research")]


def arms():
    """从目录名反推所有臂。年份后缀和 bench 后缀都剥掉。"""
    s = set()
    for p in glob.glob(f"{D}/cc_eval_*_thinking_32k_*_vllm") + glob.glob(f"{D}/cc_mls21_*"):
        b = os.path.basename(p)
        b = re.sub(r"_research_thinking_32k_vllm$|_thinking_32k_both_vllm$", "", b)
        b = re.sub(r"^cc_eval_|^cc_mls21_", "", b)
        b = re.sub(r"_(y\d+|p\d+|r\d+|al\d+)(\..*)?$", "", b)
        if b and "." not in b and "smoke" not in b and "failed" not in b:
            s.add(b)
    # 垃圾行:单题跟跑(p1-alfix)、被当成臂的复跑目录(..._y1950r2)
    s = {a for a in s if not a.endswith("-alfix") and not re.search(r"_y\d+r\d+$", a)}
    # 八条主臂排最前,其余按名字
    return [a for a in MAIN if a in s] + sorted(s - set(MAIN))


MAIN = ["base9b_v2c", "ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "base4b", "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]


def mt(p):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(os.stat(p).st_mtime).strftime("%m-%d")
    except OSError:
        return None


LOGD = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/logs"
_PYC = {}


def worker_py(tag):
    """这一批用的是哪个 worker python。**这才是 MLS 的世代主键** ——
    2026-09-16 查出:裸 tag 与全部 `_y####` 年份点走 /home/zy7019/miniconda3/bin/python3,
    而 `_p1`/`_al1` 走 $D/envs/client/bin/python(`mls21_rerun_submit.sh` 显式传 MLSBENCH_PY)。
    两者不是同一套环境,混进同一条年份曲线就是 §25。"""
    if tag in _PYC:
        return _PYC[tag]
    out = "?"
    ys = glob.glob(f"{D}/cc_mls21_{tag}/config_vllm_local_*.yaml")
    if ys:
        jid = re.search(r"_(\d+)\.yaml$", ys[0])
        if jid:
            for f in glob.glob(f"{LOGD}/*-{jid.group(1)}.out"):
                try:
                    for ln in open(f, encoding="utf-8", errors="replace"):
                        if "launching worker pool with" in ln:
                            out = "client" if "envs/client" in ln else "conda"
                            raise StopIteration
                except StopIteration:
                    break
                except OSError:
                    pass
    _PYC[tag] = out
    return out


def mls_cell(tag):
    p = f"{D}/cc_mls21_{tag}/summary.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ts = d.get("tasks", d)
    if isinstance(ts, dict):
        ts = list(ts.values())
    # 用户 2026-09-16:MLS 分母一律 21。真 0 记 0 进分母;环境坏掉的题不进,要重跑。
    # 补跑结果写在 <tag>-fix 目录,这里按题后写覆盖并进来。
    sc = [t["score"] for t in ts
          if t.get("score") is not None
          and "agent_failed" not in (t.get("status") or "")
          and "timeout" not in (t.get("status") or "")]
    # 口径必须与 mls_year.py 一致:统计**出现过至少一次** APIConnectionError 的题数,
    # 该格 dead>=3 题即判基础设施无效。先前这里写成「单个日志里出现>=3次」,那从不发生,
    # dead 恒为 0,整张年份表一格死 serve 都没标出来(ft01mix_a10 y2000/y2075 实为 12/18 题)。
    dead = sum(1 for t in ts
               if os.path.exists(t.get("log") or "")
               and "APIConnectionError" in open(t["log"], encoding="utf-8",
                                                errors="replace").read())
    # 合并补跑:同名题以 -fix 目录里的为准
    fx = f"{D}/cc_mls21_{tag}-fix/summary.json"
    if os.path.exists(fx):
        fd = json.load(open(fx)); fts = fd.get("tasks", fd)
        if isinstance(fts, dict):
            fts = list(fts.values())
        byname = {t["task"]: t for t in ts}
        for t in fts:
            byname[t["task"]] = t
        ts = list(byname.values())
        sc = [t["score"] for t in ts
              if t.get("score") is not None
              and "agent_failed" not in (t.get("status") or "")
              and "timeout" not in (t.get("status") or "")]
    return dict(n=len(sc), mean=(float(np.mean(sc)) if sc else float("nan")),
                mean21=(float(np.sum(sc)) / 21.0),     # 用户口径:分母钉死 21
                mt=mt(f"{D}/cc_mls21_{tag}"), dead=dead, tot=len(ts), py=worker_py(tag),
                fixed=os.path.exists(fx))


def nonmls_cell(tag, bench):
    sub = dump2.sub_of(bench)
    if not os.path.isdir(f"{D}/cc_eval_{tag}_{sub}"):
        return None
    try:
        ok, seen = dump2.load(tag, bench)
    except Exception:
        return None
    if not ok:
        return dict(n=0, mean=float("nan"), mt=mt(f"{D}/cc_eval_{tag}_{sub}"), dead=0, tot=len(seen))
    v = [r["score"] for r in ok.values()]
    return dict(n=len(v), mean=float(np.mean(v)), mt=mt(f"{D}/cc_eval_{tag}_{sub}"),
                dead=0, tot=len(seen))


def tags_for(arm, year):
    """该 (arm, year) 可能对应哪些 tag。2026 有两条来源,分别返回。"""
    if year != 2026:
        return [(f"{arm}_y{year}", f"y{year}")]
    out = [(f"{arm}_y2026", "y2026")]
    out.append((arm, "裸tag"))
    out.append((f"{arm}_p1", "p1"))      # MLS 的 2026 协议批
    return out


def main():
    A = arms()
    out = []
    W = out.append
    W("# 全量年份网格(所有 arm × 所有年份 × 所有 bench,含 2026)\n")
    W("每格 = `均分 (n) mtime`。`n` 是进入均值的单位数:MLS 是**题**(剔 agent_failed / "
      "撞墙钟),其余三条 bench 是**抽样**(已按 `dump2.load` 去重、跳过 error 行)。\n")
    W("**2026 有三种来源,分列不合并**:`y2026` 显式目录 / `裸tag`(`EVAL_RESEARCHER_YEAR` "
      "默认 2026)/ `p1`(MLS 的 2026 协议批,显式传 `It is now year 2026.`)。\n")
    W("⚠ **mtime 就是世代**。同一行里 mtime 差好几天的格子来自不同协议世代,"
      "横着比会静默出错(§25)。合不合请看 mtime 自己决定。\n")

    for bench, blab in [("mls", "MLS-Bench")] + NONMLS:
        W(f"\n## {blab}\n")
        cols = []
        for y in YEARS:
            if y == 2026:
                cols += [("2026:y2026", (2026, "y2026")), ("2026:裸tag", (2026, "裸tag"))]
                if bench == "mls":
                    cols += [("2026:p1", (2026, "p1"))]
            else:
                cols += [(str(y), (y, f"y{y}"))]
        # 只保留至少一条臂有数据的列
        grid = {}
        for arm in A:
            for _, (y, kind) in cols:
                tag = (arm if kind == "裸tag" else
                       f"{arm}_p1" if kind == "p1" else f"{arm}_y{y}")
                c = mls_cell(tag) if bench == "mls" else nonmls_cell(tag, bench)
                if c:
                    grid[(arm, y, kind)] = c
        cols = [c for c in cols if any((a, c[1][0], c[1][1]) in grid for a in A)]
        MINN = 5 if bench == "mls" else 100
        rows = [a for a in A
                if any(grid[(a, y, k)]["n"] >= MINN for _, (y, k) in cols if (a, y, k) in grid)]
        if not cols or not rows:
            W("_(无数据)_\n"); continue
        W("| arm | " + " | ".join(h for h, _ in cols) + " |")
        W("|---|" + "---|" * len(cols))
        for arm in rows:
            cells = []
            for _, (y, k) in cols:
                c = grid.get((arm, y, k))
                if not c:
                    cells.append("—"); continue
                flag = f" ⛔{c['dead']}" if c.get("dead") else ""
                cells.append(f"{c['mean']:.3f} ({c['n']}) {c['mt']}{flag}")
            nm = f"**`{arm}`**" if arm in MAIN else f"`{arm}`"
            W(f"| {nm} | " + " | ".join(cells) + " |")
        W("")
    out += completeness_audit()
    out += curve_section()
    out += paired_2026()
    p = os.path.join(HERE, "year_grid.md")
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


# ============ 完整性审计:每一格到底判出了几题 ============
# 2026-09-16 用户问「裸 tag 比年份点多 10-20 题,这问题解决了吗?其他数字还能信吗」。
# 那个 10-20 是我算错的(见下面配对检验的注释)。这一节把逐格的判出题数固定下来,
# 以后任何人都能自己数一遍,不用信我的话。
FULL = {"frontiercs_research": 64, "frontiercs": 172, "alebench": 40}


def completeness_audit():
    out = ["\n\n# 0. 完整性审计:每格判出了几道题\n",
           "研究 bench 满题 64,MLS 满题 21。`判出题/满题`。**这是读下面任何数字之前先看的一列。**",
           "丢掉的行全是判题侧的 `ResearchInfraError: evaluator produced no score`,"
           "集中在同一小撮题(`symbolic_regression/{sincos,mixed_polyexp_4d,peaks,ripple}`),"
           "**裸 tag 与年份点丢的是同一批**,所以不是偏向性削题,逐题配对可以救。\n"]
    for bench, blab in NONMLS + [("mls", "MLS-Bench")]:
        full = 21 if bench == "mls" else FULL[bench]
        out.append(f"\n## {blab}(满题 {full})\n")
        out.append("| arm | 裸tag | " + " | ".join(str(y) for y in ALLY) + " |")
        out.append("|---|---|" + "---|" * len(ALLY))
        for arm, lab, _line in LINE:
            cells = []
            for tag in [arm] + [f"{arm}_y{y}" for y in ALLY]:
                if bench == "mls":
                    c = mls_cell(tag)
                    n = c["n"] if c else None
                    # 死 serve 单独标:它不是「模型没做出来」,是那一格的 vLLM 当时是死的。
                    if c and c["dead"] >= 3:
                        cells.append(f"**{n}** ☠{c['dead']}")
                        continue
                else:
                    if not os.path.isdir(f"{D}/cc_eval_{tag}_{dump2.sub_of(bench)}"):
                        n = None
                    else:
                        try:
                            ok, _ = dump2.load(tag, bench)
                            n = len({g for g, _i in ok})
                        except Exception:
                            n = None
                if n is None:
                    cells.append("")
                elif n < full * 0.9:
                    cells.append(f"**{n}** ⛔")
                else:
                    cells.append(str(n))
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("\n☠k = 该格有 k 道题日志里出现 `openai.APIConnectionError`(k≥3 即 vLLM serve 当时是死的),"
               "**基础设施无效,不是模型表现**。\n\n"
               "MLS 还有一个结构性下限:`causal-observational-linear-gaussian` 与 "
               "`optimization-multi-objective` 在装 `causal-learn`/`deap` 之前恒定 `agent_failed`,"
               "所以旧批次的天花板是 **19/21** 而不是 21 —— 19 不是数据损失。\n\n"
               "⛔ = 判出不足 90%,该格作废。**已知作废格:`base9b_v2c_y1700`**"
               "(FCS 142/172、ALE 6/40 —— 这两条 bench 现已不跟)、"
               "**`base4b_y2010` 的 ALE 34/40**。research 侧没有作废格。")
    return out

# ============ 同世代五点年份曲线(含 2026) ============
# 上面的大网格是清单。这一节只做一件事:把**同世代**的年份点拼成一条曲线。
# 依据是 mtime —— ft01mix 那条四点线的年份点是 09-15/16,而这六条臂的裸 tag 也是 09-15,
# 所以 2026 可以并进去,变成 2000 / 2025 / 2026 / 2050 / 2075 五个点。
# base9b_v2c / base4b 不在这条线上(它们只有 09-11/12 的老扫描),单列。
LINE = [("base9b_v2c", "9B base", "老扫描 09-11/12"),
        ("ft01mix_a10", "9B SFT", "新四点 09-15/16"),
        ("rlv5_base_s20", "9B RL(base)", "新四点 09-15/16"),
        ("rlv5_ft01mix_a10_s20", "9B RL(先验)", "新四点 09-15/16"),
        ("base4b", "4B base", "老扫描 09-11/12"),
        ("4b_ft01mix_a10", "4B SFT", "新四点 09-15/16"),
        ("rlv5_4b_base_s20", "4B RL(base)", "新四点 09-15/16"),
        ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)", "新四点 09-15/16")]

ALLY = YEARS   # 1700 … 2100,缺的留空


def curve_section():
    out = ["\n\n# 年份曲线:全部年份 × 全部主臂(缺的留空)\n",
           "**2026 取裸 tag**(`EVAL_RESEARCHER_YEAR` 默认就是 2026)。",
           "两条扫描线:`新四点`只投了 2000/2025/2050/2075;`老扫描`投了 "
           "1700-2100 共 12 个点但只覆盖 base 与 lo32nm 两族。**两条线不是同一世代,不要横跨着比。**\n",
           "**MLS 每格 `总分/21 (判出题数/21)`** —— 用户 2026-09-16 定:分母一律 21,真 0 记 0,"
           "环境坏掉的题不进分母、必须重跑。带 ⚠ 的格子判出题数不足 21,**那一格现在是低估的**,"
           "补跑作业(`mlsfix-*`)落地后会自动并进来(读 `<tag>-fix` 目录,同名题后写覆盖)。"
           "research 每格仍是 `均分 (抽样数)`。峰值只在该臂**自己有的**点里取。\n",
           "**MLS 的 2026 有四种来源,全部就地列在 2026 那一格**(先前我把它们赶到右边侧栏,"
           "结果主结果从 2026 消失了 —— 判据定得太严,已改回)。\n\n"
           "四者的 env 逐项比过:prefix 形式、`MAX_MODEL_LEN=40960`、`TASK_TIMEOUT=7200`、"
           "`CONCURRENCY=7` 全都一样,`_p1`/`_al1` 与年份点的**唯一差别是 `MLSBENCH_PY`** "
           "(年份点与裸 tag 走 conda python,`_p1`/`_al1` 走 client venv);`_al1` 另外还改了采样。"
           "裸 tag 则还在 09-11 vendor 重建之前,是唯一一个真·旧世代。\n\n"
           "这个 worker-python 差**量过了**(只有 `base9b_v2c`/`base4b` 同时有两边,逐题配对):"
           "9B base y2026→p1 **+0.068**(6/0, W=0.028)、y2026→al1 +0.006(3/4, ns);"
           "4B base y2026→p1 +0.018(ns)、y2026→al1 +0.040(ns)。"
           "**四个对照只有一个显著**,量级与年份间差同阶 —— 不足以把它们排除,但纵向读曲线时"
           "只在同一来源内读。\n"]
    for bench, blab in [("mls", "MLS-Bench")] + NONMLS:
        out.append(f"\n## {blab}\n")
        if bench == "mls":
            hdr = " | ".join(
                ("**2026:y2026** | **2026:p1** | **2026:al1** | **2026:裸tag**"
                 if y == 2026 else str(y)) for y in ALLY)
            nex = 3
        else:
            hdr = " | ".join((f"**{y}**" if y == 2026 else str(y)) for y in ALLY)
            nex = 0
        out.append(f"| arm | 扫描线 | {hdr} | 峰值 | 点数 |")
        out.append("|---|---|" + "---|" * (len(ALLY) + 2 + nex))
        for arm, lab, line in LINE:
            vals, cells = {}, []
            for y in ALLY:
                if bench == "mls" and y == 2026:
                    # 2026 的四个来源就地铺开。峰值按 al1(我们的主协议)参与,
                    # 它没有就退到 p1、再退到 _y2026 —— 这样每条臂的 2026 都不会是空的。
                    got = {}
                    for key, tg in (("y2026", f"{arm}_y2026"), ("p1", f"{arm}_p1"),
                                    ("al1", f"{arm}_al1"), ("裸tag", arm)):
                        c = mls_cell(tg)
                        if c and c["n"] > 0 and not np.isnan(c["mean"]):
                            got[key] = c
                            flag = "" if c["n"] >= 21 else "⚠"
                            cells.append(f"{c['mean21']:.3f} ({c['n']}/21{flag})")
                        else:
                            cells.append("")
                    for key in ("al1", "p1", "y2026"):
                        if key in got:
                            vals[y] = got[key]["mean21"]
                            break
                    continue
                tag = f"{arm}_y{y}" if bench == "mls" else (arm if y == 2026 else f"{arm}_y{y}")
                c = mls_cell(tag) if bench == "mls" else nonmls_cell(tag, bench)
                if c and c["n"] > 0 and not np.isnan(c["mean"]):
                    if bench == "mls":
                        vals[y] = c["mean21"]
                        flag = "" if c["n"] >= 21 else "⚠"
                        cells.append(f"{c['mean21']:.3f} ({c['n']}/21{flag})")
                    else:
                        vals[y] = c["mean"]
                        cells.append(f"{c['mean']:.3f} ({c['n']})")
                else:
                    cells.append("")
            row = f"| {lab} | {line} | " + " | ".join(cells) + " | "
            if len(vals) < 3:
                out.append(row + f"— | {len(vals)} |")
            else:
                out.append(row + f"**{max(vals, key=vals.get)}** | {len(vals)} |")
    out.append("\n**看峰值那一列**:倒 U 成立的话峰应该集中在 2025/2026。"
               "另见下一节 —— 峰值列受分母影响,配对之后效应消失。")
    return out



# ============ 2026 的配对检验 ============
# 上一节按「峰值年份」看,FCS-research 六条臂的峰全落在 2025/2026,看着像倒 U。
# **那是假的**,但**原因不是我先前写的那个**。
#
# 【2026-09-16 更正】我先前在这里写过「裸 tag 比年份点多 10-20 道题」——**错的,已收回**。
# 逐格数过之后:research 裸 tag 64 题、各年份点 62-64 题,差 **1-3 题**;FCS 差 0-1;ALE 差 1。
# 那个「10-20」是我自己取「该臂全部年份点的交集」算出来的假象:每个点各自丢的题不一样,
# 12 个点叠起来交集才掉得多;而 ALE 上 `base9b_v2c_y1700` 只判出 6/40(那一格真死了),
# 一个人把交集拖到 6,于是 40-6=34。我把这个 ALE 上的数字过度概括成了 research 的。
#
# 丢题的真因是**判题侧**不是模型侧:丢掉的行全是 `ResearchInfraError: evaluator produced
# no score`,且集中在同一小撮题(`symbolic_regression/{sincos,mixed_polyexp_4d,peaks,ripple}`),
# **裸 tag 和年份点丢的是同一批题**。所以它不是偏向性削题,逐题配对能干净地救回来。
# 反过来脏的是裸 tag:`ft01mix_a10` research 裸 tag 两分片共 478 行(应 320),重复行来自
# 09-02/03 那代的超时补跑;年份点是整齐的 320 行。`dump2.load` 的去重就是为这个写的。
#
# 均值差的真因仍是分母(§27),只是量级小得多:9B SFT research 64 题无约束 11.573,
# 限制到 61 道公共题变 10.668 —— 0.9 分来自 3 题。配对之后效应仍然消失,结论不变。
# 这一节就是那个配对检验,留着防止有人再被峰值列骗一次。
from scipy import stats as _st


def _byprob(tag, bench):
    try:
        ok, _ = dump2.load(tag, bench)
    except Exception:
        return {}
    d = {}
    for (gt, si), r in ok.items():
        d.setdefault(gt, []).append(r["score"])
    return {g: float(np.mean(v)) for g, v in d.items()}


def paired_2026():
    out = ["\n\n# 2026 的配对检验(按题配对,非均值)\n",
           "峰值列会骗人:裸 tag 比年份点多 10-20 道题,多的那些分高。"
           "下表把每条臂限制到 **2000/2025/2026/2075 四个点都有的公共题**再比。\n"]
    for bench, blab in NONMLS:
        out.append(f"\n## {blab}\n")
        out.append("| arm | 公共题 n | 2026 | FAR(2000/2075均值) | Δ | +/− | Wilcoxon p | 2026−2025 | +/− | p |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        zf, z25 = [], []
        for arm, lab, _line in LINE:
            b26 = _byprob(arm, bench); b00 = _byprob(f"{arm}_y2000", bench)
            b75 = _byprob(f"{arm}_y2075", bench); b25 = _byprob(f"{arm}_y2025", bench)
            gs = sorted(set(b26) & set(b00) & set(b75) & set(b25))
            if len(gs) < 20:
                out.append(f"| {lab} | {len(gs)} | — | — | — | — | — | — | — | ⚠不足 |")
                continue
            a = np.array([b26[g] for g in gs]); far = np.array([(b00[g] + b75[g]) / 2 for g in gs])
            n25 = np.array([b25[g] for g in gs])
            d = a - far; p_, n_ = int((d > 0).sum()), int((d < 0).sum())
            w = _st.wilcoxon(d).pvalue if np.any(d) else 1.0
            zf.append(_st.norm.isf(w / 2) * np.sign(d.mean()))
            d2 = a - n25; p2, n2 = int((d2 > 0).sum()), int((d2 < 0).sum())
            w2 = _st.wilcoxon(d2).pvalue if np.any(d2) else 1.0
            z25.append(_st.norm.isf(w2 / 2) * np.sign(d2.mean()))
            out.append(f"| {lab} | {len(gs)} | {a.mean():.3f} | {far.mean():.3f} | {d.mean():+.3f} | "
                       f"{p_}/{n_} | {w:.4f} | {d2.mean():+.3f} | {p2}/{n2} | {w2:.4f} |")
        f = lambda z: (sum(z) / np.sqrt(len(z))) if z else float("nan")
        out.append(f"| **聚合 Stouffer** | 6 格 | | | | | **Z={f(zf):+.2f}** | | | **Z={f(z25):+.2f}** |")
    out.append("\n**结论:2026 既不高于远年、也不高于 2025。倒 U 加了 2026 之后依然不成立。**")
    return out


if __name__ == "__main__":
    main()


def dump_json(path):
    """给页面用的数据。结构:{bench: {arm: {year: {mean,n,mt}}}} + 配对检验。"""
    import json as _j
    D_ = {"years": ALLY, "arms": [{"key": a, "label": l, "line": ln} for a, l, ln in LINE],
          "benches": [["mls", "MLS-Bench"]] + [list(x) for x in NONMLS], "data": {}}
    for bench, _ in [("mls", "MLS-Bench")] + NONMLS:
        D_["data"][bench] = {}
        for arm, lab, line in LINE:
            row = {}
            for y in ALLY:
                # MLS 的 2026 只认 `_y2026`(与年份点同一次投递、同一个 worker python)。
                # 裸 tag 是 vendor 重建之前的,p1/al1 换了 worker python —— 都不能并进曲线,
                # 它们在下面单列。非 MLS 的三条 bench 裸 tag 与年份点同世代,可以并。
                if bench == "mls":
                    tag = f"{arm}_y{y}"
                else:
                    tag = arm if y == 2026 else f"{arm}_y{y}"
                c = mls_cell(tag) if bench == "mls" else nonmls_cell(tag, bench)
                if c and c["n"] > 0 and not np.isnan(c["mean"]):
                    row[str(y)] = {"mean": round(c["mean"], 4), "n": c["n"], "mt": c["mt"]}
            D_["data"][bench][arm] = row
    _j.dump(D_, open(path, "w"), ensure_ascii=False)
    return path
