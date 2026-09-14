"""Statistics for the quantitative MLS case study. Reads runs.json/texts.json/templates.json/baselines.json; writes flat.csv + report_tables.md"""
import json, csv, collections, statistics as st, math, os, sys
import numpy as np
from scipy import stats
OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{OUT}/runs.json")); runs = R["runs"]; tinfo = R["tinfo"]
BASEL = json.load(open(f"{OUT}/baselines.json")); TPL = json.load(open(f"{OUT}/templates.json"))
sys.path.insert(0, OUT)
from mls_quant_extract2 import code_metrics, region, TASKS, ARMS9, ARMS4
FAM = {a: "9B" for a in ARMS9} | {a: "4B" for a in ARMS4}
STAGE = {"base9b_v2c": "base", "ft01mix_a10": "sft", "ft03nm_a20": "sft", "lo32nm_a10": "sft", "rlv5_base_s20": "rl_base", "rlv5_ft01mix_a10_s20": "rl_sft", "rlv5_ft03nm_a20_s20": "rl_sft", "rlv5_lo32nm_a10_s20": "rl_sft",
         "base4b": "base", "4b_ft01mix_a10": "sft", "rlv5_4b_base_s20": "rl_base", "rlv5_4b_ft01mix_a10_s20": "rl_sft"}
SHORT = {"base9b_v2c": "base", "ft01mix_a10": "ft01mix", "ft03nm_a20": "ft03nm", "lo32nm_a10": "lo32nm", "rlv5_base_s20": "rlv5_base", "rlv5_ft01mix_a10_s20": "rlv5_ft01mix", "rlv5_ft03nm_a20_s20": "rlv5_ft03nm", "rlv5_lo32nm_a10_s20": "rlv5_lo32nm",
         "base4b": "base4b", "4b_ft01mix_a10": "4b_ft01mix", "rlv5_4b_base_s20": "rlv5_4b_base", "rlv5_4b_ft01mix_a10_s20": "rlv5_4b_ft01mix"}
# baseline region metrics per task
BM = {}
for t in TASKS:
    ms = {b: code_metrics(txt) for b, txt in BASEL[t].items()}
    tr = tinfo[t]; tpl = TPL[t]; tm = code_metrics(region(tpl, (tr["edit_start"], tr["edit_end"], None)))
    BM[t] = dict(baselines=ms, template=tm, loc_med=st.median([m["loc"] for m in ms.values()]) if ms else None, loc_min=min(m["loc"] for m in ms.values()) if ms else None,
                 cc_med=st.median([m["cyclomatic"] for m in ms.values() if m.get("parse_ok")]) if ms else None)

def flat_row(r):
    o = dict(era=r["era"], arm=SHORT[r["arm"]], arm_full=r["arm"], fam=FAM[r["arm"]], stage=STAGE[r["arm"]], task=r["task"], status=r["status"],
             score=r["score"] if r["score"] is not None else 0.0, score_raw_none=r["score"] is None,
             own_score=(r.get("own_score") if r.get("own_score") is not None else (0.0 if not r.get("no_trajectory") else 0.0)),
             stale_row=(r.get("lb_selected_in_run") is False), no_traj=bool(r.get("no_trajectory")), ctx400=bool(r.get("ctx400")),
             never_tested=bool(r.get("never_tested")) if not r.get("no_trajectory") else True)
    c = r.get("counters", {})
    o.update(steps=c.get("steps"), n_test=c.get("test_ok", 0), n_edit_ok=c.get("edit_ok", 0), n_edit_err=c.get("edit_err", 0), n_edit_unparsable=c.get("edit_ok_unparsable", 0), n_undo=c.get("undo_ok", 0), n_undo_err=c.get("undo_err", 0),
             n_view=c.get("view", 0), n_other_file=c.get("edit_other_file", 0), submit_ok=c.get("submit_ok", 0), completion_tokens=r.get("completion_tokens"), replay_ok=r.get("replay_ok"), scored_k=r.get("scored_k"), scored_source=r.get("scored_source"), final_source=r.get("final_source"))
    bm = BM[r["task"]]
    for which in ("final", "scored"):
        m = (r.get("metrics") or {}).get(which)
        if not m: 
            for k in ("loc","cc","ast","nest","nlit","nimp","parse","sim_tpl","kept_tpl","sim_bl","ident_bl","nearest_bl","novelty","ident_tpl","loc_ratio_bl","loc_ratio_tpl","cc_ratio_bl","below_min_bl"): o[f"{which}_{k}"] = None
            continue
        rm = m["region_metrics"]; stp = m.get("sim_template") or {}; snb = m.get("sim_nearest_baseline") or {}
        o[f"{which}_loc"] = rm["loc"]; o[f"{which}_cc"] = rm.get("cyclomatic"); o[f"{which}_ast"] = rm.get("ast_nodes"); o[f"{which}_nest"] = rm.get("max_nesting"); o[f"{which}_nlit"] = rm.get("n_num_literals"); o[f"{which}_nimp"] = rm.get("n_imports")
        o[f"{which}_parse"] = rm["parse_ok"] and m["whole_parse_ok"]
        o[f"{which}_sim_tpl"] = stp.get("tok_ratio"); o[f"{which}_kept_tpl"] = stp.get("lines_kept_frac"); o[f"{which}_sim_bl"] = snb.get("tok_ratio"); o[f"{which}_ident_bl"] = snb.get("ident_jaccard"); o[f"{which}_nearest_bl"] = m.get("nearest_baseline")
        o[f"{which}_novelty"] = 1 - max(stp.get("tok_ratio", 0), snb.get("tok_ratio", 0)) if stp else None
        o[f"{which}_ident_tpl"] = m["identical_to_template"]
        o[f"{which}_loc_ratio_bl"] = rm["loc"] / bm["loc_med"] if bm["loc_med"] else None; o[f"{which}_loc_ratio_tpl"] = rm["loc"] / max(1, bm["template"]["loc"])
        o[f"{which}_cc_ratio_bl"] = (rm.get("cyclomatic") / bm["cc_med"]) if (rm.get("cyclomatic") is not None and bm["cc_med"]) else None
        o[f"{which}_below_min_bl"] = rm["loc"] < bm["loc_min"] if bm["loc_min"] else None
    return o

rows = [flat_row(r) for r in runs]
with open(f"{OUT}/flat.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- helpers ----------------
def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]; return st.median(xs) if xs else None
def mean(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def pct(xs):
    xs = [x for x in xs if x is not None]; return f"{100*sum(bool(x) for x in xs)/len(xs):.0f}%" if xs else "–"
L = []
def P(s=""): L.append(s)
def table(header, rws):
    P("| " + " | ".join(header) + " |"); P("|" + "|".join("---" for _ in header) + "|")
    for r in rws: P("| " + " | ".join(str(x) for x in r) + " |")
    P()

def paired(a_rows, b_rows, key, higher_better=True):
    """a vs b paired by (era, task); returns dict with n, n_pos, n_neg, n_tie, median diff, wilcoxon p, sign-test p, mean a, mean b"""
    A = {(r["era"], r["task"]): r[key] for r in a_rows}; B = {(r["era"], r["task"]): r[key] for r in b_rows}
    ks = [k for k in A if k in B and A[k] is not None and B[k] is not None]
    d = [A[k] - B[k] for k in ks]
    pos = sum(x > 0 for x in d); neg = sum(x < 0 for x in d); tie = len(d) - pos - neg
    wp = stats.wilcoxon(d, zero_method="wilcox").pvalue if (pos + neg) >= 6 else None
    sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) > 0 else None
    return dict(n=len(d), pos=pos, neg=neg, tie=tie, med_diff=med(d), mean_a=mean([A[k] for k in ks]), mean_b=mean([B[k] for k in ks]), wilcoxon_p=wp, sign_p=sp,
                per_task={k: A[k] - B[k] for k in ks})

def sel(**kw):
    out = rows
    for k, v in kw.items():
        if isinstance(v, (list, tuple, set)): out = [r for r in out if r[k] in v]
        else: out = [r for r in out if r[k] == v]
    return out

# ================= 0. data quality =================
P("## 0. 数据来源与自检"); P()
P(f"- 运行数 {len(rows)}(2 代 × 12 臂 × 21 题);无轨迹(旧代 worker 缺 causallearn/deap,agent 未启动):{sum(r['no_traj'] for r in rows)}")
P(f"- 编辑栈回放 == 工作区最终文件:{sum(1 for r in rows if r['replay_ok'] is True)} 条一致,0 条不一致;{sum(1 for r in rows if r['replay_ok'] is None and not r['no_traj'])} 条无法校验(快照缺失且 str_replace 不能精确复现,直接采用工作区文件)")
P(f"- 模板文本:每题由“从未成功编辑”的运行的工作区文件确定,21 题全部唯一一致(见 runs.json templates)")
P(f"- 排行榜复现:对 504 条,按 evaluate.py 的选行规则复算得到的分数与 summary.json 官方分数 **全部一致**(504/504)")
stale = [r for r in rows if r["stale_row"]]
P(f"- **旧代污染**:{len(stale)} 条旧代运行,`mlsbench score` 选中的排行榜行**不是本次运行写的**(同名 model 在更早的作业里留下的有效行被选中);其中 {sum(1 for r in stale if abs(r['score']-r['own_score'])>1e-6)} 条分数因此不同。p1 代 model 名带 `_p1` 后缀,无此问题。")
P()
table(["era","arm","task","官方分","仅本次运行的行"], [[r["era"], r["arm"], r["task"], f(r["score"]), f(r["own_score"])] for r in stale if abs(r["score"] - r["own_score"]) > 1e-6])
P("旧代各臂均值(21 题口径,缺/崩=0):官方 vs 仅本运行行")
rws = []
for a in ARMS9 + ARMS4:
    rs = sel(era="old", arm_full=a); rws.append([SHORT[a], f(mean([r["score"] for r in rs]), 4), f(mean([r["own_score"] for r in rs]), 4), sum(r["stale_row"] for r in rs)])
table(["arm","official mean","own-run mean","stale rows"], rws)

# ================= 1. outcome decomposition =================
P("## 1. 结果分解:每臂 21 题里发生了什么"); P()
for era in ("old", "p1"):
    P(f"### {era}"); rws = []
    for a in ARMS9 + ARMS4:
        rs = sel(era=era, arm_full=a)
        rws.append([SHORT[a], f(mean([r["score"] for r in rs]), 4), sum(r["no_traj"] for r in rs), sum(r["ctx400"] for r in rs), sum(r["never_tested"] and not r["no_traj"] for r in rs),
                    sum((not r["never_tested"]) and r["score"] == 0 for r in rs), sum(r["score"] > 0 for r in rs), sum(bool(r["final_ident_tpl"]) for r in rs if r["final_ident_tpl"] is not None),
                    f(med([r["n_test"] for r in rs]), 0), f(med([r["n_edit_ok"] for r in rs]), 0), f(med([r["n_edit_err"] for r in rs]), 0), f(med([r["completion_tokens"] for r in rs]), 0)])
    table(["arm","mean","no traj","ctx400","never tested","tested but 0","score>0","final==template","med tests","med edit ok","med edit err","med compl. tokens"], rws)

# ================= 2. simplicity =================
P("## 2. 简洁性(simplicity):最终提交的可编辑区代码"); P()
P("指标:LOC(非空非注释行)、圈复杂度(自写 AST 访问器)、AST 节点数、最大嵌套、数值字面量数;`LOC/基线中位` = 该题 LOC 除以任务自带基线实现的 LOC 中位数;`<最小基线` = LOC 比最短基线还短的比例。仅统计有轨迹、最终文件≠模板的运行(模板不算“写了代码”)。")
P()
for era in ("old", "p1", "both"):
    P(f"### {era}"); rws = []
    for a in ARMS9 + ARMS4:
        rs = [r for r in (sel(arm_full=a) if era == "both" else sel(era=era, arm_full=a)) if not r["no_traj"] and r["final_ident_tpl"] is False]
        rws.append([SHORT[a], len(rs), f(med([r["final_loc"] for r in rs]), 0), f(med([r["final_loc_ratio_bl"] for r in rs]), 2), pct([r["final_below_min_bl"] for r in rs]), f(med([r["final_cc"] for r in rs]), 0), f(med([r["final_cc_ratio_bl"] for r in rs]), 2),
                    f(med([r["final_ast"] for r in rs]), 0), f(med([r["final_nest"] for r in rs]), 0), f(med([r["final_nlit"] for r in rs]), 0), pct([r["final_parse"] for r in rs])])
    table(["arm","n","med LOC","med LOC/基线中位","<最小基线","med CC","med CC/基线","med AST","med nest","med #num","parse ok"], rws)

# ================= 3. similarity =================
P("## 3. 相似度(similarity):与模板、与任务自带基线"); P()
P("指标:`sim_tpl` = 可编辑区 token 序列 difflib ratio vs 模板区;`kept_tpl` = 模板区非空行被保留的比例;`sim_bl` = 与**最相近**基线实现(基线 replace 应用到模板后的可编辑区)的 token ratio;`ident_bl` = 与最相近基线的标识符 Jaccard;`novelty` = 1 − max(sim_tpl, sim_bl)。同样只算最终文件≠模板的运行。")
P()
for era in ("old", "p1", "both"):
    P(f"### {era}"); rws = []
    for a in ARMS9 + ARMS4:
        rs = [r for r in (sel(arm_full=a) if era == "both" else sel(era=era, arm_full=a)) if not r["no_traj"] and r["final_ident_tpl"] is False]
        nb = collections.Counter(r["final_nearest_bl"] for r in rs)
        rws.append([SHORT[a], len(rs), f(med([r["final_sim_tpl"] for r in rs])), f(med([r["final_kept_tpl"] for r in rs])), f(med([r["final_sim_bl"] for r in rs])), f(med([r["final_ident_bl"] for r in rs])), f(med([r["final_novelty"] for r in rs])),
                    pct([r["final_sim_bl"] > 0.6 for r in rs]), pct([r["final_novelty"] > 0.5 for r in rs])])
    table(["arm","n","med sim_tpl","med kept_tpl","med sim_bl","med ident_bl","med novelty","sim_bl>0.6","novelty>0.5"], rws)

# ================= 4. paired tests =================
P("## 4. 配对检验(按 题×代 配对;两代 = 两次独立运行)"); P()
P("每格:n 对 / 正−负−平 / 差的中位 / Wilcoxon p / 符号检验 p。指标含分数(官方 21 题口径)、是否得到非零分、是否从未 test、LOC、圈复杂度、sim_tpl、sim_bl、novelty。代码指标只在**两边都有非模板最终文件**的配对上比较。")
P()
PAIRS = [("rlv5_ft01mix_a10_s20", "ft01mix_a10"), ("rlv5_ft03nm_a20_s20", "ft03nm_a20"), ("rlv5_lo32nm_a10_s20", "lo32nm_a10"), ("rlv5_base_s20", "base9b_v2c"),
         ("ft01mix_a10", "base9b_v2c"), ("ft03nm_a20", "base9b_v2c"), ("lo32nm_a10", "base9b_v2c"),
         ("rlv5_4b_base_s20", "base4b"), ("rlv5_4b_ft01mix_a10_s20", "4b_ft01mix_a10"), ("4b_ft01mix_a10", "base4b")]
METRICS = [("score", "score"), ("nonzero", "score>0"), ("never_tested", "never tested"), ("final_loc", "LOC"), ("final_cc", "CC"), ("final_sim_tpl", "sim_tpl"), ("final_sim_bl", "sim_bl"), ("final_novelty", "novelty"), ("n_edit_err", "edit errors"), ("completion_tokens", "compl. tokens")]
for r in rows: r["nonzero"] = 1.0 if r["score"] > 0 else 0.0; r["never_tested"] = float(r["never_tested"])
pair_results = {}
for a, b in PAIRS:
    P(f"### {SHORT[a]} − {SHORT[b]}"); rws = []
    for key, name in METRICS:
        ra = sel(arm_full=a); rb = sel(arm_full=b)
        if key.startswith("final_"):
            ra = [r for r in ra if not r["no_traj"] and r["final_ident_tpl"] is False]; rb = [r for r in rb if not r["no_traj"] and r["final_ident_tpl"] is False]
        res = paired(ra, rb, key); pair_results[(a, b, key)] = res
        rws.append([name, res["n"], f"{res['pos']}−{res['neg']}−{res['tie']}", f(res["med_diff"], 3), f(res["wilcoxon_p"], 3), f(res["sign_p"], 3), f(res["mean_a"], 3), f(res["mean_b"], 3)])
    table(["metric","n","+/−/=","med diff","Wilcoxon p","sign p", f"mean {SHORT[a]}", f"mean {SHORT[b]}"], rws)
# pooled RL-from-SFT vs SFT
P("### 三对 RL(SFT 起点) − SFT 合并(n≤126)"); rws = []
for key, name in METRICS:
    ds = []; A_all = []; B_all = []
    for a, b in PAIRS[:3]:
        res = pair_results[(a, b, key)]; ds += list(res["per_task"].values())
    pos = sum(x > 0 for x in ds); neg = sum(x < 0 for x in ds); tie = len(ds) - pos - neg
    wp = stats.wilcoxon(ds, zero_method="wilcox").pvalue if pos + neg >= 6 else None; sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else None
    rws.append([name, len(ds), f"{pos}−{neg}−{tie}", f(med(ds), 3), f(wp, 3), f(sp, 3)])
table(["metric","n","+/−/=","med diff","Wilcoxon p","sign p"], rws)
# per-task decomposition for pooled score and sim_bl, novelty
P("### 逐题分解:三对 RL−SFT 的差(每题 3 对 × 2 代 = 6 个差;列出正/负个数与中位差)"); rws = []
for t in TASKS:
    cells = [t]
    for key in ("score", "final_sim_bl", "final_novelty", "final_loc"):
        ds = [pair_results[(a, b, key)]["per_task"][k] for a, b in PAIRS[:3] for k in pair_results[(a, b, key)]["per_task"] if k[1] == t]
        cells.append(f"{sum(x>0 for x in ds)}+/{sum(x<0 for x in ds)}− med {f(med(ds),2)}" if ds else "–")
    rws.append(cells)
table(["task","score","sim_bl","novelty","LOC"], rws)

# ================= 5. within-task association: outcome vs code properties =================
P("## 5. 题内关联:代码性质 ↔ 结果(把 24 次运行在题内做秩,合并 21 题)"); P()
def within_task_rank(key, subset):
    xs = []; ys = []
    for t in TASKS:
        rs = [r for r in subset if r["task"] == t and r[key] is not None]
        if len(rs) < 4: continue
        rk = stats.rankdata([r[key] for r in rs]); sk = stats.rankdata([r["score"] for r in rs])
        xs += list(rk / len(rs)); ys += list(sk / len(rs))
    return xs, ys
subset = [r for r in rows if not r["no_traj"] and r["final_ident_tpl"] is False]
rws = []
for key, name in [("final_loc", "LOC"), ("final_cc", "CC"), ("final_ast", "AST nodes"), ("final_nest", "nesting"), ("final_nlit", "#num literals"), ("final_sim_tpl", "sim_tpl"), ("final_kept_tpl", "kept_tpl"), ("final_sim_bl", "sim_bl"), ("final_ident_bl", "ident_bl"), ("final_novelty", "novelty"), ("n_edit_err", "edit errors"), ("n_test", "#tests"), ("completion_tokens", "compl. tokens")]:
    xs, ys = within_task_rank(key, subset)
    rho, p = stats.spearmanr(xs, ys)
    rws.append([name, len(xs), f(rho, 3), f(p, 4)])
table(["property (within-task rank)","n runs","Spearman rho vs score rank","p"], rws)
P("按结果分层(非模板最终文件的运行):")
rws = []
for lab, cond in [("score>0", lambda r: r["score"] > 0), ("tested, score=0", lambda r: r["score"] == 0 and not r["never_tested"]), ("never tested (=0)", lambda r: r["never_tested"] == 1.0)]:
    rs = [r for r in subset if cond(r)]
    rws.append([lab, len(rs), f(med([r["final_loc"] for r in rs]), 0), f(med([r["final_loc_ratio_bl"] for r in rs]), 2), f(med([r["final_cc"] for r in rs]), 0), f(med([r["final_sim_tpl"] for r in rs])), f(med([r["final_sim_bl"] for r in rs])), f(med([r["final_novelty"] for r in rs])), pct([r["final_parse"] for r in rs]), f(med([r["n_edit_err"] for r in rs]), 0)])
table(["outcome","n","med LOC","med LOC/基线","med CC","med sim_tpl","med sim_bl","med novelty","parse ok","med edit err"], rws)
P("novelty 三分位(题内)× 结果:")
tert = collections.defaultdict(lambda: collections.Counter())
for t in TASKS:
    rs = [r for r in subset if r["task"] == t and r["final_novelty"] is not None]
    if len(rs) < 6: continue
    q = np.quantile([r["final_novelty"] for r in rs], [1/3, 2/3])
    for r in rs:
        b = "low" if r["final_novelty"] <= q[0] else ("mid" if r["final_novelty"] <= q[1] else "high")
        tert[b]["n"] += 1; tert[b]["score>0"] += r["score"] > 0; tert[b]["never_tested"] += r["never_tested"] == 1.0; tert[b]["parse_fail"] += not r["final_parse"]
table(["novelty tercile","n","score>0","never tested","final unparsable"], [[b, tert[b]["n"], f"{tert[b]['score>0']} ({100*tert[b]['score>0']/tert[b]['n']:.0f}%)", tert[b]["never_tested"], tert[b]["parse_fail"]] for b in ("low", "mid", "high")])

# ================= 6. stage-level summary (pooled arms) =================
P("## 6. 按阶段合并(9B:base / SFT×3 / RL-from-base / RL-from-SFT×3;两代合并)"); P()
rws = []
for fam in ("9B", "4B"):
    for stg in ("base", "sft", "rl_base", "rl_sft"):
        rs = sel(fam=fam, stage=stg); rs2 = [r for r in rs if not r["no_traj"] and r["final_ident_tpl"] is False]
        if not rs: continue
        rws.append([fam, stg, len(rs), f(mean([r["score"] for r in rs]), 4), pct([r["score"] > 0 for r in rs]), pct([r["never_tested"] for r in rs]), pct([r["final_ident_tpl"] for r in rs if r["final_ident_tpl"] is not None]),
                    f(med([r["final_loc"] for r in rs2]), 0), f(med([r["final_loc_ratio_bl"] for r in rs2]), 2), f(med([r["final_cc"] for r in rs2]), 0), f(med([r["final_sim_tpl"] for r in rs2])), f(med([r["final_sim_bl"] for r in rs2])), f(med([r["final_novelty"] for r in rs2])), pct([r["final_parse"] for r in rs2]), f(med([r["n_edit_err"] for r in rs2]), 0)])
table(["fam","stage","n runs","mean score","score>0","never tested","final==tpl","med LOC","LOC/基线","med CC","sim_tpl","sim_bl","novelty","parse ok","edit err"], rws)

open(f"{OUT}/report_tables.md", "w").write("\n".join(L))
json.dump({f"{a}|{b}|{k}": {kk: vv for kk, vv in v.items() if kk != "per_task"} for (a, b, k), v in pair_results.items()}, open(f"{OUT}/pair_results.json", "w"), indent=1)
print("\n".join(L))
