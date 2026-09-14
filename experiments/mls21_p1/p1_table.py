"""Re-score the 21-task MLS set for the _p1 reruns via `mlsbench score` (one call per task, all models),
reconcile with the old 09-02/03 runs (summary.json per-task scores), and write the markdown table.
Missing / crashed task = 0 over a fixed 21-task denominator, for both eras."""
import json, subprocess, sys, os, glob
from concurrent.futures import ThreadPoolExecutor
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
S = os.path.dirname(os.path.abspath(__file__))
PY = f"{D}/envs/client/bin/python"
TASKS = "causal-discovery-discrete causal-observational-linear-gaussian causal-observational-linear-non-gaussian causal-observational-nonlinear causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy optimization-hyperparameter-search optimization-multi-objective optimization-nas optimization-online-bandit".split()
ARMS9 = "base9b_v2c ft01mix_a10 ft03nm_a20 lo32nm_a10 rlv5_base_s20 rlv5_ft01mix_a10_s20 rlv5_ft03nm_a20_s20 rlv5_lo32nm_a10_s20".split()
ARMS4 = "base4b 4b_ft01mix_a10 rlv5_4b_base_s20 rlv5_4b_ft01mix_a10_s20".split()
ARMS = ARMS9 + ARMS4
OLD_MEAN = {"base9b_v2c":0.1141,"ft01mix_a10":0.1425,"ft03nm_a20":0.1430,"lo32nm_a10":0.1525,"rlv5_base_s20":0.1513,
            "rlv5_ft01mix_a10_s20":0.1572,"rlv5_ft03nm_a20_s20":0.1760,"rlv5_lo32nm_a10_s20":0.1806,
            "base4b":0.0304,"4b_ft01mix_a10":0.0072,"rlv5_4b_base_s20":0.1223,"rlv5_4b_ft01mix_a10_s20":0.1088}

def score_task(task):
    out = subprocess.run([PY, "-m", "mlsbench", "score", task, "--format", "json"], cwd=f"{D}/mlsroot",
                         capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONPATH": f"{D}/mlsroot/src", "MLSBENCH_ROOT": f"{D}/mlsroot", "MLSBENCH_DATA_ROOT": f"{D}/mlsvendor/data"})
    try:
        rows = json.loads(out.stdout)[task]
    except Exception as e:
        return task, {"__error__": out.stdout[-300:] + out.stderr[-300:]}
    return task, {r["model"]: r["task_score"] for r in rows}

def main():
    with ThreadPoolExecutor(6) as ex:
        allscores = dict(ex.map(score_task, TASKS))
    json.dump(allscores, open(f"{S}/p1_scores_raw.json", "w"), indent=1)
    # old per-task from summary.json
    old, oldstat, p1stat, p1sum = {}, {}, {}, {}
    for a in ARMS:
        for era, key in (("old", a), ("p1", a + "_p1")):
            p = f"{D}/outputs/cc_mls21_{key}/summary.json"
            if not os.path.exists(p): continue
            d = json.load(open(p))
            for x in d["tasks"]:
                (old if era == "old" else p1sum)[(a, x["task"])] = x["score"]
                (oldstat if era == "old" else p1stat)[(a, x["task"])] = x["status"]
    def new(a, t):
        v = allscores.get(t, {}).get(f"vllm/{a}_p1")
        return 0.0 if v is None else float(v)
    def oldv(a, t):
        v = old.get((a, t)); return 0.0 if v is None else float(v)
    def done(a):
        return all((a, t) in p1sum for t in TASKS)
    L = []
    L.append("# MLS-21, `_p1` 重跑(2026-09-14)对 09-02/03 旧跑:21 题口径,缺题/崩溃 = 0\n")
    L.append("口径:系统提示含 “It is now year 2026.”;MAX_MODEL_LEN 40960、CONCURRENCY 7、TASK_TIMEOUT 7200;worker 用 client python(有 causallearn/deap);"
             "vendor 重建后 ml-active-learning 两处环境故障已修(pandas-3 pickle;pkg_configs 链接悬空导致 strategy.py 首行被抹)。"
             "p1 分数用 `mlsbench score <task>` 按 leaderboard 重算(ml-active-learning 六臂靠单题补跑写行);旧分数取旧跑 summary.json。每臂一次运行。\n")
    for grp, arms in (("9B(4 主臂 + 4 RL 臂)", ARMS9), ("4B", ARMS4)):
        L.append(f"\n## {grp}:总分(21 题均值)\n")
        L.append("| arm | old (21) | p1 (21) | diff | p1 scored/21 | p1 ctx-overflow crashes | p1 timeouts | p1 complete |")
        L.append("|---|---:|---:|---:|---:|---:|---:|:--|")
        for a in arms:
            o = sum(oldv(a, t) for t in TASKS) / 21; n = sum(new(a, t) for t in TASKS) / 21
            ns = sum(1 for t in TASKS if new(a, t) > 0)
            logs = glob.glob(f"{D}/outputs/cc_mls21_{a}_p1/task_logs/*.log")
            ctx = sum(1 for f in logs if "maximum context length" in open(f, errors="replace").read())
            to = sum(1 for f in logs if "### TIMEOUT after" in open(f, errors="replace").read())
            L.append(f"| {a} | {o:.4f} | {n:.4f} | {n-o:+.4f} | {ns} | {ctx} | {to} | {'yes' if done(a) else 'PARTIAL'} |")
        L.append(f"\n## {grp}:逐题(old → p1)\n")
        L.append("| task | " + " | ".join(arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        for t in TASKS:
            L.append(f"| {t} | " + " | ".join(f"{oldv(a,t):.3f}→{new(a,t):.3f}" for a in arms) + " |")
        L.append(f"\n### {grp}:符号检验(逐题 p1−old,忽略两边都为 0 的题)\n")
        L.append("| arm | + | − | tied | 注 |")
        L.append("|---|---:|---:|---:|---|")
        for a in arms:
            pos = sum(1 for t in TASKS if new(a,t) > oldv(a,t) + 1e-9); neg = sum(1 for t in TASKS if new(a,t) < oldv(a,t) - 1e-9)
            tie = 21 - pos - neg
            L.append(f"| {a} | {pos} | {neg} | {tie} | 单次运行对单次运行,只作描述 |")
    # consistency: p1 recomputed vs p1 summary (non-AL tasks)
    mism = [(a, t, p1sum[(a,t)], new(a,t)) for a in ARMS for t in TASKS if t != "ml-active-learning" and (a,t) in p1sum
            and abs((p1sum[(a,t)] or 0.0) - new(a,t)) > 1e-6]
    L.append("\n## 一致性:p1 重算 vs 主跑 summary.json(除 ml-active-learning)\n")
    L.append(f"不一致 {len(mism)} 处" + (":" if mism else "。"))
    for m in mism: L.append(f"- {m[0]} / {m[1]}: summary={m[2]} recomputed={m[3]:.4f}")
    L.append("\n## 旧跑 21 题口径校验(脚本重算 vs 记录值)\n")
    for a in ARMS:
        o = sum(oldv(a, t) for t in TASKS) / 21
        L.append(f"- {a}: 重算 {o:.4f} vs 记录 {OLD_MEAN[a]:.4f}" + ("" if abs(o-OLD_MEAN[a]) < 6e-4 else "  **不一致**"))
    open(f"{S}/eight_arm_p1_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
if __name__ == "__main__":
    main()
