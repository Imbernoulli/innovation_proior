"""最小排行榜评分:只留 baseline 行 + 一条待评行。

为什么不能直接用活的 leaderboard.csv:
  - 它一直在被别的作业写(y2100 这批现在还在写),读改写整份 CSV,后写的覆盖先写的;
    09-16 那批 al1 的行已经有被抹掉的(causal-discovery-discrete 只剩 -fix 那一条)。
  - `evaluate_task` 选行的规则是「final 行里最完整的、再取最新」,同名模型多次运行
    的行混在一起,选中哪条取决于当时文件里有什么。
归一化只用 baseline 行(anchors.py 只认 model=baseline:*),所以
「baseline 全留 + 只放我要评的那一条」既去掉了污染,又和原口径等价。
"""
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
ROOT = D / "mlsroot"
SHADOW = D / "minilb_root"
PY = str(D / "envs" / "client" / "bin" / "python")
ENV = dict(os.environ, PYTHONPATH=str(SHADOW / "src"))


def build(force=False):
    if SHADOW.exists() and not force:
        return
    if SHADOW.exists():
        shutil.rmtree(SHADOW)
    SHADOW.mkdir(parents=True)
    for sub in ("tasks", "src", "configs"):
        shutil.copytree(ROOT / sub, SHADOW / sub, ignore_dangling_symlinks=True, symlinks=False)
    chk = subprocess.run([PY, "-c", "import mlsbench;print(mlsbench.PROJECT_ROOT)"],
                         env=ENV, capture_output=True, text=True).stdout.strip()
    assert chk == str(SHADOW), f"影子 PROJECT_ROOT={chk},应为 {SHADOW}"
    # 原始排行榜只读一次,存成 .orig,后面每次评分都从它重建。
    for t in (SHADOW / "tasks").iterdir():
        lb = t / "leaderboard.csv"
        if lb.exists():
            shutil.copy2(lb, t / "leaderboard.orig.csv")


def _orig(task):
    return SHADOW / "tasks" / task / "leaderboard.orig.csv"


def baselines(task):
    p = _orig(task)
    if not p.exists():
        return [], []
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        rows = [x for x in r if str(x.get("model", "")).startswith("baseline:")]
    return cols, rows


def score_row(task, model, row_metrics, tag="probe"):
    """只放 baseline + 一条 final 行,返回 task_score。"""
    cols, bl = baselines(task)
    if not cols:
        return None
    lb = SHADOW / "tasks" / task / "leaderboard.csv"
    new = {c: "" for c in cols}
    new["timestamp"] = "2026-09-17T00:00:00+00:00"
    new["model"] = model
    new["is_final"] = "true"
    new["seed"] = "42"
    for k, v in row_metrics.items():
        if k in cols:
            new[k] = v
    with open(lb, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(bl + [new])
    out = subprocess.run([PY, "-m", "mlsbench", "score", task, "--model", model,
                          "--format", "json"], cwd=str(SHADOW), env=ENV,
                         capture_output=True, text=True)
    try:
        j = json.loads(out.stdout)
    except Exception:
        return None
    rows = j.get(task) or []
    return rows[0].get("task_score") if rows else None


def live_rows(task, model):
    """活排行榜里这个模型的行(用于复现校验)。"""
    p = _orig(task)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [x for x in csv.DictReader(f) if x.get("model") == model]


META = {"timestamp", "model", "is_final", "seed"}


def metrics_of(row):
    return {k: v for k, v in row.items()
            if k not in META and not k.startswith("elapsed_")
            and not k.endswith("_std") and v not in ("", None)}
