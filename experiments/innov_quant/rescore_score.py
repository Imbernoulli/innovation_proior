"""「最终文件状态」这把尺子:as-run 与 file-state 并排,永不合并。

as-run  = MLS 原样口径。agent 不主动 finalize 就记 0。
file-state = 用户裁决口径:agent 把方法写进了 workspace 却没测,我们替它测一遍
             (rescore_cell.py 的产物),把那条空 final 行换成真实指标再评分。

两把尺子都在影子根 $D/rescore_root 上跑,绝不碰 $D/mlsroot 的真排行榜
(排行榜按 model 名选行,串行污染踩过一次)。影子根必须有真的 src/,因为
mlsbench.PROJECT_ROOT 由模块自身位置推出、无视 MLSBENCH_ROOT;软链会被
resolve() 跟穿,所以只能实拷。
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
ROOT = D / "mlsroot"
SHADOW = D / "rescore_root"
RES = D / "outputs" / "rescore_al1"
PY = str(D / "envs" / "client" / "bin" / "python")

ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]

META = {"timestamp", "model", "is_final", "seed"}


def build_shadow(force=False):
    if SHADOW.exists() and not force:
        return
    if SHADOW.exists():
        shutil.rmtree(SHADOW)
    SHADOW.mkdir(parents=True)
    for sub in ("tasks", "src", "configs"):
        shutil.copytree(ROOT / sub, SHADOW / sub, ignore_dangling_symlinks=True, symlinks=False)
    chk = subprocess.run([PY, "-c", "import mlsbench;print(mlsbench.PROJECT_ROOT)"],
                         env=dict(os.environ, PYTHONPATH=str(SHADOW / "src")),
                         capture_output=True, text=True).stdout.strip()
    assert chk == str(SHADOW), f"影子 PROJECT_ROOT={chk},应为 {SHADOW}"


def tasks21():
    """21 题集合:以 al1 的 summary.json 为准。"""
    out = set()
    for tag, _ in ARMS:
        for d in (f"cc_mls21_{tag}_al1", f"cc_mls21_{tag}_al1-fix"):
            f = D / "outputs" / d / "summary.json"
            if not f.exists():
                continue
            j = json.load(open(f))
            ts = j.get("tasks", j)
            ts = list(ts.values()) if isinstance(ts, dict) else ts
            out |= {t["task"] for t in ts}
    return sorted(out)


def asrun_scores():
    s = {}
    for tag, _ in ARMS:
        cur = {}
        for d in (f"cc_mls21_{tag}_al1", f"cc_mls21_{tag}_al1-fix"):
            f = D / "outputs" / d / "summary.json"
            if not f.exists():
                continue
            j = json.load(open(f))
            ts = j.get("tasks", j)
            ts = list(ts.values()) if isinstance(ts, dict) else ts
            for t in ts:
                cur[t["task"]] = t.get("score")
        s[tag] = cur
    return s


def patch_leaderboard(task, tag, metrics):
    """把该臂那条空 final 行换成带真实指标的 final 行。返回是否改动。"""
    lb = SHADOW / "tasks" / task / "leaderboard.csv"
    if not lb.exists():
        return False
    model = f"vllm/{tag}_al1"
    with open(lb, newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        rows = list(r)
    unknown = [k for k in metrics if k not in cols]
    if unknown:
        print(f"    [warn] {task}/{tag}: 排行榜没有这些列 {unknown[:4]}")
    mine = [x for x in rows if x.get("model") == model]
    if not mine:
        return False
    # 丢掉这臂所有 final 行(as-run 里只有那条空的),再补一条真的。
    rows = [x for x in rows
            if not (x.get("model") == model and str(x.get("is_final", "")).lower() == "true")]
    new = {c: "" for c in cols}
    new["timestamp"] = "2026-09-17T00:00:00+00:00"
    new["model"] = model
    new["is_final"] = "true"
    new["seed"] = "42"
    for k, v in metrics.items():
        if k in cols:
            new[k] = v
    rows.append(new)
    with open(lb, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return True


def score(task, tag):
    out = subprocess.run([PY, "-m", "mlsbench", "score", task,
                          "--model", f"vllm/{tag}_al1", "--format", "json"],
                         cwd=str(SHADOW),
                         env=dict(os.environ, PYTHONPATH=str(SHADOW / "src")),
                         capture_output=True, text=True)
    try:
        j = json.loads(out.stdout)
    except Exception:
        return None
    rows = j.get(task) or []
    return rows[0].get("task_score") if rows else None


def main():
    build_shadow(force="--rebuild" in sys.argv)
    T21 = tasks21()
    asrun = asrun_scores()

    patched = {tag: [] for tag, _ in ARMS}
    empty = {tag: [] for tag, _ in ARMS}
    for tag, _ in ARMS:
        d = RES / tag
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            j = json.load(open(f))
            m = j.get("metrics") or {}
            if not m:
                empty[tag].append(j["task"])
                continue
            if patch_leaderboard(j["task"], tag, m):
                patched[tag].append(j["task"])

    fs = {}
    for tag, _ in ARMS:
        fs[tag] = {t: score(t, tag) for t in T21}

    print("## 逐臂:as-run vs file-state(al1,分母 21)\n")
    print("| 臂 | 补测格子 | as-run 均分 | file-state 均分 | Δ | as-run 非零 | file-state 非零 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    store = {}
    for tag, name in ARMS:
        a = [asrun[tag].get(t) or 0.0 for t in T21]
        b = [fs[tag].get(t) or 0.0 for t in T21]
        store[tag] = (a, b)
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        print(f"| {name} | {len(patched[tag])} | {ma:.4f} | {mb:.4f} | "
              f"{mb-ma:+.4f} | {sum(1 for x in a if x>0)}/21 | {sum(1 for x in b if x>0)}/21 |")

    print("\n## 对照(file-state 口径)\n")
    print("| 对照 | as-run Δ | file-state Δ |")
    print("|---|---:|---:|")
    for lo, hi, lbl in [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
                        ("ft01mix_a10", "rlv5_ft01mix_a10_s20", "9B 我们 − SFT"),
                        ("base9b_v2c", "rlv5_ft01mix_a10_s20", "9B 我们 − base"),
                        ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
                        ("4b_ft01mix_a10", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − SFT"),
                        ("base4b", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − base")]:
        da = sum(store[hi][0]) / 21 - sum(store[lo][0]) / 21
        db = sum(store[hi][1]) / 21 - sum(store[lo][1]) / 21
        print(f"| {lbl} | {da:+.4f} | {db:+.4f} |")

    nb = {t: n for t, n in [(tag, len(v)) for tag, v in empty.items()]}
    print("\n跑出来没有指标的格子(仍记 0):", {k: v for k, v in nb.items() if v})
    json.dump({"tasks": T21, "asrun": {t: store[t][0] for t, _ in ARMS},
               "filestate": {t: store[t][1] for t, _ in ARMS},
               "patched": patched, "empty": empty},
              open(RES / "rescore_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
