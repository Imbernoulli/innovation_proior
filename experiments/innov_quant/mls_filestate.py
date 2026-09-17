# -*- coding: utf-8 -*-
"""MLS 的第二套口径:file-state。

as-run  = 现有口径。agent 必须显式 finalize,否则 record_zero_if_no_finals
          (tools.py:4548) 补一条空指标的 is_final=true 行 -> 排最差 -> 0 分。
          测的是「能不能独立走完一个 agentic 流程」。
file-state = 模型给了方法就得测。没 finalize 的格子,取它**最后一批**跑出真实
          指标的非 final 运行,提升为 final 再判分。测的是「留下的东西值多少分」。

取「最后一批」不是「最好的一批」:直接沿用 harness 自己的
_latest_valid_nonfinal_batch(tools.py:4292)语义,避免择优。
_has_real_metrics 判据也照抄 tools.py:4167(忽略 elapsed_* 与 *_std)。

原 leaderboard **只读**。所有改写发生在影子 root 的副本上,as-run 数据不受影响。
影子 root 用 mlsbench score 判分,已验证能逐位复现 as-run 的 task_score
(optimization-evolution-strategy / 4B ours al1 = 0.4865582582357749)。

用法: python mls_filestate.py <suffix>   # 例如 al1
"""
import csv, json, os, shutil, subprocess, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ROOT = f"{D}/mlsroot"
SHADOW = f"{D}/filestate_root"
PY = f"{D}/envs/client/bin/python"
META = {"timestamp", "model", "is_final", "seed"}

ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL(先验)", "rlv5_ft01mix_a10_s20"),
        ("4B base", "base4b"), ("4B SFT", "4b_ft01mix_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL(先验)", "rlv5_4b_ft01mix_a10_s20")]


def has_real(rec):
    for k, v in rec.items():
        if k in META or k.startswith("elapsed_") or k.endswith("_std"):
            continue
        if v in ("", None):
            continue
        return True
    return False


def as_run(tag):
    """summary.json 的 as-run 分,合并 -fix。"""
    out = {}
    for d in (f"{D}/outputs/cc_mls21_{tag}", f"{D}/outputs/cc_mls21_{tag}-fix"):
        p = f"{d}/summary.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        ts = j.get("tasks", j)
        ts = ts if isinstance(ts, list) else list(ts.values())
        for t in ts:
            out[t["task"]] = t.get("score")
    return out


def repair(task, model):
    """影子副本里把空 final 行换成最后一批有真实指标的非 final 行。
    -> (是否改过, 原因)"""
    src = f"{ROOT}/tasks/{task}/leaderboard.csv"
    dst = f"{SHADOW}/tasks/{task}/leaderboard.csv"
    if not os.path.exists(src):
        return False, "无 leaderboard"
    rows = list(csv.DictReader(open(src)))
    if not rows:
        return False, "空 leaderboard"
    mine = [r for r in rows if (r.get("model") or "") == model]
    finals = [r for r in mine if str(r.get("is_final", "")).lower() == "true"]
    if not finals:
        return False, "无 final 行"
    if any(has_real(r) for r in finals):
        return False, "已有有效 final"      # as-run 就是真分,不动
    # 最后一批有真实指标的非 final:按文件顺序取最后出现的 timestamp
    nf = [r for r in mine
          if str(r.get("is_final", "")).lower() == "false"
          and r.get("seed") != "mean" and has_real(r)]
    if not nf:
        return False, "没跑出过指标"        # 只能重新执行,本脚本不做
    last_ts = None
    for r in nf:
        ts = str(r.get("timestamp", ""))
        if ts:
            last_ts = ts
    batch = [r for r in nf if str(r.get("timestamp", "")) == last_ts]
    keep = [r for r in rows if r not in finals]        # 丢掉空 final 行
    for r in batch:
        new = dict(r); new["is_final"] = "true"
        keep.append(new)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(keep)
    return True, f"提升 {len(batch)} 行(ts={last_ts[-8:]})"


def score(task, model, root):
    env = dict(os.environ, PYTHONPATH=f"{root}/src", MLSBENCH_ROOT=root,
               MLSBENCH_DATA_ROOT=f"{D}/mlsvendor/data")
    try:
        out = subprocess.run([PY, "-m", "mlsbench", "score", task, "--model", model,
                              "--format", "json"], cwd=root, env=env,
                             capture_output=True, text=True, timeout=300).stdout
        d = json.loads(out)
        for e in d.get(task, []):
            if e.get("model") == model:
                return e.get("task_score")
    except Exception:
        return None
    return None


def main():
    suf = sys.argv[1] if len(sys.argv) > 1 else "p1"   # 主表口径:p1
    if os.path.exists(SHADOW):
        shutil.rmtree(SHADOW)
    # PROJECT_ROOT 是从 mlsbench/__init__.py 的位置往上三层推出来的,不看 MLSBENCH_ROOT。
    # 所以影子 root 必须自带一份**真实**的 src/(软链接不行:Path.resolve() 会跟回
    # mlsroot,结果就是打分读的还是原目录——第一版就是这么算出「Δ 全为 0」的假结果,
    # 是把该模型的行从影子里全删掉分数纹丝不动才发现的)。
    # vendor 重建在个别题里留下了名为 "*" 的悬空软链接,不忽略会整个 copytree 失败。
    shutil.copytree(f"{ROOT}/tasks", f"{SHADOW}/tasks", ignore_dangling_symlinks=True)
    shutil.copytree(f"{ROOT}/src", f"{SHADOW}/src", ignore_dangling_symlinks=True)
    for extra in ("configs", "vendor/pkg_configs", "vendor/packages.yaml"):
        s = f"{ROOT}/{extra}"
        if os.path.isdir(s):
            shutil.copytree(s, f"{SHADOW}/{extra}", dirs_exist_ok=True,
                            ignore_dangling_symlinks=True)
        elif os.path.isfile(s):
            os.makedirs(os.path.dirname(f"{SHADOW}/{extra}"), exist_ok=True)
            shutil.copy2(s, f"{SHADOW}/{extra}")
    # 自检:影子里的 PROJECT_ROOT 必须指向影子自己,否则下面全是假数
    chk = subprocess.run([PY, "-c", "import mlsbench;print(mlsbench.PROJECT_ROOT)"],
                         env=dict(os.environ, PYTHONPATH=f"{SHADOW}/src"),
                         capture_output=True, text=True).stdout.strip()
    assert chk == SHADOW, f"影子 PROJECT_ROOT={chk},应为 {SHADOW}"

    tasks = sorted(os.listdir(f"{ROOT}/tasks"))
    rows = []
    for lab, arm in ARMS:
        tag = f"{arm}_{suf}"
        ar = as_run(tag)
        if not ar:
            continue
        model = f"vllm/{tag}"
        for t in tasks:
            if t not in ar:
                continue
            ok, why = repair(t, model)
            a = ar[t] or 0.0
            f = score(t, model, SHADOW) if ok else a
            rows.append(dict(arm=lab, tag=tag, task=t, as_run=a,
                             file_state=(f if f is not None else a),
                             changed=ok, why=why))
    json.dump(rows, open(f"{os.path.dirname(os.path.abspath(__file__))}/mls_filestate_{suf}.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"# MLS 两套口径:as-run vs file-state(协议 `{suf}`,分母 21)\n")
    print("恢复规则:agent 没 finalize 但**已跑出过真实指标**的格子,取它最后一批非 final 运行提升为 final。")
    print("取「最后一批」而非「最好的一批」,沿用 harness 自己的 `_latest_valid_nonfinal_batch` 语义。")
    print("从没跑出过指标的格子本脚本不动(需重新执行 run_eval,那是另一次运行,不等于当时该得的分)。\n")
    print("| 臂 | as-run 均分/21 | file-state 均分/21 | Δ | 恢复格子数 |")
    print("|---|---|---|---|---|")
    for lab, arm in ARMS:
        rs = [r for r in rows if r["arm"] == lab]
        if not rs:
            continue
        a = sum(r["as_run"] for r in rs) / 21.0
        f = sum(r["file_state"] for r in rs) / 21.0
        n = sum(1 for r in rs if r["changed"])
        print(f"| {lab} | {a:.4f} | {f:.4f} | {f-a:+.4f} | {n} |")
    print("\n## 被恢复的格子\n")
    print("| 臂 | task | as-run | file-state | 说明 |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["changed"]:
            print(f"| {r['arm']} | `{r['task']}` | {r['as_run']:.4f} | {r['file_state']:.4f} | {r['why']} |")
    print("\n## 仍是 0 且从没跑出过指标(要重新执行才能判)\n")
    print("| 臂 | task |")
    print("|---|---|")
    for r in rows:
        if not r["changed"] and r["why"] == "没跑出过指标" and r["as_run"] == 0.0:
            print(f"| {r['arm']} | `{r['task']}` |")


if __name__ == "__main__":
    main()
