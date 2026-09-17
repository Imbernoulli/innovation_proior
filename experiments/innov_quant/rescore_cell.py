#!/usr/bin/env python
"""按「最终文件状态」重评一个 (臂, 题) 格子。

背景:MLS 的分数只认 agent 主动 finalize 的那一次 test。模型把方法写进了
custom_algorithm.py 却没调用 test/submit 时,harness 追加一条空 final 行,
该题记 0。用户裁决:给了方法就得测,不能不测就算 0。

本脚本不重新推理,只拿 agent 跑完后**留在 workspace 里的最终文件**,原样跑一遍
任务自己的 test_cmds(含 hidden),用任务自己的 host 端 parser 出指标。
产物写 $D/outputs/rescore_al1/<tag>/<task>.json,评分是另一支脚本的事。

复用 WorkspaceTools._build_apptainer_cmd,是为了让容器、bind、env 与当时一模一样
——手搓命令会漏掉 pkg_config 的 apptainer_flags / data_deps。
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
ROOT = D / "mlsroot"
sys.path.insert(0, str(ROOT / "src"))
OUT = D / "outputs" / "rescore_al1"
WSROOT = D / "outputs" / "rescore_ws"

from mlsbench.agent.tools import WorkspaceTools
from mlsbench.agent.parsers import load_parser


def saved_workspace(task: str, tag: str) -> Path | None:
    """agent 当时那次运行留下的 workspace(al1 世代,取最新的一个)。"""
    base = ROOT / "vendor" / "workspace" / task
    if not base.is_dir():
        return None
    cands = sorted((p for p in base.iterdir()
                    if p.is_dir() and p.name.startswith(f"vllm_{tag}_al1_")),
                   key=lambda p: p.name)
    return cands[-1] if cands else None


def secs(t, default=3600):
    p = str(t or "").split(":")
    try:
        if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + int(p[2])
        if len(p) == 2: return int(p[0])*60 + int(p[1])
        return int(float(p[0]))
    except Exception:
        return default


def main(task: str, tag: str):
    src = saved_workspace(task, tag)
    if src is None:
        print(f"[skip] 没有 workspace: {task} / {tag}"); return 2
    exp = f"{tag}__rescore"
    wtd = WSROOT / task / exp
    if wtd.exists():
        shutil.rmtree(wtd)
    wtd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, wtd, symlinks=True, ignore_dangling_symlinks=True)

    cfg = json.load(open(ROOT / "tasks" / task / "config.json"))
    # data_root / save_path 必须跟当时那批作业一致:作业里写死
    # MLSBENCH_DATA_ROOT=$D/mlsvendor/data,save_path=<out>/saves。默认值
    # (project_root/vendor/data) 虽然是同一份的软链,但 save_path 不给就不会
    # 注入 SAVE_PATH/OUTPUT_DIR,写 OUTPUT_DIR 的题会行为不同。
    save = D / "outputs" / "rescore_al1" / "_saves"
    save.mkdir(parents=True, exist_ok=True)
    gcfg = {"data_root": str(D / "mlsvendor" / "data"), "save_path": str(save),
            "container_runtime": "apptainer", "seeds": [42]}
    tools = WorkspaceTools(
        task_name=task,
        config_task=cfg,
        config_edit={},
        workspace_root=WSROOT,
        project_root=ROOT,
        max_tests=1,
        model_name=f"vllm/{tag}_al1",
        exp_name=exp,
        container_runtime="apptainer",
        seeds=[42],
        save_path=str(save),
        global_config=gcfg,
    )
    parser = load_parser(task, ROOT)
    metrics, per_cmd = {}, {}
    for entry in tools.test_cmd_entries:
        label = entry.get("label") or entry.get("cmd")
        cmd = tools._build_apptainer_cmd(entry, 42)
        budget = secs(entry.get("time"))
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=budget)
            raw, rc = (r.stdout or "") + (r.stderr or ""), r.returncode
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b"").decode("utf8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            rc = -9
        el = time.time() - t0
        got = {}
        if parser is not None:
            try:
                got = parser.parse(label, raw).metrics or {}
            except Exception as ex:
                got = {}
                raw += f"\n[rescore] parser 抛异常: {ex!r}"
        metrics.update(got)
        per_cmd[label] = dict(rc=rc, secs=round(el, 1), n_metrics=len(got),
                              budget=budget, tail=raw[-1500:])
        print(f"  [{label}] rc={rc} {el:.0f}s metrics={len(got)}")

    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(task=task, tag=tag, workspace=str(src), metrics=metrics,
                   per_cmd=per_cmd), open(d / f"{task}.json", "w"), indent=1)
    print(f"[ok] {tag}/{task}: {len(metrics)} 个指标 -> {d/(task+'.json')}")
    shutil.rmtree(wtd, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
