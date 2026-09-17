"""补测失败的原因分类:模型的方法本身跑不起来,还是我们的环境问题。

只看最后一段 traceback 的最后一行(踩过一次:看第一行会把 harness 的包装异常
当成死因)。分五类:
  语法错     —— 模型把 tool-call 标记之类的东西写进了源文件
  导入错     —— 用了容器里没有的包
  运行错     —— 真的跑起来了但算崩了(形状/类型/数值)
  超时       —— 撞 test_cmd 的时间预算
  无输出     —— 跑通了但 parser 一个指标都没解析出来
"""
import json
import re
from pathlib import Path

D = Path("/scratch/gpfs/CHIJ/ziran/innov_v2_multi")
RES = D / "outputs" / "rescore_al1"
ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]


def last_exc(tail: str) -> str:
    lines = [l.rstrip() for l in (tail or "").splitlines() if l.strip()]
    for l in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Warning|Interrupt)\b", l):
            return l
    return lines[-1] if lines else ""


def classify(cell: dict) -> tuple[str, str]:
    if cell["metrics"]:
        return "有指标", ""
    why = []
    for lbl, c in cell["per_cmd"].items():
        if c["rc"] == -9:
            why.append(("超时", f"{lbl}: 撞 {c['budget']}s 预算"))
            continue
        e = last_exc(c["tail"])
        if c["rc"] == 0:
            why.append(("无输出", f"{lbl}: rc=0 但解析不出指标"))
        elif e.startswith(("SyntaxError", "IndentationError", "TabError")):
            why.append(("语法错", f"{lbl}: {e[:110]}"))
        elif e.startswith(("ImportError", "ModuleNotFoundError")):
            why.append(("导入错", f"{lbl}: {e[:110]}"))
        else:
            why.append(("运行错", f"{lbl}: {e[:110]}"))
    order = ["语法错", "导入错", "超时", "运行错", "无输出"]
    kinds = {k for k, _ in why}
    top = next((o for o in order if o in kinds), "运行错")
    msg = next(m for k, m in why if k == top)
    return top, msg


def main():
    rows = {}
    for tag, name in ARMS:
        d = RES / tag
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            j = json.load(open(f))
            rows[(tag, j["task"])] = classify(j)

    kinds = ["有指标", "语法错", "导入错", "超时", "运行错", "无输出"]
    print("## 补测 71 个「有方法但记 0」的格子:跑得起来吗\n")
    print("| 臂 | 补测 | " + " | ".join(kinds) + " |")
    print("|---|---:|" + "---:|" * len(kinds))
    tot = {k: 0 for k in kinds}
    for tag, name in ARMS:
        cs = [v[0] for (t, _), v in rows.items() if t == tag]
        if not cs:
            continue
        c = {k: cs.count(k) for k in kinds}
        for k in kinds:
            tot[k] += c[k]
        print(f"| {name} | {len(cs)} | " + " | ".join(str(c[k]) for k in kinds) + " |")
    print(f"| **合计** | **{sum(tot.values())}** | " + " | ".join(f"**{tot[k]}**" for k in kinds) + " |")

    print("\n## 逐格死因\n")
    print("| 臂 | 题 | 类 | 最后一行 |")
    print("|---|---|---|---|")
    for tag, name in ARMS:
        for (t, task), (k, msg) in sorted(rows.items()):
            if t != tag or k == "有指标":
                continue
            print(f"| {name} | `{task}` | {k} | `{msg[:120]}` |")


if __name__ == "__main__":
    main()
