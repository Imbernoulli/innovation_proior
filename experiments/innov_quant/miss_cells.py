"""缺格盘点:一个格子没有分,到底是谁的问题。

起因(用户 2026-09-17:「lora 的结果如果有问题或者不全的,修一下」):
`rlv5_lo32nm_a10_s20` 在 FCS-research 上 320 格里缺 9 格,是十四条臂里最多的。
逐条读错误串之后发现,这**不是判题坏了**,其中 8 格是模型自己把判题进程写崩的:

    self.index = faiss.IndexHNSWFlat(dim, 16, 80)      # ← rlv5_lo32nm_a10_s20 / low_latency / s2

`IndexHNSWFlat(d, M, metric)` 的第三个位置参数是 **metric**,不是 `ef_construction`。
传进去的 80 不是合法的 `faiss::MetricType`,检索时 `with_metric_type` 走到 default 分支
`FAISS_THROW`,C++ 异常穿过 SWIG 边界 → `std::terminate` → SIGABRT(rc=-6),整个
evaluator 进程被杀,所以「produced no result」。**提交是无效的,诚实的分数是 0,不是缺失。**

把这种格子从分母里丢掉,等于替犯错的臂免掉一次 0 分 —— 而且各臂犯错次数不一样
(lora 9B 8 次、我们 4 次、RL(base) 1 次),所以这个偏差**不是对称的**。

分类口径(只按落盘的错误串判,不猜):
  model   证据显示 evaluator 已经跑起来、死在模型自己的产物上 —— faiss 的 SIGABRT、
          solution.py 的语法错、PySR 的参数越界、模型写出的表达式解析不了。记 0。
  infra   证据显示是环境:资源文件没铺好、判题超时、写权限。保持缺失,不替模型背锅。
  unknown 落盘只留了 stderr 最后一行(julia 的 `PythonCall/.../C.jl:63`),
          从这一行分不出是模型的回调抛异常还是 depot 锁。保持缺失,并且单列出来。

用法:miss_cells.py  → 打印并写 miss_cells.md
"""
import json, os, re, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump2 import load, files, ARMS

MAIN = [a for a, v in ARMS.items() if v[1] != "rep"]
BENCHES = ["frontiercs_research", "frontiercs", "alebench"]
BN = {"frontiercs_research": "FCS-research", "frontiercs": "FrontierCS", "alebench": "ALE-Bench"}
NAME = {"base9b_v2c": "9B base", "ft01mix_a10": "9B SFT", "ft03nm_a20": "9B SFT ft03nm",
        "lo32nm_a10": "9B SFT lora", "rlv5_base_s20": "9B RL(base)",
        "rlv5_ft01mix_a10_s20": "9B RL 我们", "rlv5_ft03nm_a20_s20": "9B RL ft03nm",
        "rlv5_lo32nm_a10_s20": "9B RL lora", "base4b": "4B base", "4b_ft01mix_a10": "4B SFT",
        "4b_lo32nm_a10": "4B SFT lora", "rlv5_4b_base_s20": "4B RL(base)",
        "rlv5_4b_ft01mix_a10_s20": "4B RL 我们", "rlv5_4b_lo32nm_a10_s20": "4B RL lora"}

# 证据必须指向模型自己的产物。每一条都对着落盘错误串核过,不做正则外推。
MODEL = [
    (r"faiss::with_metric_type", "faiss SIGABRT(模型把 metric 位传成了 ef)"),
    (r"invalid syntax \(solution\.py", "solution.py 语法错"),
    (r"was never closed \(solution\.py", "solution.py 括号没闭合"),
    (r"closing parenthesis .* does not match", "solution.py 括号不配对"),
    (r"unexpected indent \(solution\.py", "solution.py 缩进错"),
    (r"PySR requires a maxsize of at least", "PySR 参数越界"),
    (r"`tournament_selection_n` parameter must be smaller", "PySR 参数越界"),
    (r"Failed to parse expression", "模型给的表达式解析不了"),
]
INFRA = [
    (r"No such file or directory", "资源/产物文件没铺好"),
    (r"timed out after", "判题超时"),
    (r"Permission denied", "写权限"),
]
UNKNOWN = [(r"julia_depot/packages/PythonCall", "只留了 julia 最后一行,分不出")]


def classify(msg):
    for pats, tag in ((MODEL, "model"), (INFRA, "infra"), (UNKNOWN, "unknown")):
        for p, why in pats:
            if re.search(p, msg):
                return tag, why
    return "unknown", "未匹配任何指纹"


def missing(arm, bench):
    """-> {(gt, si): (tag, why, msg)};  只含真的没有分的格子。"""
    ok, seen = load(arm, bench)
    probs = sorted({k[0] for k in seen})
    miss = set((p, i) for p in probs for i in range(5)) - set(ok)
    last = {}
    for f in files(arm, bench):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("data_source") != bench:
                continue
            k = (str(r["ground_truth"]), int(r.get("sample_idx", -1)))
            if k in miss and r.get("error"):
                last[k] = str(r["error"])
    out = {}
    for k in sorted(miss):
        m = last.get(k, "(没有 error 行)")
        t, w = classify(m)
        out[k] = (t, w, m)
    return out


def mean5(ok, probs, extra_zero=()):
    """逐题等权:先在题内对已有的抽样求均值,再对题求平均。extra_zero 里的格子按 0 计入。"""
    z = set(extra_zero)
    per = []
    for p in probs:
        v = [ok[(p, i)]["score"] for i in range(5) if (p, i) in ok]
        v += [0.0 for i in range(5) if (p, i) in z]
        if v:
            per.append(statistics.mean(v))
    return statistics.mean(per) if per else float("nan")


B = []
def emit(s=""):
    print(s)
    B.append(s)


def main():
    emit("# 缺格盘点:没有分的格子是谁的问题\n")
    emit("判据只看落盘的错误串,分三类:**model**(evaluator 跑起来了,死在模型自己的产物上——")
    emit("记 0)、**infra**(环境没铺好/超时/权限——保持缺失)、**unknown**(只留了 stderr 最后一行,")
    emit("分不出——保持缺失)。口径与理由见脚本 docstring。\n")

    allmiss = {}
    for b in BENCHES:
        for a in MAIN:
            allmiss[(a, b)] = missing(a, b)

    emit("## 1. 逐臂缺格(格子数 = 题 × 5 抽)\n")
    emit("| 臂 | " + " | ".join(f"{BN[b]} 缺 / 其中 model" for b in BENCHES) + " |")
    emit("|---|" + "---:|" * len(BENCHES))
    for a in MAIN:
        cells = []
        for b in BENCHES:
            m = allmiss[(a, b)]
            nm = sum(1 for v in m.values() if v[0] == "model")
            cells.append(f"{len(m)} / **{nm}**" if nm else f"{len(m)} / 0")
        emit(f"| {NAME[a]} `{a}` | " + " | ".join(cells) + " |")
    emit()

    emit("## 2. model 类缺格的具体死法\n")
    emit("| 臂 | bench | 题 | 抽样 | 死法 |")
    emit("|---|---|---|---:|---|")
    n = 0
    for b in BENCHES:
        for a in MAIN:
            for (p, i), (t, w, _) in sorted(allmiss[(a, b)].items()):
                if t == "model":
                    emit(f"| {NAME[a]} | {BN[b]} | `{p}` | {i} | {w} |")
                    n += 1
    emit(f"\n合计 **{n}** 格。\n")

    emit("## 3. 把 model 类记 0 之后,FCS-research 的 mean@5\n")
    emit("`现口径` = 现在主表用的(缺格直接从分母里去掉);`记0口径` = 只把上面那 %d 格记 0,"
         "infra / unknown 一格不动。\n" % n)
    emit("| 臂 | n题 | 现口径 mean@5 | 记0口径 mean@5 | Δ |")
    emit("|---|---:|---:|---:|---:|")
    cur, fix = {}, {}
    for a in MAIN:
        ok, seen = load(a, "frontiercs_research")
        probs = sorted({k[0] for k in seen})
        z = [k for k, v in allmiss[(a, "frontiercs_research")].items() if v[0] == "model"]
        cur[a] = mean5(ok, probs)
        fix[a] = mean5(ok, probs, z)
        emit(f"| {NAME[a]} `{a}` | {len(probs)} | {cur[a]:.3f} | {fix[a]:.3f} | {fix[a]-cur[a]:+.3f} |")
    emit()

    emit("## 4. 主对照在两种口径下的方向\n")
    emit("| 对照 | 现口径 Δ | 记0口径 Δ | 方向变了吗 |")
    emit("|---|---:|---:|:---:|")
    for lo, hi, lbl in [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
                        ("rlv5_base_s20", "rlv5_lo32nm_a10_s20", "9B lora − RL(base)"),
                        ("rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20", "9B lora − 我们"),
                        ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
                        ("rlv5_4b_base_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − RL(base)"),
                        ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − 我们")]:
        d0, d1 = cur[hi] - cur[lo], fix[hi] - fix[lo]
        emit(f"| {lbl} | {d0:+.3f} | {d1:+.3f} | {'**变了**' if (d0 > 0) != (d1 > 0) else '没变'} |")
    emit()

    emit("## 5. 保持缺失的那些(infra / unknown)\n")
    emit("| 臂 | bench | 题 | 抽样 | 类 | 理由 |")
    emit("|---|---|---|---:|---|---|")
    for b in BENCHES:
        for a in MAIN:
            for (p, i), (t, w, _) in sorted(allmiss[(a, b)].items()):
                if t != "model":
                    emit(f"| {NAME[a]} | {BN[b]} | `{p}` | {i} | {t} | {w} |")
    emit()


main()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "miss_cells.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(B) + "\n")
