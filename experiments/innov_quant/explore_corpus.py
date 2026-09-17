"""@5 探索度的盲盒语料:每题一份,八臂各一块,块序打乱、只给字母。

问题:「@5 的时候我们的模型是否倾向于探索更多做法」。sim_self / self_sim 都是
字符串相似度,答不了这个 —— 两份写法不同的同一个贪心算法相似度很低,两份写法
相近的不同算法相似度很高。所以改成让标注者数「有几种不同做法」。

口径:
  - **横跨三个 benchmark**,且必须包含 research 那个(用户 2026-09-17 定的:
    「所有这种抽题都要横跨多个 benchmark,尤其要包含 research 的」)。只在
    FCS-research 上抽题,等于把「会不会探索」这件事只在一种题型上问了一遍。
  - 八臂每题正好 5 抽(按 (gt, sample_idx) 去重后核过)。
  - 一块 = 一个臂在一道题上的 5 抽的**最终答案**(</think> 之后),不给思维链。
  - **没产出可辨认方法的抽样自成一类**,这样坍塌的臂也进分母,不会被静默剔除。
  - 块序按题打乱,字母与臂的对应只写进 key.json,不进标注者看到的文件。

slug 前缀区分 bench:FCS-research 保持**裸名**(那 26 题已经标完,改名会让已有标注
全部挂空),新加的两个 bench 用 `fcs_` / `ale_`。bench 归属另写 `key_bench.json`,
不动 key.json 的形状 —— explore_agg.py 还在按老形状读它。
"""
import collections
import glob
import json
import os
import random
import re
import sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "./explore_blind"

ARMS = ["base9b_v2c", "ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "base4b", "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
LETTERS = "ABCDEFGH"

# (bench, 产出目录后缀, 题面 parquet, slug 前缀, 抽几题)
BENCHES = [
    # 末位 complete_only:八臂都得有 5 抽齐全才进候选。FCS-research 那 26 题是按
    # 历史口径(不过滤)抽的、并且已经标完,这里必须保持 False —— 一开过滤器候选集
    # 就变了,同一个种子打乱出的前 26 题也就变了,已有标注会整批挂空。
    ("frontiercs_research", "research_thinking_32k_vllm", "frontiercs/research.parquet", "", 26, False),
    ("frontiercs",          "thinking_32k_both_vllm",     "frontiercs/full.parquet",     "fcs_", 13, True),
    ("alebench",            "thinking_32k_both_vllm",     "alebench/full40.parquet",     "ale_", 13, True),
]


def load(arm, bench, sub):
    """dump2 的去重规则:按 (gt, sample_idx),分片排序后后来的覆盖先前的。"""
    ok = {}
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub}/shard_*/samples.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("data_source") != bench:
                continue
            k = (str(r["ground_truth"]), int(r.get("sample_idx", -1)))
            if r.get("error"):
                continue
            m = r.get("metrics") or {}
            s = m.get("score", r.get("score"))
            if s is None:
                continue
            ok[k] = dict(score=float(s), text=r.get("text") or "")
    return ok


def final_of(text):
    i = text.rfind("</think>")
    if i < 0:
        return ""                       # 思维链没收尾 = 没有最终答案
    return text[i + len("</think>"):].strip()


def statements(rel):
    out = {}
    try:
        import pandas as pd
        df = pd.read_parquet(f"{FS}/{rel}")
        for _, r in df.iterrows():
            out[str(r["reward_model"]["ground_truth"])] = r["prompt"][0]["content"]
    except Exception as e:
        print(f"[warn] 题面读不到 {rel}:{e!r}")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    kf, bf = f"{OUT}/key.json", f"{OUT}/key_bench.json"
    key = json.load(open(kf)) if os.path.exists(kf) else {}
    kb = json.load(open(bf)) if os.path.exists(bf) else {}
    made = collections.Counter()
    for bench, sub, pq, pre, nprob, complete_only in BENCHES:
        data = {a: load(a, bench, sub) for a in ARMS}
        # 八臂都得有这道题的 5 抽,否则块会缺
        probs = sorted({p for a in ARMS for (p, _) in data[a]})
        if complete_only:
            probs = [p for p in probs
                     if all(all((p, i) in data[a] for i in range(5)) for a in ARMS)]
        # FCS-research 那批是用**整数**种子 20260917 抽的,那 26 题已经标完。
        # 换成字符串种子(哪怕内容看起来一样)打乱结果完全不同,前 26 题会整批换人。
        rng = random.Random(20260917 if bench == "frontiercs_research"
                            else f"20260917|{bench}")
        rng.shuffle(probs)              # 固定种子打乱,再取前 N:子集嵌套
        stm = statements(pq)
        n = 0
        for p in probs:
            if n >= nprob:
                break
            slug = pre + re.sub(r"[^A-Za-z0-9_.-]+", "_", p)
            kb.setdefault(slug, bench)
            if slug in key:
                n += 1
                continue                # 已经生成过的题不重排,免得作废已有标注
            order = ARMS[:]
            # 每题用自己的种子,子集可嵌套:先跑 3 题再扩到 24 题不会改动前 3 题的块序。
            random.Random(f"20260917|{slug}").shuffle(order)
            lines = [f"# 题目 `{p}`(bench: {bench})", "",
                     "下面是 8 个互相独立的「块」,每块是同一个模型对这道题的 5 次独立抽样的**最终答案**。",
                     "块与块之间没有任何关系,不要跨块比较。", ""]
            if stm.get(p):
                lines += ["<details><summary>题面</summary>", "", "```",
                          stm[p][:6000], "```", "", "</details>", ""]
            key[slug] = {}
            for li, arm in zip(LETTERS, order):
                key[slug][li] = arm
                lines += [f"## 块 {li}", ""]
                for si in range(5):
                    v = data[arm].get((p, si))
                    fin = final_of(v["text"]) if v else ""
                    lines += [f"### 块 {li} · 抽样 {si + 1}", ""]
                    if not fin:
                        lines += ["(没有最终答案)", ""]
                    else:
                        lines += ["```", fin[:4000], "```", ""]
            open(f"{OUT}/{slug}.md", "w").write("\n".join(lines))
            made[bench] += 1
            n += 1
        print(f"[{bench}] 可用题 {len(probs)},本次覆盖 {n} 题,新生成 {made[bench]}")
    json.dump(key, open(kf, "w"), indent=1)
    json.dump(kb, open(bf, "w"), indent=1, ensure_ascii=False)
    print(f"[ok] 累计 {len(key)} 题 × 8 块 -> {OUT}(key.json / key_bench.json 不给标注者看)")


if __name__ == "__main__":
    main()
