"""@5 探索度的盲盒语料:每题一份,八臂各一块,块序打乱、只给字母。

问题:「@5 的时候我们的模型是否倾向于探索更多做法」。sim_self / self_sim 都是
字符串相似度,答不了这个 —— 两份写法不同的同一个贪心算法相似度很低,两份写法
相近的不同算法相似度很高。所以改成让标注者数「有几种不同做法」。

口径:
  - FCS-research,64 题,八臂每题正好 5 抽(按 (gt, sample_idx) 去重后核过)。
  - 一块 = 一个臂在一道题上的 5 抽的**最终答案**(</think> 之后),不给思维链。
  - **没产出可辨认方法的抽样自成一类**,这样坍塌的臂也进分母,不会被静默剔除。
  - 块序按题打乱,字母与臂的对应只写进 key.json,不进标注者看到的文件。
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
NPROB = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = 全部 64 题

ARMS = ["base9b_v2c", "ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "base4b", "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
BENCH = "frontiercs_research"
SUB = "research_thinking_32k_vllm"
LETTERS = "ABCDEFGH"


def load(arm):
    """dump2 的去重规则:按 (gt, sample_idx),分片排序后后来的覆盖先前的。"""
    ok = {}
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{SUB}/shard_*/samples.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("data_source") != BENCH:
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


def main():
    data = {a: load(a) for a in ARMS}
    probs = sorted({p for a in ARMS for (p, _) in data[a]})
    rng = random.Random(20260917)
    rng.shuffle(probs)                  # 固定种子打乱,再取前 N:子集嵌套
    if NPROB:
        probs = probs[:min(NPROB, len(probs))]
    stm = {}
    try:
        import pandas as pd
        df = pd.read_parquet(f"{FS}/frontiercs/research.parquet")
        for _, r in df.iterrows():
            stm[str(r["reward_model"]["ground_truth"])] = r["prompt"][0]["content"]
    except Exception as e:
        print(f"[warn] 题面读不到:{e!r}")

    os.makedirs(OUT, exist_ok=True)
    kf = f"{OUT}/key.json"
    key = json.load(open(kf)) if os.path.exists(kf) else {}
    for p in probs:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", p)
        if slug in key:
            continue                    # 已经生成过的题不重排,免得作废已有标注
        order = ARMS[:]
        # 每题用自己的种子,子集可嵌套:先跑 3 题再扩到 24 题不会改动前 3 题的块序。
        random.Random(f"20260917|{slug}").shuffle(order)
        lines = [f"# 题目 `{p}`", "",
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
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    print(f"[ok] {len(probs)} 题 × 8 块 -> {OUT}(key.json 不给标注者看)")


if __name__ == "__main__":
    main()
