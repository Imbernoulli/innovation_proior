"""@5 做法聚类的**第二版语料**:不截断,只放 RL 之后的四条臂。

为什么要重做:v1 语料把每条最终答案截到 4000 字符。按标注者实际看到的比例分层:

  看到完整答案      n=447,被判「没产出方法」 1.8%
  只看到 50-100%   n= 46,                8.7%
  只看到 25-50%    n= 11,               27.3%
  **只看到 <25%**   n= 31,              **58.1%**

也就是说 4000 字这道闸门自己在**制造**「没产出方法」。而各臂被它砸中的程度不一样
(ALE 上我们 9B 有 7 个「判无方法」的格子可见比例 <50%,对手只有 1 个),所以
v1 的 n_method 差值里混着一截纯粹的语料假象。

v2 的改法:
  - **一个字都不截**(全量 2.6MB / 39 题,最大一题 463KB,标注者分几次读得完)。
  - 只放 RL 之后的四条臂 —— 用户裁决主表只看 RL 之后,四块也让单题文件小一半。
  - 题目集合与 v1 **完全相同**(直接取 v1 已标注的 39 个 slug),这样 v1/v2 可配对。
  - 块序换一个新种子重排:v2 的 A 不是 v1 的 A,免得沿用上一版的印象。
"""
import collections, glob, json, os, random, re, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{S}/explore_blind2"
LAB1 = f"{S}/explore_labels"          # v1 标注:题目集合以它为准

ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
LETTERS = "ABCD"
FENCE = "`" * 8                       # 正文里本来就有 ```cpp,三反引号会把文件劈坏

# (bench, 产出目录后缀, 题面 parquet, slug 前缀)
BENCHES = [("frontiercs_research", "research_thinking_32k_vllm", "frontiercs/research.parquet", ""),
           ("frontiercs",          "thinking_32k_both_vllm",     "frontiercs/full.parquet",     "fcs_"),
           ("alebench",            "thinking_32k_both_vllm",     "alebench/full40.parquet",     "ale_")]


def load(arm, bench, sub):
    ok = {}
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub}/shard_*/samples.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("data_source") != bench or r.get("error"):
                continue
            m = r.get("metrics") or {}
            s = m.get("score", r.get("score"))
            if s is None:
                continue
            ok[(str(r["ground_truth"]), int(r.get("sample_idx", -1)))] = r.get("text") or ""
    return ok


def final_of(text):
    i = text.rfind("</think>")
    return text[i + 8:].strip() if i >= 0 else ""


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
    want = {f[:-5] for f in os.listdir(LAB1) if f.endswith(".json")}
    os.makedirs(OUT, exist_ok=True)
    key, kb, made = {}, {}, collections.Counter()
    for bench, sub, pq, pre in BENCHES:
        data = {a: load(a, bench, sub) for a in ARMS}
        probs = sorted({p for a in ARMS for (p, _) in data[a]})
        stm = statements(pq)
        for p in probs:
            slug = pre + re.sub(r"[^A-Za-z0-9_.-]+", "_", p)
            if slug not in want:
                continue
            kb[slug] = bench
            order = ARMS[:]
            random.Random(f"v2|20260917|{slug}").shuffle(order)
            lines = [f"# 题目 `{p}`(bench: {bench})", "",
                     "下面是 4 个互相独立的「块」,每块是同一个模型对这道题的 5 次独立抽样的**最终答案**。",
                     "块与块之间没有任何关系,不要跨块比较。**答案未做任何截断,请完整读完再判断。**", ""]
            if stm.get(p):
                lines += ["<details><summary>题面</summary>", "", FENCE,
                          stm[p][:6000], FENCE, "", "</details>", ""]
            key[slug] = {}
            for li, arm in zip(LETTERS, order):
                key[slug][li] = arm
                lines += [f"## 块 {li}", ""]
                for si in range(5):
                    fin = final_of(data[arm].get((p, si), ""))
                    lines += [f"### 块 {li} · 抽样 {si + 1}", ""]
                    lines += ["(没有最终答案)", ""] if not fin else [FENCE, fin, FENCE, ""]
            open(f"{OUT}/{slug}.md", "w").write("\n".join(lines))
            made[bench] += 1
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    json.dump(kb, open(f"{OUT}/key_bench.json", "w"), indent=1, ensure_ascii=False)
    miss = sorted(want - set(key))
    print(f"[ok] {sum(made.values())} 题 × 4 块 -> {OUT}  {dict(made)}")
    if miss:
        print(f"[bad] v1 有标注但 v2 没生成的题:{miss}")


if __name__ == "__main__":
    main()
