"""「一条输出之内考虑过几条路」的**盲评语料** —— 把 `explore.py` 的正则指标搬上人工标注。

起因(用户 2026-09-17:「Routes considered per proposal 那些也要标注哈」)。
`explore.py` 的 n_reason / n_abandon / explore_ratio 是靠 `recomb.PAT` 这张**关键词正则表**
数「推理里点到的技术家族」,再和最终代码里点到的家族取差。它量的是**词表命中**,
不是「模型真的认真考虑过一条路」。@5 那张表已经手工标过了,这一块还没有。

所以这份语料让标注者直接读 `</think>` 之前的推理,回答两件事:
  (1) 这条推理里真正认真考虑过的解法路线有哪几条;
  (2) 其中哪几条最后没有出现在最终代码里(考虑过又放弃)。

设计与 @5 的 v3 语料对齐,好配对:
  - **同样的 39 题**(直接取 explore_labels3/ 的 slug)。
  - **同样的 6 条 RL 臂**,同样 A–F 盲块,换一个新种子重排。
  - 但每块只放 **1 条**推理(`sample_idx=0`,固定不挑),不是 5 条 ——
    这个指标本来就是逐样本的,放 5 条会让单题文件到 1MB 以上没法读。
    汇总时正则指标也只在 sample_idx=0 上重算,两边口径才对得上。
  - 只放推理,**不放最终答案**:要判的是「考虑过什么」,给了答案会让人反过来倒推。
    「有没有进最终代码」由标注者从推理本身的收尾判断 —— 推理末尾模型自己会说选了哪条。
"""
import collections, glob, json, os, random, re, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{S}/explore_reason"
LAB3 = f"{S}/explore_labels3"          # 题目集合以 v3 标注为准

ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20"]
LETTERS = "ABCDEF"
FENCE = "`" * 8
SI = 0                                  # 固定第一抽,不挑

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
            if m.get("score", r.get("score")) is None:
                continue
            ok[(str(r["ground_truth"]), int(r.get("sample_idx", -1)))] = r.get("text") or ""
    return ok


def reasoning_of(text):
    i = text.rfind("</think>")
    return (text[:i] if i >= 0 else text).strip()


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
    want = {f[:-5] for f in os.listdir(LAB3) if f.endswith(".json") and not f.startswith("_")}
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
            random.Random(f"reason|20260917|{slug}").shuffle(order)
            lines = [f"# 题目 `{p}`(bench: {bench})", "",
                     "下面是 6 个互相独立的「块」,每块是**一个模型**解这道题时 `</think>` 之前的**推理全文**",
                     "(同一次抽样,未做任何截断)。**最终代码没有给你** —— 要判的是「考虑过什么」,",
                     "看了答案会反过来倒推。推理收尾处模型自己通常会说它选了哪条路。", "",
                     "块与块之间没有任何关系,不要跨块比较,也不要猜哪个块是哪个模型。", ""]
            if stm.get(p):
                lines += ["<details><summary>题面</summary>", "", FENCE,
                          stm[p][:6000], FENCE, "", "</details>", ""]
            key[slug] = {}
            for li, arm in zip(LETTERS, order):
                key[slug][li] = arm
                rs = reasoning_of(data[arm].get((p, SI), ""))
                lines += [f"## 块 {li}", ""]
                lines += ["(这一抽没有推理内容)", ""] if not rs else [FENCE, rs, FENCE, ""]
            open(f"{OUT}/{slug}.md", "w").write("\n".join(lines))
            made[bench] += 1
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    json.dump(kb, open(f"{OUT}/key_bench.json", "w"), indent=1, ensure_ascii=False)
    miss = sorted(want - set(key))
    print(f"[ok] {sum(made.values())} 题 × 6 块(每块 1 条推理,sample_idx={SI}) -> {OUT}  {dict(made)}")
    if miss:
        print(f"[bad] v3 有标注但这份没生成的题:{miss}")


if __name__ == "__main__":
    main()
