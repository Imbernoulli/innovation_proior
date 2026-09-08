"""v3 judgement bench with the human-reviewer oracle as a row.

The oracle is not a model: for every pair it simply picks whichever paper the real
conference reviewers scored higher (`meta.reviewer_edge`). That is what a competent
human referee, reading the same two papers at submission time, actually concluded.
"""
import json, glob, os, collections, statistics

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]
EXT = [t.split("cc_judge3_")[-1] for t in
       sorted(glob.glob(f"{D}/outputs/cc_judge3_ext_*")) if os.path.isdir(t)]
TASKS = ["impact_pair", "impact_contrarian", "novelty_pair"]
N = 5


def arm_acc(tag):
    """penalise lens: an unparsed draw counts as wrong, nothing is dropped."""
    f = f"{D}/outputs/cc_judge3_{tag}/samples.jsonl"
    if not os.path.exists(f):
        return None
    d = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        d[(r["task"], r["id"])].append(bool(r.get("correct")))
    out = {}
    for t in TASKS:
        per = [sum(v[:N]) / N for (tt, _), v in d.items() if tt == t and len(v) >= N]
        out[t] = 100.0 * statistics.mean(per) if per else float("nan")
    return out


def main():
    rows = [json.loads(l) for l in open(f"{D}/ideabench/tasks_v3.jsonl")]
    pairs = {}
    for r in rows:
        pairs.setdefault((r["task"], r["meta"]["pair_id"]), r["meta"])
    orc = {}
    for t in ["impact_pair", "impact_contrarian"]:
        ms = [m for (tt, _), m in pairs.items() if tt == t]
        orc[t] = 100.0 * sum(1 for m in ms if m["reviewer_edge"] > 0) / len(ms)
    n_ip = sum(1 for (t, _) in pairs if t == "impact_pair")
    n_ic = sum(1 for (t, _) in pairs if t == "impact_contrarian")
    comb = (orc["impact_pair"] * n_ip + orc["impact_contrarian"] * n_ic) / (n_ip + n_ic)

    print("| 参照 / 模型 | `impact_pair` | `impact_contrarian` | 两者合并(500 对) | `novelty_pair` |")
    print("|---|---|---|---|---|")
    print(f"| **真人评审(照抄评审分)** | **{orc['impact_pair']:.2f}** | **{orc['impact_contrarian']:.2f}** "
          f"| **{comb:.2f}** | 无评审信号,不适用 |")
    print("| 随机猜 | 50.00 | 50.00 | 50.00 | 50.00 |")
    for tag in ARMS + EXT:
        a = arm_acc(tag)
        if a is None:
            continue
        c = (a["impact_pair"] + a["impact_contrarian"]) / 2
        label = f"`{tag}`" + (" *(外部模型)*" if tag.startswith("ext_") else "")
        print(f"| {label} | {a['impact_pair']:.2f} | {a['impact_contrarian']:.2f} | {c:.2f} "
              f"| {a['novelty_pair']:.2f} |")


main()
