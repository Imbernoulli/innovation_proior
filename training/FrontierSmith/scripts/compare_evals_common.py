#!/usr/bin/env python3
"""Compare eval runs on a COMMON problem set, recomputed from samples.jsonl.

Why not just read summary_shard.json: every run summarises over whatever it
managed to score, and those denominators drift. In the agentic-ablation dirs
they were 143 / 178 / 187 / 188 FrontierCS problems for runs sitting in the
same table, and one arm's ALE was never scored at all. Ranking models across
different denominators is not a comparison; the arm that lost its 10 hardest
problems just looks better.

So: rebuild per-problem scores from the raw samples, intersect the problem sets,
require the same number of samples per problem in every run, and only then take
means. The printed n_problems / n_samples are the actual shared denominator.

samples.jsonl is APPEND-ONLY -- a resumed run rewrites rows -- so a row is keyed
by (data_source, problem_idx, sample_idx) and the LAST occurrence wins. Rows
carrying a non-null `error` are dropped: those are infra failures scored 0.0 by
FAIL_SOFT, and counting them as real zeros is what makes a judge outage look
like a model regression.

  usage: compare_evals_common.py <eval_dir> [<eval_dir> ...]
         compare_evals_common.py --selfcheck <eval_dir>     # vs its own summary
"""
import json, glob, sys, os, statistics

FCS_SCORE = ("frontiercs", "score")
ALE_SCORE = ("alebench", "performance")


def load_run(d):
    """-> {data_source: {problem_idx: {sample_idx: value}}}, dropping error rows."""
    last = {}
    for p in sorted(glob.glob(os.path.join(d, "shard_*", "samples.jsonl"))):
        # problem_idx is numbered WITHIN a shard: shard_0's problem 18 and
        # shard_1's problem 18 are different problems. Keying without the shard
        # silently halves the problem set (the selfcheck caught exactly that:
        # 94 recomputed vs 188 published).
        for line in open(p, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            # ground_truth is the only STABLE problem identity across runs:
            # FrontierCS puts the problem number there ('145'), ALE the contest
            # id ('ahc007'). problem_idx is a position within a shard, and the
            # shard split is not the same in every run -- the base dir splits
            # FrontierCS 70/73 where the arms split 94/94, so problem_idx 5 is
            # a different problem in each. Keying on it silently compares
            # unrelated problems and made ALE intersect to nothing.
            k = (r.get("data_source"), r.get("ground_truth"), r.get("sample_idx"))
            if k[0] is None or k[1] is None or k[2] is None:
                continue
            last[k] = r
    out = {}
    for (src, pid, sid), r in last.items():
        if r.get("error"):
            continue
        m = r.get("metrics") or {}
        field = FCS_SCORE[1] if src == FCS_SCORE[0] else (ALE_SCORE[1] if src == ALE_SCORE[0] else None)
        if field is None or m.get(field) is None:
            continue
        out.setdefault(src, {}).setdefault(pid, {})[sid] = float(m[field])
    return out


def per_problem(run, src, problems, k):
    """-> {pid: [k sample values]} using the k lowest sample_idx, for determinism."""
    res = {}
    for pid in problems:
        s = run[src][pid]
        idx = sorted(s)[:k]
        res[pid] = [s[i] for i in idx]
    return res


def summarise(vals, src):
    probs = list(vals.values())
    if not probs:
        return {}
    mean_at_k = statistics.fmean(statistics.fmean(v) for v in probs)
    best_at_k = statistics.fmean(max(v) for v in probs)
    out = {"mean@k": round(mean_at_k, 4), "best@k": round(best_at_k, 4)}
    if src == FCS_SCORE[0]:
        # NOT the leaderboard's pass@1. Calibrating against a full-denominator
        # run, official pass@1 counted 20/940 samples while score>0 counts 32,
        # so the official predicate is stricter than "scored anything" and its
        # exact definition is not recoverable from the sample fields (FCS
        # reward == score == score_unbounded here). mean@k DOES reproduce the
        # published avg_at_5 exactly, so ranking is done on that; these two are
        # labelled for what they are -- how often the model scored at all.
        out["nonzero@1"] = round(statistics.fmean(statistics.fmean(1.0 if x > 0 else 0.0 for x in v) for v in probs), 4)
        out["nonzero@k"] = round(statistics.fmean(1.0 if max(v) > 0 else 0.0 for v in probs), 4)
    return out


def main(argv):
    selfcheck = False
    if argv and argv[0] == "--selfcheck":
        selfcheck = True; argv = argv[1:]
    dirs = argv
    if not dirs:
        print(__doc__); return 1
    runs = {d: load_run(d) for d in dirs}
    for src in (FCS_SCORE[0], ALE_SCORE[0]):
        present = [d for d in dirs if src in runs[d]]
        if not present:
            continue
        # a problem counts only where it has samples in EVERY run being compared
        common = None
        for d in present:
            s = set(runs[d][src])
            common = s if common is None else (common & s)
        if not common:
            print(f"\n=== {src}: no common problems across {len(present)} runs ==="); continue
        # Fixed k (the protocol's n=5), not min-over-runs: letting k collapse
        # to whatever the thinnest problem has turns avg@5 into score@1, a far
        # noisier metric, without saying so. Problems that lack k samples in
        # any run are dropped instead.
        k = int(os.environ.get("EVAL_K", "5"))
        thin = {pid for pid in common if any(len(runs[d][src][pid]) < k for d in present)}
        if thin:
            common = common - thin
        if not common:
            print(f"\n=== {src}: no problem has {k} samples in all runs ==="); continue
        common = sorted(common)
        print(f"\n=== {src}: {len(common)} common problems x k={k} samples "
              f"({len(present)}/{len(dirs)} runs have this source"
              + (f"; {len(thin)} dropped for <{k} samples" if thin else "") + ") ===")
        rows = []
        for d in present:
            rows.append((os.path.basename(d), summarise(per_problem(runs[d], src, common, k), src),
                         len(runs[d][src]), len(common)))
        keys = list(rows[0][1].keys())
        w = max(len(r[0]) for r in rows)
        print(f"  {'run'.ljust(w)}  " + "  ".join(kk.rjust(9) for kk in keys) + "   own_problems")
        for name, s, own, nc in rows:
            print(f"  {name.ljust(w)}  " + "  ".join(f"{s[kk]:9.4f}" for kk in keys)
                  + f"   {own}" + ("" if own == nc else f"  (dropped {own-nc} not shared)"))
    if selfcheck:
        d = dirs[0]
        print("\n=== selfcheck vs published summary_shard.json ===")
        tot = {}
        for p in sorted(glob.glob(os.path.join(d, "shard_*", "summary_shard.json"))):
            s = json.load(open(p))
            lb = (s.get("official_leaderboard_metrics") or {}).get("frontiercs") or {}
            n = lb.get("num_problems") or 0
            if n:
                tot["n"] = tot.get("n", 0) + n
                tot["avg"] = tot.get("avg", 0.0) + (lb.get("avg_at_5") or 0) * n
                tot["p1"] = tot.get("p1", 0.0) + (lb.get("pass_at_1") or 0) * n
        if tot.get("n"):
            print(f"  published: fcs n={tot['n']} avg@5={tot['avg']/tot['n']:.4f} pass@1={tot['p1']/tot['n']:.4f}")
        r = runs[d]
        if FCS_SCORE[0] in r:
            allp = sorted(r[FCS_SCORE[0]])
            kk = min(len(r[FCS_SCORE[0]][pid]) for pid in allp)
            s = summarise(per_problem(r, FCS_SCORE[0], allp, kk), FCS_SCORE[0])
            print(f"  recomputed: fcs n={len(allp)} k={kk} mean@k={s['mean@k']:.4f} (must equal published avg@5)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
