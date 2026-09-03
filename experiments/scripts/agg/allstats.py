"""allstats.py -- per-problem @5 statistics for every arm on one identical problem set.

Why this exists, and why it is NOT paircommon.py:
  The sampling protocol is N=5 (EVAL_PLAN.md: temp 1.0, top_p 0.95, top_k 20,
  presence 1.5, N=5). paircommon.py treats each (problem, sample_idx) pair as an
  independent unit and averages over all 860/200/320 of them. That answers "what
  does one sample from this model score", which is mean@5 -- but it cannot answer
  best@5 / worst@5 / pass@5, and its bootstrap resamples SAMPLES, which understates
  the CI because the 5 samples of one problem are correlated.

  This tool aggregates to the PROBLEM first (172 FCS / 40 ALE / 10 ALE-lite /
  64 research), then bootstraps over PROBLEMS. That is the correct unit.

Equal denominator, strictly:
  A problem counts for an arm only if ALL 5 of its samples are usable (no `error`,
  numeric score). The reported set is the intersection across every arm AND the
  anchor, so every number in a row comes off the same problems. COVERAGE prints
  what each arm lost so the cost is never silent.

ALE-10 is free:
  ALE-Bench Lite is 10 problems (ahc008 ahc011 ahc015 ahc016 ahc024 ahc025 ahc026
  ahc027 ahc039 ahc046) and every one of them is already inside the 40 we ran, so
  `alebench_lite` is a filter over existing samples -- no new compute.

  python3 allstats.py <bench> <tag1> <tag2> ...        # human table
  python3 allstats.py <bench> <tags...> --json out.json
"""
import json, glob, sys, random, statistics, os

ROOT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"

ALE_LITE = {"ahc008", "ahc011", "ahc015", "ahc016", "ahc024",
            "ahc025", "ahc026", "ahc027", "ahc039", "ahc046"}

BENCH = {
    # name             data_source            output subdir                     filter
    "frontiercs":          ("frontiercs",          "thinking_32k_both_vllm",     None),
    "alebench":            ("alebench",            "thinking_32k_both_vllm",     None),
    "alebench_lite":       ("alebench",            "thinking_32k_both_vllm",     ALE_LITE),
    "frontiercs_research": ("frontiercs_research", "research_thinking_32k_vllm", None),
}

N_EXPECTED = 5


def load(tag, bench):
    """-> {problem: {sample_idx: score}} keeping only rows with a numeric score."""
    src, sub, keep = BENCH[bench]
    out = {}
    for f in glob.glob(f"{ROOT}/cc_eval_{tag}_{sub}/shard_*/samples.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("data_source") != src:
                continue
            p = str(r["ground_truth"])
            if keep is not None and p not in keep:
                continue
            if r.get("error"):
                continue
            s = (r.get("metrics") or {}).get("score", r.get("score"))
            if s is None:
                continue
            # a resumed run can rewrite a sample; last writer wins, same as the harness
            out.setdefault(p, {})[r.get("sample_idx")] = float(s)
    return out


def complete(d):
    """problems whose 5 samples are ALL present."""
    return {p for p, v in d.items() if len(v) >= N_EXPECTED}


def stats(d, problems):
    """per-problem @5 aggregates over a fixed problem set."""
    mean, best, worst, anyhit, allhit = [], [], [], [], []
    for p in problems:
        v = [d[p][i] for i in sorted(d[p])[:N_EXPECTED]]
        mean.append(sum(v) / len(v))
        best.append(max(v))
        worst.append(min(v))
        anyhit.append(1.0 if any(x > 0 for x in v) else 0.0)
        allhit.append(1.0 if all(x > 0 for x in v) else 0.0)
    return {"mean@5": mean, "best@5": best, "worst@5": worst,
            "pass@5": anyhit, "pass@1_all": allhit}


def boot_ci(vals, n=5000, seed=0):
    """bootstrap over PROBLEMS (the independent unit), not samples."""
    if not vals:
        return (float("nan"), float("nan"))
    random.seed(seed)
    N = len(vals)
    out = [sum(vals[random.randrange(N)] for _ in range(N)) / N for _ in range(n)]
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def boot_paired_ci(a, b, n=5000, seed=0):
    """paired bootstrap of (a-b) over problems; returns (diff, lo, hi, P(diff>0))."""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return (float("nan"),) * 3 + (float("nan"),)
    random.seed(seed)
    N = len(d)
    out = [sum(d[random.randrange(N)] for _ in range(N)) / N for _ in range(n)]
    out.sort()
    return (sum(d) / N, out[int(0.025 * n)], out[int(0.975 * n)],
            sum(1 for x in out if x > 0) / n)


def main():
    argv = [a for a in sys.argv[1:]]
    jsonout = None
    if "--json" in argv:
        i = argv.index("--json")
        jsonout = argv[i + 1]
        del argv[i:i + 2]
    bench, tags = argv[0], argv[1:]
    if bench not in BENCH:
        sys.exit(f"bench must be one of {sorted(BENCH)}")

    data = {t: load(t, bench) for t in tags}
    have = {t: complete(data[t]) for t in tags}
    present = [t for t in tags if have[t]]
    if not present:
        sys.exit("no arm has any complete problem")
    common = sorted(set.intersection(*(have[t] for t in present)))

    print(f"=== {bench} | problems complete in every arm: {len(common)} "
          f"| {len(present)}/{len(tags)} arms have data ===")
    print("COVERAGE (problems with all 5 samples / kept in common):")
    for t in tags:
        miss = "  NO DATA" if not have[t] else ""
        print(f"  {t:30s} {len(have[t]):4d} / {len(common):4d}{miss}")
    if not common:
        sys.exit("empty common set")

    res = {}
    for t in present:
        s = stats(data[t], common)
        res[t] = {k: (sum(v) / len(v), *boot_ci(v)) for k, v in s.items()}
        res[t]["_raw"] = s

    print(f"\n{'arm':30s} {'mean@5':>10s} {'[95% CI]':>20s} {'best@5':>10s} "
          f"{'worst@5':>10s} {'pass@5':>8s} {'all5>0':>8s}")
    order = sorted(present, key=lambda t: -res[t]["mean@5"][0])
    for t in order:
        m, lo, hi = res[t]["mean@5"]
        b = res[t]["best@5"][0]
        w = res[t]["worst@5"][0]
        p = res[t]["pass@5"][0]
        a = res[t]["pass@1_all"][0]
        print(f"  {t:28s} {m:10.4f} [{lo:8.3f},{hi:8.3f}] {b:10.4f} "
              f"{w:10.4f} {p:8.1%} {a:8.1%}")

    if jsonout:
        dump = {"bench": bench, "n_problems": len(common), "problems": common,
                "arms": {t: {k: v for k, v in res[t].items() if k != "_raw"}
                         for t in present},
                "per_problem": {t: {k: dict(zip(common, v))
                                    for k, v in res[t]["_raw"].items()}
                                for t in present}}
        json.dump(dump, open(jsonout, "w"), indent=1)
        print(f"\nwrote {jsonout}")


if __name__ == "__main__":
    main()
