"""j3stats.py -- the research-judgement (v3) bench, aggregated the way it was pre-registered.

WHY THIS FILE EXISTS SEPARATELY FROM ideastats.py
  The v3 bench differs from every earlier idea bench in two ways that the older
  aggregator gets wrong if you point it here:

  1. Every item exists TWICE, once per presentation order (`ip0o1` / `ip0o2`), with
     the gold answer flipped. The independent unit is therefore the PAIR, not the
     item -- treating the two orders as two observations would halve the interval
     of a quantity that is really one measurement plus its mirror. This script
     collapses orders to `meta.pair_id` first and bootstraps over pairs.

  2. The RL arms truncate. On v3 the RL-stage control emits no VERDICT on ~5% of
     samples where the SFT arms lose <=0.06% (report §14.3: degenerate repetition,
     not a token budget -- the median completion is ~4-5k against a 16k cap). An
     unparsed sample is not missing at random: it is concentrated on the arm that
     degenerated, and (measured with the four SFT arms) on items that are 9.5-14.4pp
     harder for everyone. So there is no single honest denominator, and the two
     defensible conventions were WRITTEN DOWN BEFORE THE NUMBERS WERE LOOKED AT:

       --lens strict   an item counts only if EVERY arm parsed all 5 samples, and a
                       pair counts only if both of its orders survive. Unbiased on
                       the surviving pairs, but the surviving set is selected by the
                       control arm's failure mode -- it discards the hard items the
                       control could not finish, which flatters the control.
       --lens penalise unparsed == wrong, nothing dropped, all 250/250 pairs. Charges
                       degeneration to the arm that degenerated. Conflates "judged
                       badly" with "never answered".

     Report BOTH. Where they disagree, that disagreement IS the finding.

WHAT EACH BLOCK PRINTS
  ACC        per-task pair-level accuracy, and the paired contrast against --control.
             The bootstrap resamples PAIRS (10000 reps, fixed seed) and is paired --
             the same resampled pairs are used for both arms, so the shared
             item-difficulty variance cancels.
  MIMICRY    acc(impact_pair) - acc(impact_contrarian). impact_pair is the half where
             the higher-impact paper also got the better review score; impact_contrarian
             is the half where the reviewers backed the wrong paper. A model that has
             learned to imitate reviewers scores high on the first and 0 on the second
             (the reviewer-copy oracle is exactly 100%/0%). A large positive gap is
             mimicry; a small one means the judgement is about the work. The two tasks
             use DISJOINT pair sets, so this contrast is not paired: each task is
             resampled independently inside the same replicate.
  ORDER      how often the two orders of a pair name the same paper. This is the
             internal-consistency check: a model that is really reading the papers
             should give the same answer when they are swapped. Reported as
             majority-agreement (does the 5-sample majority pick the same paper) and
             as mean |acc_o1 - acc_o2|.
  RECOGNISE  the "does it just know the famous paper" check. lc_gap is held inside a
             narrow band (default 1.0-1.7) so the two halves are equally separated in
             impact, then pairs are split at the median `hi_cites` -- i.e. on whether
             the WINNER is famous or obscure. Year stratification does NOT substitute:
             fame and gap are both confounded with year, holding year holds neither.

  python3 j3stats.py --lens strict   --control rlv5_base_s20 <tag> <tag> ...
  python3 j3stats.py --lens penalise --control base9b_v2c    <tag> <tag> ...
"""
import argparse, json, os, random, statistics, sys
from collections import defaultdict

ROOT = os.environ.get("INNOV_OUTPUTS",
                      "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs")
PREFIX = os.environ.get("J3_PREFIX", "cc_judge3_")
TASKS = ["impact_pair", "impact_contrarian", "novelty_pair"]
N_EXPECTED = 5
N_BOOT = 10000
SEED = 20260908


def load(tag):
    """-> {item_id: {"task","pair_id","order","meta", "ok":[bool]*n, "parsed":[bool]*n}}"""
    f = os.path.join(ROOT, f"{PREFIX}{tag}", "samples.jsonl")
    out = {}
    seen = {}
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        iid = r["id"]
        # a resumed run can rewrite a sample; last writer wins, same as the harness
        key = (iid, str(r.get("sample_idx")))
        seen[key] = r
    for (iid, _), r in seen.items():
        m = r["meta"]
        d = out.setdefault(iid, {"task": r["task"], "pair_id": m["pair_id"],
                                 "order": m["order"], "meta": m,
                                 "ok": [], "parsed": []})
        unp = r["unparsed"] in (True, "True", "true", 1)
        d["parsed"].append(not unp)
        d["ok"].append((not unp) and (r["correct"] in (True, "True", "true", 1)))
    return out


def item_acc(d, lens):
    """5-sample accuracy of one item under a lens, or None if the item is dropped."""
    n = len(d["ok"])
    if n < N_EXPECTED:
        return None
    if lens == "strict":
        if not all(d["parsed"][:N_EXPECTED]):
            return None
        return sum(d["ok"][:N_EXPECTED]) / N_EXPECTED
    return sum(d["ok"][:N_EXPECTED]) / N_EXPECTED   # penalise: unparsed already False


def pair_table(arms, lens):
    """-> {task: {pair_id: {arm: (acc, acc_o1, acc_o2)}}} over pairs kept for ALL arms."""
    per = {t: defaultdict(dict) for t in TASKS}
    meta = {}
    common = None
    for tag, data in arms.items():
        byp = defaultdict(dict)
        for iid, d in data.items():
            a = item_acc(d, lens)
            if a is None:
                continue
            byp[d["pair_id"]][d["order"]] = a
            meta[d["pair_id"]] = d["meta"] | {"task": d["task"]}
        keep = {p for p, o in byp.items() if 1 in o and 2 in o}
        common = keep if common is None else (common & keep)
        for p in keep:
            per[meta[p]["task"]][p][tag] = ((byp[p][1] + byp[p][2]) / 2,
                                            byp[p][1], byp[p][2])
    for t in TASKS:
        for p in list(per[t]):
            if p not in common:
                del per[t][p]
    return per, meta


def boot_diff(a, b, rng):
    """paired bootstrap of mean(a-b) -> (diff, lo, hi, P(diff>0))"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if not n:
        return (float("nan"),) * 4
    reps = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return (sum(d) / n, reps[int(0.025 * N_BOOT)], reps[int(0.975 * N_BOOT)],
            sum(1 for x in reps if x > 0) / N_BOOT)


def boot_gap(pa, pb, ca, cb, rng):
    """(acc_pair - acc_contrarian) for arm minus the same for control.

    The two tasks are disjoint pair sets, so they are resampled independently inside
    each replicate; arm and control ARE paired within each task.
    """
    na, nb = len(pa), len(pb)
    obs = (sum(x - y for x, y in zip(pa, ca)) / na
           - sum(x - y for x, y in zip(pb, cb)) / nb)
    reps = []
    for _ in range(N_BOOT):
        ia = [rng.randrange(na) for _ in range(na)]
        ib = [rng.randrange(nb) for _ in range(nb)]
        reps.append(sum(pa[i] - ca[i] for i in ia) / na
                    - sum(pb[i] - cb[i] for i in ib) / nb)
    reps.sort()
    return (obs, reps[int(0.025 * N_BOOT)], reps[int(0.975 * N_BOOT)],
            sum(1 for x in reps if x > 0) / N_BOOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--lens", choices=["strict", "penalise"], required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--gap-lo", type=float, default=1.0)
    ap.add_argument("--gap-hi", type=float, default=1.7)
    ap.add_argument("--json")
    a = ap.parse_args()
    tags = a.tags
    if a.control not in tags:
        sys.exit(f"--control {a.control} must be among the tags")

    arms = {t: load(t) for t in tags}
    per, meta = pair_table(arms, a.lens)

    print(f"=== v3 judgement | lens={a.lens} | control={a.control} "
          f"| {N_BOOT} paired bootstrap reps ===")
    print("TRUNCATION (share of samples with no VERDICT, all tasks pooled):")
    for t in tags:
        tot = sum(len(d["parsed"]) for d in arms[t].values())
        unp = sum(len(d["parsed"]) - sum(d["parsed"]) for d in arms[t].values())
        print(f"  {t:28s} {unp}/{tot} = {unp/tot:.2%}")
    print("PAIRS KEPT (identical set for every arm):")
    for t in TASKS:
        print(f"  {t:20s} {len(per[t])}")

    out = {"lens": a.lens, "control": a.control, "arms": {}}
    # ---- ACC -------------------------------------------------------------------
    for t in TASKS:
        pairs = sorted(per[t])
        if not pairs:
            continue
        print(f"\n[ACC] {t}  (n={len(pairs)} pairs)")
        ctrl = [per[t][p][a.control][0] for p in pairs]
        for tag in tags:
            v = [per[t][p][tag][0] for p in pairs]
            line = f"  {tag:28s} {sum(v)/len(v):7.2%}"
            if tag != a.control:
                rng = random.Random(SEED)
                d, lo, hi, P = boot_diff(v, ctrl, rng)
                line += (f"   vs control {d:+.2%} CI[{lo:+.2%},{hi:+.2%}] P(>0)={P:.3f}")
                out["arms"].setdefault(tag, {})[f"acc_{t}"] = [sum(v)/len(v), d, lo, hi, P]
            else:
                out["arms"].setdefault(tag, {})[f"acc_{t}"] = [sum(v)/len(v)]
            print(line)

    # ---- MIMICRY ---------------------------------------------------------------
    pa_ids = sorted(per["impact_pair"])
    pb_ids = sorted(per["impact_contrarian"])
    if pa_ids and pb_ids:
        print(f"\n[MIMICRY] acc(impact_pair) - acc(impact_contrarian)"
              f"   (n={len(pa_ids)}/{len(pb_ids)} pairs)")
        ca = [per["impact_pair"][p][a.control][0] for p in pa_ids]
        cb = [per["impact_contrarian"][p][a.control][0] for p in pb_ids]
        for tag in tags:
            va = [per["impact_pair"][p][tag][0] for p in pa_ids]
            vb = [per["impact_contrarian"][p][tag][0] for p in pb_ids]
            g = sum(va)/len(va) - sum(vb)/len(vb)
            line = f"  {tag:28s} gap={g:+7.2%}"
            if tag != a.control:
                rng = random.Random(SEED + 1)
                d, lo, hi, P = boot_gap(va, vb, ca, cb, rng)
                line += f"   excess {d:+.2%} CI[{lo:+.2%},{hi:+.2%}] P(>0)={P:.3f}"
                out["arms"][tag]["mimicry"] = [g, d, lo, hi, P]
            else:
                out["arms"][tag]["mimicry"] = [g]
            print(line)

    # ---- ORDER -----------------------------------------------------------------
    print("\n[ORDER] two orders of the same pair name the same paper")
    for t in TASKS:
        pairs = sorted(per[t])
        if not pairs:
            continue
        print(f"  {t}")
        for tag in tags:
            agree, spread = [], []
            for p in pairs:
                _, o1, o2 = per[t][p][tag]
                agree.append(1.0 if (o1 > 0.5) == (o2 > 0.5) else 0.0)
                spread.append(abs(o1 - o2))
            print(f"    {tag:26s} majority-agree={sum(agree)/len(agree):6.1%} "
                  f"mean|acc1-acc2|={sum(spread)/len(spread):.3f}")
            out["arms"][tag][f"order_{t}"] = [sum(agree)/len(agree),
                                              sum(spread)/len(spread)]

    # ---- RECOGNISE -------------------------------------------------------------
    print(f"\n[RECOGNISE] lc_gap held in [{a.gap_lo},{a.gap_hi}], split at median hi_cites")
    for t in ("impact_pair", "impact_contrarian"):
        band = [p for p in sorted(per[t])
                if a.gap_lo <= float(meta[p]["lc_gap"]) <= a.gap_hi]
        if len(band) < 20:
            print(f"  {t}: only {len(band)} pairs in band -- skipped")
            continue
        med = statistics.median(float(meta[p]["hi_cites"]) for p in band)
        famous = [p for p in band if float(meta[p]["hi_cites"]) > med]
        obscure = [p for p in band if float(meta[p]["hi_cites"]) <= med]
        gm = statistics.mean(float(meta[p]["lc_gap"]) for p in famous)
        go = statistics.mean(float(meta[p]["lc_gap"]) for p in obscure)
        print(f"  {t}  n={len(band)} (famous {len(famous)} / obscure {len(obscure)}), "
              f"mean lc_gap {gm:.2f} vs {go:.2f}")
        for tag in tags:
            f = sum(per[t][p][tag][0] for p in famous) / len(famous)
            o = sum(per[t][p][tag][0] for p in obscure) / len(obscure)
            print(f"    {tag:26s} famous={f:6.1%} obscure={o:6.1%} "
                  f"delta={(f-o)*100:+.1f}pp")
            out["arms"][tag][f"recog_{t}"] = [f, o, f - o]

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
