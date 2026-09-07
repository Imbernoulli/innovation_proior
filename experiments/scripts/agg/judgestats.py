"""judgestats.py -- turn a judge_pairwise.py output dir into a reportable number.

WHAT IT PRINTS, AND WHY EACH COLUMN IS THERE
  win        item-level win rate. The 4 draws of one item are collapsed to one
             number first, then the bootstrap resamples ITEMS. Resampling draws
             would treat 4 correlated judgements of the same idea as 4 independent
             observations and overstate the interval.
  CI / P     10000 item-level bootstrap replicates. P is the fraction above 50% --
             NOT a p-value, and it says nothing about training-seed variance, since
             every arm was trained once. See EVAL_REPORT_v2_9B_zh.md §2.2b.
  agree      fraction of pairs where the two orders (candidates swapped) named the
             same candidate. This is the honest measure of how much signal the judge
             has: at 50% the judge is a coin and its win rate is meaningless. Below
             ~55-60% do not report the win rate at all.
  unparsed   fraction where the judge never emitted a VERDICT. Anything above a few
             percent means the judge's max_tokens is binding and the comparison is
             measuring truncation, not quality -- that failure has already happened
             once in this campaign (report §12.1). Check this column FIRST.
  chars      mean characters on each side. Length is the most common confound; having
             it printed is what let §11.2 show the RL-stage loss happened while our
             side was 19-41% longer.

  python3 judgestats.py <task> <judge_out_dir> [more dirs...]
"""
import json, os, random, sys
from collections import defaultdict


def stats(task, d):
    rows = []
    p = os.path.join(d, "judgements.jsonl")
    if not os.path.exists(p):
        return None
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("task") == task:
            rows.append(r)
    if not rows:
        return None

    # a win counts only when BOTH orders agree; everything else is a tie
    byitem = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["outcome"] == "a_wins":
            byitem[r["id"]][0] += 1
        elif r["outcome"] == "b_wins":
            byitem[r["id"]][1] += 1
    xs = [w / (w + l) for w, l in byitem.values() if w + l]
    if not xs:
        return None

    n = len(xs)
    rng = random.Random(20260907)
    bs = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    decided = sum(1 for r in rows if r["outcome"] in ("a_wins", "b_wins"))
    return {
        "rows": len(rows), "items": n,
        "win": sum(xs) / n, "lo": bs[250], "hi": bs[9750],
        "p_gt_half": sum(1 for x in bs if x > 0.5) / len(bs),
        "agree": decided / len(rows),
        "unparsed": sum(1 for r in rows if r["outcome"] == "unparsed") / len(rows),
        "a_chars": sum(r["a_chars"] for r in rows) / len(rows),
        "b_chars": sum(r["b_chars"] for r in rows) / len(rows),
    }


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    task, dirs = sys.argv[1], sys.argv[2:]
    for d in dirs:
        s = stats(task, d)
        print(os.path.basename(d.rstrip("/")))
        if not s:
            print(f"  no decided rows for {task}")
            continue
        print(f"  rows={s['rows']} items={s['items']} "
              f"win={s['win']:.1%} CI[{s['lo']:.1%},{s['hi']:.1%}] "
              f"P(>50%)={s['p_gt_half']:.3f} agree={s['agree']:.1%} "
              f"unparsed={s['unparsed']:.1%} chars={s['a_chars']:.0f}/{s['b_chars']:.0f}")


if __name__ == "__main__":
    main()
