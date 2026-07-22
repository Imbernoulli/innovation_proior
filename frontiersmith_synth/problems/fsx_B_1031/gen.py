#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE apical-dominance / heterochrony instance to stdout.

A shoot axis differentiates H lateral-bud positions, one per developmental tick
i = 1..H (bud i differentiates exactly at tick i).  Each bud i carries a
COMMITMENT COST c_i (a positive integer): once past its own founding instant, a
bud needs c_i cumulative auxin-released ("low") ticks before it commits to
becoming an active lateral branch.

The trap ladder (testId 1..10, H growing 14..48):
  * roughly the first 45% of positions ("early zone") are DENSELY seeded with
    buds that must stay arrested forever (95% of early positions) -- protecting
    all of them via a single contiguous suppression prefix is *possible* (a
    single switch time is a fine recipe there) but blocks the few early targets
    interspersed among them;
  * the remaining "late zone" is still majority-arrested (70%) but the
    arrested buds are scattered THROUGHOUT the rest of the whole developmental
    window, not concentrated where any single switch-time could reach them.
  A one-parameter "flip the dominance switch once at time W" strategy (the
  natural first idea from the seed's own trap: a scalar/constant strength) can
  only ever protect a prefix -- so it is structurally blind to arrested buds
  born after its switch, while sacrificing real targets born before it.  Only a
  genuinely time-varying schedule -- suppressing exactly the founding instant of
  each unwanted bud, nowhere else -- reaches full precision (no leaked
  arrested bud) at minimal cost; note that vetoing every non-target founding
  tick still spends shared release ticks that some late, high-cost targets
  needed to accumulate their own commitment, so even this schedule can miss a
  handful of targets -- recall is not automatically perfect, it is a further
  budget/order trade-off left for the solver to improve on.

Every target is chosen to be REACHABLE: a target bud i is only ever assigned
a commitment cost c_i <= (H - i + 1), the number of ticks remaining in its
own life (including its own founding tick), so an all-release schedule from
i onward always has enough ticks to reach c_i. The last position i = H can
never satisfy c_i >= 2 within its 1-tick remaining life, so it is never
selected as a target.

STDOUT format:
    line 1:  H BUDGET K
    line 2:  c_1 c_2 ... c_H
    line 3:  t_1 t_2 ... t_K      (K target bud IDs, required commitment order)
"""
import sys, random

H_LADDER = [14, 16, 18, 22, 24, 28, 32, 36, 42, 48]

EARLY_FORBID_P = 0.95
LATE_FORBID_P = 0.70
EARLY_FRAC = 0.45
CMAX_FRAC = 0.15
MIN_TARGET_FRAC = 0.28


def build(t):
    rng = random.Random(90031 + t * 7919)
    H = H_LADDER[(t - 1) % len(H_LADDER)]
    Cmax = max(3, int(H * CMAX_FRAC))
    early_cut = max(1, int(H * EARLY_FRAC))

    def life(i):
        return H - i + 1

    def draw_cost(i, target):
        # A target's cost must fit inside its own remaining life so an
        # all-release schedule from i onward is always enough to reach it.
        hi = min(Cmax, life(i)) if target else Cmax
        hi = max(2, hi)
        return rng.randint(2, hi)

    c = [0] * (H + 1)
    target_nodes = set()
    # position H has only 1 tick of remaining life (itself) and c_i >= 2
    # always, so it can never commit -- exclude it from target eligibility.
    for i in range(1, H + 1):
        eligible = i < H and life(i) >= 2
        p_forbid = EARLY_FORBID_P if i <= early_cut else LATE_FORBID_P
        is_target = eligible and rng.random() >= p_forbid
        if is_target:
            target_nodes.add(i)
        c[i] = draw_cost(i, is_target)

    min_targets = max(5, round(H * MIN_TARGET_FRAC))
    if len(target_nodes) < min_targets:
        pool = [i for i in range(1, H) if i not in target_nodes]  # excludes H
        rng.shuffle(pool)
        need = min_targets - len(target_nodes)
        for i in pool[:need]:
            target_nodes.add(i)
            c[i] = draw_cost(i, True)  # re-draw so it stays reachable

    T = sorted(target_nodes)
    nforbidden = H - len(T)
    BUDGET = nforbidden
    return H, BUDGET, c[1:], T


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H, BUDGET, c, T = build(t)
    out = []
    out.append("%d %d %d" % (H, BUDGET, len(T)))
    out.append(" ".join(str(x) for x in c))
    out.append(" ".join(str(x) for x in T))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
