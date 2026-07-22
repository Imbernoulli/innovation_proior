#!/usr/bin/env python3
# Instance generator for quantile-hedged-cargo-manifest (format C).
#   python3 gen.py <testId>   ->  ONE instance on stdout (seeded by testId only).
#
# Instance = a multidimensional (weight, volume) knapsack over N cargo items,
# each with a vector of S per-scenario payoffs.  The score of a chosen manifest
# is the 20th-percentile (k-th smallest) of its S scenario totals.
#
# PLANTED STRUCTURE (the trap): items come as ANTI-CORRELATED PAIRS keyed to a
# shared "market factor" f[s] in [0,1].  Within a pair, the HI item pays a lot
# when f is high and little when f is low; the LO item is the mirror image.  A
# common market shock (the low-f scenarios) drags every HI item down together,
# so a manifest packed with high-mean HI items craters exactly at the low
# quantile.  Hedging HI with anti-correlated LO items lifts the tail.
import sys, random


def emit(testId):
    rng = random.Random(1000 + 7919 * testId)
    S = 40
    k = 8  # 20th percentile of 40 -> 8th smallest scenario total

    # ladder: more pairs / heavier market swing / tighter caps with testId
    n_pairs = 20 + 2 * testId          # 22 .. 40 pairs
    n_filler = 28 + testId             # 29 .. 38 flat filler items
    swing = 80 + 6 * testId            # amplitude of the market factor swing

    # shared market factor per scenario.  Bimodal: a minority of "crash"
    # scenarios (low f) sit below the 20th percentile and DEFINE the score;
    # the majority are "normal" (high f) where HI items look great.  So HI has
    # the higher mean yet craters exactly at the graded tail.
    n_crash = 12
    f = []
    for s in range(S):
        if s < n_crash:
            f.append(rng.uniform(0.0, 0.22))
        else:
            f.append(rng.uniform(0.5, 1.0))
    rng.shuffle(f)

    items = []  # each: (w, v, [vals over S])

    # anti-correlated pairs keyed to f -------------------------------------
    for _ in range(n_pairs):
        qh = rng.uniform(0.90, 1.20)          # HI quality (high mean)
        ql = rng.uniform(0.85, 1.05)          # LO quality
        base_hi = rng.uniform(10, 16)
        base_lo = rng.uniform(4, 9)
        # slight idiosyncratic wobble so the pair is not perfectly rank-1
        jitter_h = [rng.uniform(-4, 4) for _ in range(S)]
        jitter_l = [rng.uniform(-4, 4) for _ in range(S)]
        hi = [max(1, int(round(qh * (base_hi + swing * f[s]) + jitter_h[s])))
              for s in range(S)]
        lo = [max(1, int(round(ql * (base_lo + swing * (1.0 - f[s])) + jitter_l[s])))
              for s in range(S)]
        wh = rng.randint(6, 15); vh = rng.randint(6, 15)
        wl = rng.randint(6, 15); vl = rng.randint(6, 15)
        items.append((wh, vh, hi))
        items.append((wl, vl, lo))

    # flat filler: moderate mean, low variance, no market exposure ---------
    for _ in range(n_filler):
        q = rng.uniform(0.8, 1.2)
        base = rng.uniform(18, 30)
        vals = [max(1, int(round(q * base + rng.uniform(-5, 5)))) for _ in range(S)]
        w = rng.randint(6, 15); vv = rng.randint(6, 15)
        items.append((w, vv, vals))

    # shuffle so file order is arbitrary (baseline sees no structure)
    rng.shuffle(items)
    N = len(items)

    sumw = sum(it[0] for it in items)
    sumv = sum(it[1] for it in items)
    # capacities bind on BOTH dims; tighter for larger testId
    frac = 0.40 - 0.010 * testId       # 0.39 .. 0.30
    W = int(round(frac * sumw))
    V = int(round(frac * sumv))

    out = ["%d %d %d %d %d" % (N, S, W, V, k)]
    for (w, v, vals) in items:
        out.append("%d %d %s" % (w, v, " ".join(str(x) for x in vals)))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <testId>\n"); sys.exit(2)
    emit(int(sys.argv[1]))


if __name__ == "__main__":
    main()
