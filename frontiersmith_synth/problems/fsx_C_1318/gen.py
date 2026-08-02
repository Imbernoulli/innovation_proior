#!/usr/bin/env python3
"""gen.py <testId> -- prints one alloy-composition-search instance to stdout.

Instance format:
    K W numBins
    s_1 s_2 ... s_K       (solid-solution strengthening coefficients)
    b_1 b_2 ... b_K       (intermetallic-forming tendency, always >= 1)
    T_0 T_1 ... T_{numBins-1}   (per-bin intermetallic-score budget; bin k covers
                                 total solute X in [k*W, (k+1)*W - 1])

MAXX = numBins*W - 1 is the hard total-solute cap.

Every b_i >= 1. Band thresholds T[k] are scaled around AVG_B = 5 (the midpoint
of the b_i range [1,9]) so that a composition using elements of roughly
AVERAGE brittleness tendency can reach a meaningful way into most bands --
reachability doesn't depend on an instance happening to contain an extremal
b_i=1 element. T[k] is deliberately NOT required to stay below (k+1)*W: a
below-average-cost mix can legitimately push a band's achievable X past its
"nominal" width, and a generous band's threshold can comfortably exceed what
an average-cost mix spends reaching its own top edge. Feasibility is always
decided directly from the submitted composition's own (X, IM): X // W picks
the band, IM <= T[band] is the gate. Nothing about the geometry requires
monotonicity band-to-band.

Deterministic: everything is seeded from testId only.
"""
import random
import sys

AVG_B = 5.0


def build_case(testId: int):
    rng = random.Random(1000003 * testId + 20831)

    K_by_test = {1: 3, 2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 5, 8: 6, 9: 6, 10: 6}
    K = K_by_test[testId]
    W = 20
    numBins = 10
    MAXX = numBins * W - 1  # = 199

    b = [rng.randint(1, 9) for _ in range(K)]
    # mild positive correlation: the strongest strengtheners tend to also be
    # decent intermetallic formers, but noise keeps some low-b elements
    # competitive -- the solver must actually compute the trade-off, not guess it.
    s = [rng.randint(15, 60) + 3 * b[i] for i in range(K)]

    # >=3 of the 10 cases get a genuine retrograde-solvus dip: a generous region,
    # then a narrow brittle band, then an even more generous region beyond it that
    # a single monotone incremental walk (stop-at-first-wall) will never reach.
    dip_tests = {3, 4, 5, 6, 7, 8, 9, 10}

    T = [0] * numBins
    if testId in dip_tests:
        # vary dip geometry per test for ladder diversity
        d1 = 2 + (testId % 3)              # dip starts right after bin d1 (2..4)
        d2 = d1 + 2 + (testId % 2)         # dip spans 3-4 bins
        d2 = min(d2, numBins - 3)          # always leave room for a recovery tail
        dip_frac = 0.08 + 0.02 * (testId % 3)      # 0.08 .. 0.12
        pre_frac = 0.80 + 0.03 * (testId % 2)      # 0.80 .. 0.83
        rec_frac = 1.15 + 0.05 * (testId % 3)      # 1.15 .. 1.25
        for k in range(numBins):
            base = AVG_B * (k + 1) * W
            if k <= d1:
                frac = pre_frac
            elif k <= d2:
                frac = dip_frac
            else:
                frac = rec_frac
            T[k] = max(1, int(frac * base))
    else:
        for k in range(numBins):
            base = AVG_B * (k + 1) * W
            T[k] = max(1, int(0.85 * base))

    return K, W, numBins, s, b, T


def main():
    testId = int(sys.argv[1])
    K, W, numBins, s, b, T = build_case(testId)
    out = []
    out.append(f"{K} {W} {numBins}")
    out.append(' '.join(map(str, s)))
    out.append(' '.join(map(str, b)))
    out.append(' '.join(map(str, T)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
