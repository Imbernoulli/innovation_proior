#!/usr/bin/env python3
# Deterministic checker for Split-Season Nutrient Scheduling (format C, maximize
# value-weighted plant uptake under a leaching + ion-antagonism soil model).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys, math

TOL = 1e-6
PASS_EPS = 1e-9


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def simulate(T, retain, kappa, D, v, A):
    """Run the soil-plant model for one nutrient trio schedule A over T steps.
    D, v, A are dicts keyed 'N','K','Mg'. Returns total value-weighted uptake."""
    pN = pK = pMg = 0.0
    rN, rK, rMg = retain
    vN, vK, vMg = v
    F = 0.0
    AN, AK, AMg = A['N'], A['K'], A['Mg']
    DN, DK, DMg = D['N'], D['K'], D['Mg']
    for t in range(T):
        pN = pN * rN + AN[t]
        pK = pK * rK + AK[t]
        pMg = pMg * rMg + AMg[t]

        upN = min(pN, DN[t])
        upK = min(pK, DK[t])
        # antagonistic-ion-uptake: high K pool relative to Mg pool throttles Mg uptake.
        avail_frac = kappa * pMg / max(pK, 1e-9)
        if avail_frac > 1.0:
            avail_frac = 1.0
        upMg = min(pMg, DMg[t] * avail_frac)

        pN -= upN
        pK -= upK
        pMg -= upMg

        F += vN * upN + vK * upK + vMg * upMg
    return F


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = iter(itoks)
        T = int(next(p)); P = int(next(p))
        vN = float(next(p)); vK = float(next(p)); vMg = float(next(p))
        rN = float(next(p)); rK = float(next(p)); rMg = float(next(p))
        kappa = float(next(p))
        BN = float(next(p)); BK = float(next(p)); BMg = float(next(p))
        DN = []; DK = []; DMg = []
        for _ in range(T):
            DN.append(float(next(p))); DK.append(float(next(p))); DMg.append(float(next(p)))
    except Exception:
        fail("bad instance")

    try:
        olines = [ln for ln in open(sys.argv[2]).read().split("\n") if ln.strip() != ""]
    except Exception:
        fail("no output")

    if len(olines) != T:
        fail("expected exactly %d lines (one per day), got %d" % (T, len(olines)))

    AN = []; AK = []; AMg = []
    for t in range(T):
        parts = olines[t].split()
        if len(parts) != 3:
            fail("day %d: expected 3 numbers, got %d" % (t + 1, len(parts)))
        try:
            aN = float(parts[0]); aK = float(parts[1]); aMg = float(parts[2])
        except Exception:
            fail("bad amount on day %d" % (t + 1))
        if not (math.isfinite(aN) and math.isfinite(aK) and math.isfinite(aMg)):
            fail("non-finite amount on day %d" % (t + 1))
        if aN < -TOL or aK < -TOL or aMg < -TOL:
            fail("negative application on day %d" % (t + 1))
        AN.append(max(0.0, aN)); AK.append(max(0.0, aK)); AMg.append(max(0.0, aMg))

    # budget feasibility (relative tolerance)
    sN, sK, sMg = sum(AN), sum(AK), sum(AMg)
    if sN > BN * (1.0 + 1e-6) + 1e-6:
        fail("N budget exceeded: %.6f > %.6f" % (sN, BN))
    if sK > BK * (1.0 + 1e-6) + 1e-6:
        fail("K budget exceeded: %.6f > %.6f" % (sK, BK))
    if sMg > BMg * (1.0 + 1e-6) + 1e-6:
        fail("Mg budget exceeded: %.6f > %.6f" % (sMg, BMg))

    # pass-budget feasibility: a "pass" is any day with a nonzero application
    passes = sum(1 for t in range(T) if (AN[t] + AK[t] + AMg[t]) > PASS_EPS)
    if passes > P:
        fail("used %d passes > budget %d" % (passes, P))

    D = {'N': DN, 'K': DK, 'Mg': DMg}
    v = (vN, vK, vMg)
    retain = (rN, rK, rMg)

    F = simulate(T, retain, kappa, D, v, {'N': AN, 'K': AK, 'Mg': AMg})

    # internal trivial baseline: front-load the ENTIRE seasonal budget of each
    # nutrient in a single application on day 1 (the "efficient in labour" trap:
    # 1 pass total, leaches away before most of the uptake curve).
    A0N = [0.0] * T; A0K = [0.0] * T; A0Mg = [0.0] * T
    A0N[0] = BN; A0K[0] = BK; A0Mg[0] = BMg
    B = simulate(T, retain, kappa, D, v, {'N': A0N, 'K': A0K, 'Mg': A0Mg})

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f passes=%d/%d Ratio: %.6f" % (F, B, passes, P, sc / 1000.0))


if __name__ == "__main__":
    main()
