#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN dispersion trace to stdout.

Setting: a specimen supports TWO propagation modes (branches) of phase
velocity vs. frequency:

  branch A (low-frequency-dominant, unbounded):  vA(f) = CA * sqrt(k * f)
  branch B (high-frequency-dominant, saturating): vB(f) = sqrt(k) * (CB - CD/(f+CE))

Both share the SAME hidden material stiffness k (> 0); CA, CB, CD, CE are
known per-specimen geometry constants (printed in the input). At each
frequency the sensor reports whichever mode actually propagates, which is
the LOWER-velocity branch (mode mixing / branch-crossing): v(f) = min(vA, vB).
Since vA(f) -> 0 as f -> 0 and vB(f) -> a finite positive constant, while
vA(f) grows without bound and vB(f) saturates, the two branches cross EXACTLY
once at some f_cross (a strictly increasing function's unique root; see
verify.py) and the identity of "the observed mode" swaps there.

Each testId fixes a DIFFERENT hidden (k, CA, CB, CD, CE) tuple, drawn so the
crossing frequency lands in a REGIME bucket controlled by (t mod 10) -- most
buckets place f_cross ABOVE the training band (so training sees ONLY branch A
and the crossing is a pure held-out surprise), a couple place it INSIDE the
training band, and one places it very early.

STDOUT prints ONLY: a header "<n_train> <test_id>", one line "CA CB CD CE",
then n_train rows "<f> <v_noisy>". k, f_cross and the RNG seed are NEVER
printed -- the hidden stiffness lives only inside the checker/gen code.
"""
import sys, random, math

N_TRAIN = 26
F_TRAIN_LO, F_TRAIN_HI = 3.0, 25.0

REGIME_BUCKET = {
    1: (30.0, 82.0), 2: (30.0, 82.0), 3: (30.0, 82.0), 4: (30.0, 82.0),
    5: (30.0, 82.0), 6: (30.0, 82.0), 7: (30.0, 82.0),
    8: (9.0, 24.0), 9: (9.0, 24.0),
    0: (2.0, 7.0),   # t % 10 == 0  (e.g. t=10)
}


def branch_consts(rng):
    while True:
        CA = rng.uniform(0.8, 1.3)
        CB = rng.uniform(3.0, 6.0)
        CD = rng.uniform(5.0, 20.0)
        CE = rng.uniform(1.0, 4.0)
        if CB - CD / CE > 0.3:
            return CA, CB, CD, CE


def find_crossing(CA, CB, CD, CE):
    """g(f) = CA*sqrt(f) - (CB - CD/(f+CE)) is STRICTLY increasing (g'>0 always),
    g(0)<0<g(+inf) by construction -> unique root via bisection."""
    def g(f):
        return CA * math.sqrt(f) - (CB - CD / (f + CE))
    lo, hi = 1e-6, 5000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def params(t):
    """Hidden dispersion parameters for this test id (duplicated verbatim in
    verify.py; never printed)."""
    bucket = REGIME_BUCKET.get(t % 10, (35.0, 260.0))
    lo_t, hi_t = bucket
    rng = random.Random(9173 + t * 7919)
    k = CA = CB = CD = CE = fc = None
    for _ in range(50000):
        k = rng.uniform(1.0, 5.0)
        CA, CB, CD, CE = branch_consts(rng)
        fc = find_crossing(CA, CB, CD, CE)
        if lo_t <= fc <= hi_t:
            break
    return k, CA, CB, CD, CE, fc


def true_v(f, k, CA, CB, CD, CE):
    vA = CA * math.sqrt(k * f)
    vB = math.sqrt(k) * (CB - CD / (f + CE))
    return min(vA, vB)


def train_freqs():
    return [F_TRAIN_LO + i * (F_TRAIN_HI - F_TRAIN_LO) / (N_TRAIN - 1) for i in range(N_TRAIN)]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    k, CA, CB, CD, CE, fc = params(t)
    sigma_rel = 0.03 + 0.01 * ((t - 1) % 5)
    rng = random.Random(555013 + t * 131071)

    freqs = train_freqs()
    out = ["%d %d" % (N_TRAIN, t), "%.6f %.6f %.6f %.6f" % (CA, CB, CD, CE)]
    for f in freqs:
        v = true_v(f, k, CA, CB, CD, CE)
        noisy = v + rng.gauss(0.0, sigma_rel * max(v, 0.5))
        out.append("%.6f %.6f" % (f, noisy))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
