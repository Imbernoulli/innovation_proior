#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy assembly-stackup logbook to stdout.

A hidden variance-composition law relates the total end-to-end positional
deviation D of an n-component mechanical stack to two structural summaries:

    S2  = sum_i sigma_i^2            "RSS" term  (independent per-part variance)
    B2  = sum_j batch_size_j^2       "common-mode" term (same-batch correlation)

Components are cut in production BATCHES (lots): a run of n parts is split
into m batches of sizes b_1..b_m (sum b_j = n). Parts from the SAME batch
share a small common-mode bias (tool wear / raw-material lot / same shift),
which correlates their deviations; a batch of size b contributes ~b^2 worth
of correlated variance (vs. only ~b to the independent-part sum S2). Hidden
per-test-id law:  D = sqrt(S2 + TAU2 * B2) * (lognormal sensor noise), with a
fixed hidden coefficient TAU2 > 0 (units of sigma^2).

The TRAIN log is what the solver SEES: a short-run PILOT line. Assemblies are
small (n in [3,12]) and shipped in SMALL batches (each batch has at most 3
parts), so B2 stays roughly proportional to n and the correlated term
TAU2*B2 is a small fraction of S2 -- comparable to, or below, the measurement
noise. The HELD-OUT grading grid (regenerated ONLY inside the grader) uses
much LONGER assemblies (n in [50,220]) shipped in a HANDFUL of LARGE batches
(few big lots), so B2 grows QUADRATICALLY with n there while S2 only grows
linearly -- the correlated term becomes the dominant contributor exactly
where the pilot-line data gave it no visibility.

STDOUT prints ONLY: header "<rows> <test_id>" then rows "<n> <m> <S2> <B2> <D>".
The hidden law, its coefficients, and the seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
SEED_LAW_BASE   = 800000
SEED_TRAIN_BASE = 211
SPREAD          = math.log(1.4)   # component-to-component sigma jitter
N_TRAIN_LO, N_TRAIN_HI = 3, 12    # pilot-line assembly size range
NOISE_TRAIN     = 0.04            # multiplicative lognormal sensor noise (train)
N_TRAIN         = 160


def params(t):
    """Hidden variance-composition law for this test id (identical in gen.py
    and verify.py)."""
    rng = random.Random(SEED_LAW_BASE + t * 9176111)
    SIG0 = math.exp(rng.uniform(math.log(0.4), math.log(4.0)))   # base per-part sigma scale
    FRAC = rng.uniform(0.01, 0.06)     # correlated-term strength, as a fraction of SIG0^2
    TAU2 = SIG0 * SIG0 * FRAC
    return SIG0, TAU2


def batches_train(rng, n):
    """Pilot line: small, independent-of-n batch sizes (<=3)."""
    sizes = []
    rem = n
    while rem > 0:
        b = min(rem, rng.randint(1, 3))
        sizes.append(b)
        rem -= b
    return sizes


def gen_sigmas(rng, n, SIG0):
    return [SIG0 * math.exp(rng.uniform(-SPREAD, SPREAD)) for _ in range(n)]


def true_D(S2, B2, TAU2):
    return math.sqrt(S2 + TAU2 * B2)


def gen_train(t):
    SIG0, TAU2 = params(t)
    rng = random.Random(SEED_TRAIN_BASE + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        n = rng.randint(N_TRAIN_LO, N_TRAIN_HI)
        sizes = batches_train(rng, n)
        m = len(sizes)
        sigmas = gen_sigmas(rng, n, SIG0)
        S2 = sum(s * s for s in sigmas)
        B2 = float(sum(b * b for b in sizes))
        Dtrue = true_D(S2, B2, TAU2)
        D = Dtrue * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((n, m, S2, B2, D))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for n, m, S2, B2, D in rows:
        out.append("%d %d %.8g %.8g %.8g" % (n, m, S2, B2, D))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
