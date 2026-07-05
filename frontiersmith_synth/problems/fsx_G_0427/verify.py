#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>     (ans is ignored)

Deterministic grader for the turbulence-lab dimensionless-invariant task.

The participant reads a TRAIN table (channels x0..x5 across many experiments) and
must output an INTEGER exponent vector a = (a0..a5), |a_i| <= AMAX, not all zero,
defining a candidate dimensionless power-law group

        Pi = prod_i  x_i ** a_i .

Scoring (MAXIMIZATION -- "how invariant is Pi on a held-out, higher-Reynolds
regime"):
  * The checker recovers the case id from <in>, then regenerates the HELD-OUT
    EXTRAPOLATION split (a Reynolds regime strictly above the training rows)
    entirely from that id -- the generative law lives ONLY here.
  * On the held-out log-data (per-channel centered), let
        proj_j = sum_i a_i * (log x_ij - mean_i)
        s(a)   = std_j(proj_j) / ||a||_2            # scale-free spread per unit exponent
    Small s == strong cancellation == invariant group.  Dividing by ||a||_2
    removes the a -> c*a degeneracy; a == 0 is rejected outright.
        M(a) = -log10(s + 1e-12)                    # invariance, higher = better
        F(a) = max(0, M(a) - BETA * sum_i |a_i|)    # mild parsimony penalty
  * Baseline B = F(V0) for a fixed mediocre reference group V0 that the checker
    builds itself.  Ratio = min(1000, 100*F/B)/1000.

An irreducible noise floor + a residual finite-Reynolds drift keep even the best
real invariant well below the 1.0 cap (headroom).  A group that overfits the
TRAIN-only latent coupling (channel x4) loses on the decoupled held-out regime,
so held-out scoring rewards generalization, not memorization.
"""
import sys, math, random

NCOLS = 6
AMAX = 6
BETA = 0.02
V0 = [-2, 1, 1, 0, 0, 0]        # fixed reference group (the checker's baseline)

# ---- generative constants: identical to gen.py ----
CINF = 0.5
ACORR = 5.0
SIGMA_G = 0.20
U_LO, U_HI = 0.05, 5.0
L_LO, L_HI = 0.01, 1.0
M_LO, M_HI = 0.001, 0.05
W_LO, W_HI = 1.0, 100.0
RE_HELD_LO = 3.0e4
N_HELD = 300


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def params(t):
    ntrain = max(90, 250 - 15 * t)
    sigma_meas = 0.03 + 0.005 * t
    re_held_hi = 3.0e4 * (2 + t)
    return ntrain, sigma_meas, re_held_hi


def _uni_log(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def make_row(rng, re_lo, re_hi, sigma_meas, is_train):
    Re = math.exp(rng.uniform(math.log(re_lo), math.log(re_hi)))
    U = _uni_log(rng, U_LO, U_HI)
    L = _uni_log(rng, L_LO, L_HI)
    nu = U * L / Re
    Re_lambda = math.sqrt(15.0 * Re)
    Ceps = CINF * (1.0 + ACORR / Re_lambda)
    eps_clean = Ceps * U ** 3 / L
    baseM = _uni_log(rng, M_LO, M_HI)
    baseW = _uni_log(rng, W_LO, W_HI)
    g = rng.gauss(0.0, 1.0)
    if is_train:
        g_eps = g
        g_M = g
    else:
        g_eps = rng.gauss(0.0, 1.0)
        g_M = rng.gauss(0.0, 1.0)
    eps = eps_clean * math.exp(SIGMA_G * g_eps)
    M = baseM * math.exp(SIGMA_G * g_M)
    W = baseW
    cols = [U, L, eps, nu, M, W]
    cols = [c * math.exp(sigma_meas * rng.gauss(0.0, 1.0)) for c in cols]
    return cols


def gen_heldout(t):
    _, sigma_meas, re_held_hi = params(t)
    rng = random.Random(500009 * t + 13)
    return [make_row(rng, RE_HELD_LO, re_held_hi, sigma_meas, False)
            for _ in range(N_HELD)]


def read_testid(path):
    with open(path) as f:
        first = f.readline().split()
    if len(first) < 2 or first[0] != "TESTID":
        fail("bad instance header")
    return int(first[1])


def parse_exponents(path):
    try:
        with open(path) as f:
            data = f.read(100000)
    except Exception:
        fail("cannot read output")
    toks = data.split()
    if len(toks) != NCOLS:
        fail("expected exactly %d integer exponents, got %d" % (NCOLS, len(toks)))
    a = []
    for tk in toks:
        try:
            v = float(tk)
        except ValueError:
            fail("non-numeric exponent %r" % tk)
        if not math.isfinite(v):
            fail("non-finite exponent")
        if abs(v - round(v)) > 1e-9:
            fail("exponent %r is not an integer" % tk)
        iv = int(round(v))
        if abs(iv) > AMAX:
            fail("exponent %d exceeds |a| <= %d" % (iv, AMAX))
        a.append(iv)
    if all(x == 0 for x in a):
        fail("all-zero exponent vector is not a group")
    return a


def eval_F(a, logcols, means):
    """logcols[i] = list of log x over held-out rows; means[i] = column mean."""
    n = len(logcols[0])
    norm2 = sum(x * x for x in a)
    if norm2 == 0:
        return 0.0
    s1 = 0.0
    s2 = 0.0
    for j in range(n):
        p = 0.0
        for i in range(NCOLS):
            if a[i]:
                p += a[i] * (logcols[i][j] - means[i])
        s1 += p
        s2 += p * p
    var = s2 / n - (s1 / n) ** 2
    if var < 0.0:
        var = 0.0
    s = math.sqrt(var / norm2)
    M = -math.log10(s + 1e-12)
    c = sum(abs(x) for x in a)
    F = M - BETA * c
    if not math.isfinite(F) or F < 0.0:
        return 0.0
    return F


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    t = read_testid(in_path)
    a = parse_exponents(out_path)

    held = gen_heldout(t)
    logcols = [[math.log(row[i]) for row in held] for i in range(NCOLS)]
    means = [sum(col) / len(col) for col in logcols]

    F = eval_F(a, logcols, means)
    B = eval_F(V0, logcols, means)
    if not math.isfinite(F):
        fail("non-finite objective")
    if B <= 1e-9:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.6f baseline=%.6f exps=%s  Ratio: %.6f"
          % (F, B, ",".join(str(x) for x in a), sc / 1000.0))


if __name__ == "__main__":
    main()
