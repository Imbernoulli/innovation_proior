#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the warranty-reserve claims-triangle problem.

Reads the test id `t` from the instance header, then regenerates the FULL
hidden generative model (per-type hazard shapes, the shared reporting-lag
kernel, per-type ultimate value rates, per-cohort idiosyncratic quality
factor) via `build_instance(t)` -- byte-for-byte the same function as in
gen.py -- to recover each cohort's TRUE ultimate value U_c and hence its
true RESERVE (still-to-be-reported money) = U_c - (already reported).

The participant's artifact is one non-negative RESERVE ESTIMATE per cohort
(in input order).  Scored by exposure-weighted accuracy against the true
reserve, normalized against a floor tied to the cohort's own ultimate size
(so a cohort that is already fully developed, true reserve ~= 0, does not
blow up the relative error of a tiny nonzero guess).
"""
import sys, math

SEED_BASE = 733001
FLOOR_FRAC = 0.04
MAX_RESERVE = 1e9


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------- hidden generative model (identical to gen.py) ----------
def build_instance(t):
    import random
    rng = random.Random(SEED_BASE + 97 * t)
    if t <= 2:
        C, P, H, Lmax = 16, 2, 5, 2
    elif t <= 4:
        C, P, H, Lmax = 18, 2, 5, 2
    elif t <= 6:
        C, P, H, Lmax = 20, 3, 6, 2
    elif t <= 8:
        C, P, H, Lmax = 22, 3, 6, 3
    else:
        C, P, H, Lmax = 24, 3, 7, 3
    A_full = H + Lmax - 1

    betas = [(1.0, 6.0), (6.0, 1.0), (2.5, 2.5)]
    shapes = []
    for k in range(P):
        a, b = betas[k % 3]
        w = [(d + 1.0) ** a * (H - d) ** b for d in range(H)]
        s = sum(w)
        shapes.append([x / s for x in w])

    rho = rng.uniform(0.35, 0.55)
    gw = [rho ** l for l in range(Lmax + 1)]
    gs = sum(gw)
    g = [x / gs for x in gw]

    v = [round(rng.uniform(2.0, 6.0), 4) for _ in range(P)]

    shift_tbl = {1: 0.0, 2: 0.05, 3: 0.15, 4: 0.35, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0}
    shift = shift_tbl.get(t, 0.5)

    old_mix = [0.0] * P
    old_mix[0] = 0.85
    rem = 0.15
    for k in range(1, P):
        old_mix[k] = rem / (P - 1)
    new_mix = [0.0] * P
    slow_idx = min(1, P - 1)
    new_mix[slow_idx] = 0.85
    rem2 = 0.15
    others = [k for k in range(P) if k != slow_idx]
    for k in others:
        new_mix[k] = rem2 / len(others)

    KAPPA = 9.0
    NCAL = 2 * P
    Ctrend = C - NCAL

    mixes = [None] * C
    ages_arr = [0] * C
    for c in range(1, NCAL + 1):
        idx = c - 1
        alphas = [0.4] * P
        alphas[idx % P] = 10.0
        draw = [rng.gammavariate(a, 1.0) for a in alphas]
        s = sum(draw)
        mixes[c - 1] = [x / s for x in draw]
        ages_arr[c - 1] = A_full + (NCAL - c) + 2

    for c in range(NCAL + 1, C + 1):
        order = c - NCAL
        frac_new = (order - 1) / (Ctrend - 1) if Ctrend > 1 else 0.0
        alpha = frac_new * shift
        target = [(1 - alpha) * old_mix[k] + alpha * new_mix[k] for k in range(P)]
        alphas = [max(0.35, KAPPA * target[k]) for k in range(P)]
        draw = [rng.gammavariate(a, 1.0) for a in alphas]
        s = sum(draw)
        mixes[c - 1] = [x / s for x in draw]
        age = round((A_full - 1) * (1.0 - frac_new)) if A_full > 1 else 0
        ages_arr[c - 1] = max(0, min(A_full - 1, age))

    exposures = []
    q = []
    for c in range(1, C + 1):
        exposures.append(round(rng.uniform(300.0, 3000.0), 2))
        q.append(round(rng.uniform(0.98, 1.02), 4))

    return dict(C=C, P=P, H=H, Lmax=Lmax, A_full=A_full, shapes=shapes, g=g, v=v,
                mixes=mixes, exposures=exposures, q=q, ages=ages_arr)


def simulate(inst):
    C, P, H, Lmax, A_full = inst['C'], inst['P'], inst['H'], inst['Lmax'], inst['A_full']
    shapes, g, v = inst['shapes'], inst['g'], inst['v']
    mixes, exposures, q = inst['mixes'], inst['exposures'], inst['q']
    R = []
    U = []
    for c in range(C):
        fail = [0.0] * H
        for k in range(P):
            coeff = exposures[c] * q[c] * mixes[c][k] * v[k]
            for d in range(H):
                fail[d] += coeff * shapes[k][d]
        rep = [0.0] * (A_full + 1)
        for d in range(H):
            for l in range(Lmax + 1):
                a = d + l
                if a <= A_full:
                    rep[a] += fail[d] * g[l]
        cum = []
        s = 0.0
        for a in range(A_full + 1):
            s += rep[a]
            cum.append(s)
        R.append(cum)
        U.append(cum[A_full])
    return R, U


def finite(x):
    return isinstance(x, float) and x == x and x not in (float("inf"), float("-inf"))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        C = int(header[0]); P = int(header[1])
        t = int(header[5])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or C < 1 or C > 100000:
        fail("bad test id / dims")

    inst = build_instance(t)
    if inst['C'] != C or inst['P'] != P:
        fail("instance/header mismatch")
    R, U = simulate(inst)

    # observed "latest reported" per cohort (needed for the true reserve target)
    observed_last = []
    for c in range(C):
        m = min(inst['ages'][c], inst['A_full'])
        observed_last.append(R[c][m])

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(2_000_000)
    except Exception:
        fail("cannot read output")
    toks = raw.decode("utf-8", "replace").split()
    if len(toks) != C:
        fail("expected %d reserve values, got %d" % (C, len(toks)))

    reserve_hat = []
    for tok in toks:
        try:
            x = float(tok)
        except ValueError:
            fail("non-numeric token '%s'" % tok)
        if not finite(x):
            fail("non-finite reserve value")
        if x < -1e-6:
            fail("negative reserve estimate")
        if x > MAX_RESERVE:
            fail("reserve estimate out of range")
        reserve_hat.append(max(0.0, x))

    num = 0.0
    den = 0.0
    for c in range(C):
        reserve_true = max(0.0, U[c] - observed_last[c])
        scale = max(reserve_true, FLOOR_FRAC * U[c], 1e-6)
        err = abs(reserve_hat[c] - reserve_true) / scale
        acc = max(0.0, 1.0 - err)
        w = inst['exposures'][c]
        num += w * acc
        den += w
    F = num / max(1e-9, den)

    # ---- internal baseline B: naive "as reported" -> assume reserve == 0 everywhere ----
    num_b = 0.0
    den_b = 0.0
    for c in range(C):
        reserve_true = max(0.0, U[c] - observed_last[c])
        scale = max(reserve_true, FLOOR_FRAC * U[c], 1e-6)
        acc = max(0.0, 1.0 - reserve_true / scale)
        w = inst['exposures'][c]
        num_b += w * acc
        den_b += w
    B = max(1e-9, num_b / max(1e-9, den_b))

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
