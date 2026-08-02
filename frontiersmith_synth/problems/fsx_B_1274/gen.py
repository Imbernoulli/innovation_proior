#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE claims-development instance to stdout.

A warranty book of C cohorts (product launch batches), each a blend of P
hidden product TYPES.  Each type has its own failure-hazard SHAPE over
development age (when, relative to a cohort's own origin, units tend to
fail) and the book shares one reporting-LAG kernel (extra delay between a
failure occurring and the claim being recorded).  Older cohorts are more
"developed" (observed for longer) than newer ones -- the classic triangle.

Recent cohorts drift toward a slower-developing type mix (the mix-shift
trap).  The hidden per-type hazard shapes, the lag kernel, the per-type
ultimate value rates, and each cohort's small idiosyncratic quality factor
are NEVER printed -- they live only in `build_instance` (identical copy in
gen.py and verify.py) and are reconstructed deterministically from the test
id alone, exactly like the visible instance is.
"""
import sys

SEED_BASE = 733001


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
    NCAL = 2 * P            # oldest NCAL cohorts = pure-type calibration batches (always mature)
    Ctrend = C - NCAL       # remaining cohorts follow the age-trending mix shift (always immature)

    mixes = [None] * C
    ages_arr = [0] * C
    for c in range(1, NCAL + 1):
        idx = c - 1
        alphas = [0.4] * P
        alphas[idx % P] = 10.0
        draw = [rng.gammavariate(a, 1.0) for a in alphas]
        s = sum(draw)
        mixes[c - 1] = [x / s for x in draw]
        ages_arr[c - 1] = A_full + (NCAL - c) + 2   # comfortably mature

    for c in range(NCAL + 1, C + 1):
        order = c - NCAL                            # 1..Ctrend
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


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inst = build_instance(t)
    R, U = simulate(inst)
    C, P, H, Lmax, A_full = inst['C'], inst['P'], inst['H'], inst['Lmax'], inst['A_full']

    out = ["%d %d %d %d %d %d" % (C, P, H, Lmax, A_full, t)]
    for c in range(C):
        age = inst['ages'][c]
        m = min(age, A_full)
        K = m + 1
        row = ["%.6f" % inst['exposures'][c], "%d" % age]
        row += ["%.6f" % x for x in inst['mixes'][c]]
        row.append("%d" % K)
        row += ["%.6f" % R[c][a] for a in range(K)]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
