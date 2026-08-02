#!/usr/bin/env python3
# Deterministic checker for structural-damage-modal (format C, MAXIMIZE).
# CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
#
# The ground truth crack (x*, s*) is NEVER written to the .in file -- it is
# recovered here by re-running the exact same deterministic construction
# gen.py used, keyed only by the testId that IS printed in the .in file
# (harness runs solutions OS-sandboxed with this source tree hidden, so a
# participant program can never read this recomputation).
import sys, math, random

L_LADDER = [10, 14, 18, 22, 28, 34, 42, 52, 65, 80]
K_LADDER = [3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
G_LADDER = [7, 8, 8, 9, 9, 10, 10, 11, 12, 13]
TRAP_IDS = {3, 4, 5, 7, 9}

KMAX = 14
S_MIN, S_MAX = 0.08, 0.30
SIGMA_F = 0.0035
SIGMA_SHAPE = 0.012
W_BUMP_FRAC = 0.10

S_MAX_OUT = 0.5
TAU_LOC, TAU_SEV, TAU_HOLD = 0.05, 0.07, 0.05
H_HOLD = 3


def _phi(m, x, L):
    return math.cos(m * math.pi * x / L)


def _nearest_node_dist(m, x, L):
    jf = (x * 2 * m / L + 1) / 2.0
    j = max(1, min(m, round(jf)))
    node = (2 * j - 1) * L / (2 * m)
    return abs(x - node)


def build_instance(t):
    t = max(1, min(10, t))
    idx = t - 1
    L, K, G = L_LADDER[idx], K_LADDER[idx], G_LADDER[idx]
    rng = random.Random(31337 + 101 * t)

    modes = sorted(rng.sample(range(2, KMAX + 1), K))
    m1 = modes[0]
    sstar = round(rng.uniform(S_MIN, S_MAX), 4)

    if t in TRAP_IDS:
        js = list(range(1, m1 + 1))
        rng.shuffle(js)
        xstar = None
        for j in js:
            cand = (2 * j - 1) * L / (2 * m1)
            if abs(cand - L / 2) > 0.08 * L and 0.03 * L < cand < 0.97 * L:
                xstar = cand
                break
        if xstar is None:
            xstar = 0.31 * L
    else:
        xstar = None
        for _ in range(300):
            cand = rng.uniform(0.05 * L, 0.95 * L)
            if abs(cand - L / 2) < 0.08 * L:
                continue
            if all(_nearest_node_dist(m, cand, L) > 0.12 * (L / m) for m in modes):
                xstar = cand
                break
        if xstar is None:
            xstar = 0.4 * L

    f0, fdam = [], []
    for m in modes:
        base = 3.0 + 0.7 * m + 0.15 * (m % 3)
        shift = sstar * _phi(m, xstar, L) ** 2
        noise = rng.gauss(0.0, SIGMA_F)
        f0.append(base)
        fdam.append(base * (1.0 - shift + noise))

    gpts = [L * g / (G + 1) for g in range(1, G + 1)]
    w = W_BUMP_FRAC * L
    shape_u, shape_d = [], []
    for m in modes:
        urow = [_phi(m, xg, L) for xg in gpts]
        drow = []
        for xg in gpts:
            bump = math.exp(-((xg - xstar) / w) ** 2)
            val = _phi(m, xg, L) * (1.0 - sstar * bump) + rng.gauss(0.0, SIGMA_SHAPE)
            drow.append(val)
        shape_u.append(urow)
        shape_d.append(drow)

    return dict(t=t, L=L, K=K, G=G, modes=modes, sstar=sstar, xstar=xstar,
                f0=f0, fdam=fdam, gpts=gpts, shape_u=shape_u, shape_d=shape_d)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def score_F(xh, sh, L, xstar, sstar, hold_modes):
    loc = math.exp(-abs(xh - xstar) / (TAU_LOC * L))
    sev = math.exp(-abs(sh - sstar) / TAU_SEV)
    holds = []
    for m in hold_modes:
        pred = sh * math.cos(m * math.pi * xh / L) ** 2
        true = sstar * math.cos(m * math.pi * xstar / L) ** 2
        holds.append(math.exp(-abs(pred - true) / TAU_HOLD))
    hold = sum(holds) / len(holds)
    return 0.35 * loc + 0.15 * sev + 0.50 * hold


def main():
    try:
        itxt = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    try:
        p = 0
        t = int(itxt[p]); p += 1
        L = int(itxt[p]); p += 1
        G = int(itxt[p]); p += 1
        K = int(itxt[p]); p += 1
        modes = [int(itxt[p + i]) for i in range(K)]; p += K
        f0 = [float(itxt[p + i]) for i in range(K)]; p += K
        fdam = [float(itxt[p + i]) for i in range(K)]; p += K
        gpts = [float(itxt[p + i]) for i in range(G)]; p += G
        shape_u = []
        for i in range(K):
            shape_u.append([float(itxt[p + j]) for j in range(G)]); p += G
        shape_d = []
        for i in range(K):
            shape_d.append([float(itxt[p + j]) for j in range(G)]); p += G
        if len(itxt) != p:
            fail("trailing tokens in instance")
    except Exception:
        fail("bad instance parse")

    inst = build_instance(t)
    if inst["L"] != L or inst["G"] != G or inst["K"] != K or inst["modes"] != modes:
        fail("instance/testId mismatch")
    for a, b in zip(inst["f0"], f0):
        if abs(a - b) > 1e-3:
            fail("instance f0 mismatch")
    for a, b in zip(inst["fdam"], fdam):
        if abs(a - b) > 1e-3:
            fail("instance fdam mismatch")

    xstar, sstar = inst["xstar"], inst["sstar"]

    hold_rng = random.Random(70021 + 13 * t)
    hold_candidates = [m for m in range(2, KMAX + 1) if m not in modes]
    hold_modes = sorted(hold_rng.sample(hold_candidates, min(H_HOLD, len(hold_candidates))))

    try:
        otxt = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(otxt) < 2:
        fail("expected 2 numbers: x_hat s_hat")
    try:
        x_hat = float(otxt[0])
        s_hat = float(otxt[1])
    except Exception:
        fail("non-numeric output")
    if not (math.isfinite(x_hat) and math.isfinite(s_hat)):
        fail("non-finite output")
    if not (-1e-6 <= x_hat <= L + 1e-6):
        fail("x_hat out of [0,L]")
    if not (-1e-6 <= s_hat <= S_MAX_OUT + 1e-6):
        fail("s_hat out of [0,S_MAX_OUT]")
    x_hat = max(0.0, min(float(L), x_hat))
    s_hat = max(0.0, min(S_MAX_OUT, s_hat))

    F = score_F(x_hat, s_hat, L, xstar, sstar, hold_modes)

    # ---- internal baseline: ignore location entirely (midpoint), estimate
    # severity from the plain average observed |relative frequency drop|
    # across the measured modes (a reasonable, data-driven, but
    # location-blind guess -- always feasible). ---------------------------
    x_B = L / 2.0
    rel = [abs(1.0 - fdam[i] / f0[i]) for i in range(K)]
    s_B = max(0.0, min(S_MAX_OUT, sum(rel) / len(rel)))
    B = score_F(x_B, s_B, L, xstar, sstar, hold_modes)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("t=%d F=%.6f B=%.6f Ratio: %.6f" % (t, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
