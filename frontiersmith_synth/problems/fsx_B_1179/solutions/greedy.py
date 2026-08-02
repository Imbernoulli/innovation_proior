# TIER: greedy
"""
The "obvious" recipe: set up the per-day linear system (dilution * kernel-row) . E
= (concentration - background) exactly as the physics dictates, and solve it by
ordinary WEIGHTED least squares -- weighting each day by its RAW CONCENTRATION
MAGNITUDE, i.e. "trust the loudest / highest-quality-looking days most". No
regularization: a textbook regression, not a hand-tuned one.

This is a perfectly reasonable-sounding data-quality heuristic, and a trap.
Stagnant, low-speed days are simultaneously the LOUDEST (least dilution) and
the LEAST able to tell colocated-bearing sources apart (broad kernel, nearly
identical rows for cluster members) -- so most of the fit's weight sits on
exactly the rows that make the cluster's columns collinear. Without
regularization the unweighted-information least-squares solution for that
block is free to swing to whatever combination fits the noise, while the
few quiet, sharp, direction-varying days that would pin it down are
outweighed and barely count.
"""
import sys, math


def kernel(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0):
    r = math.hypot(sx, sy)
    brg = math.degrees(math.atan2(-sy, -sx)) % 360.0
    d = abs(wd - brg) % 360.0
    delta = d if d <= 180.0 else 360.0 - d
    sigma = SIGMA_MAX / (1.0 + ALPHA * ws)
    return math.exp(-(delta * delta) / (2.0 * sigma * sigma)) * math.exp(-r / L0)


def dilution(ws, BETA):
    return 1.0 / (1.0 + BETA * ws)


def background(day_id, A0, A1, P):
    return A0 + A1 * math.sin(2.0 * math.pi * day_id / P)


def solve_linear(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                if f != 0.0:
                    M[r] = [mrv - f * mcv for mrv, mcv in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def main():
    data = sys.stdin.read().split()
    p = 0
    test_id = int(data[p]); p += 1
    K = int(data[p]); p += 1
    D = int(data[p]); p += 1
    A0 = float(data[p]); p += 1
    A1 = float(data[p]); p += 1
    P = int(float(data[p])); p += 1
    SIGMA_MAX = float(data[p]); p += 1
    ALPHA = float(data[p]); p += 1
    L0 = float(data[p]); p += 1
    BETA = float(data[p]); p += 1

    sources = []
    for _ in range(K):
        sx = float(data[p]); p += 1
        sy = float(data[p]); p += 1
        sources.append((sx, sy))

    days = []
    for _ in range(D):
        day_id = int(data[p]); p += 1
        wd = float(data[p]); p += 1
        ws = float(data[p]); p += 1
        conc = float(data[p]); p += 1
        days.append((day_id, wd, ws, conc))

    AtWA = [[0.0] * K for _ in range(K)]
    AtWy = [0.0] * K
    for (day_id, wd, ws, conc) in days:
        kvals = [kernel(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0) for (sx, sy) in sources]
        dil = dilution(ws, BETA)
        a_row = [dil * kv for kv in kvals]
        target = conc - background(day_id, A0, A1, P)
        w = max(conc, 1e-6) ** 2.5  # trust the loudest days most (and then some)
        for i in range(K):
            AtWy[i] += w * a_row[i] * target
            ai = a_row[i]
            if ai == 0.0:
                continue
            for j in range(K):
                AtWA[i][j] += w * ai * a_row[j]

    # a nominal numerical-safety nudge only (not real regularization)
    for i in range(K):
        AtWA[i][i] += 1e-6

    E = solve_linear(AtWA, AtWy)
    E = [max(0.0, v) for v in E]
    print(" ".join("%.6f" % v for v in E))


if __name__ == "__main__":
    main()
