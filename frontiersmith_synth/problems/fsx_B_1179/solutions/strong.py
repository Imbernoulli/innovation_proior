# TIER: strong
"""
Insight: on any single day, colocated-bearing sources are confounded -- the
only thing that ever separates them is how much wind DIRECTION differs across
days. So weight each day not by how loud it is, but by how clearly wind
DIRECTION ALONE (the ANGULAR part of the kernel, with the day-independent
distance decay factored back out -- two sources at different distances always
have different raw kernel magnitude, even on a day that cannot tell their
BEARINGS apart) singles out one dominant candidate: compare the day's largest
angular-kernel value to its runner-up. A stagnant, broad-kernel day aimed at a
cluster leaves two or three members in a near tie (gap ~ 0) -- worthless for
telling them apart however loud it reads. A sharp, well-aimed day gives one
bearing a clearly dominant angular kernel (gap ~ 1) -- worth a lot however
quiet.

Correctly keys the background trend by the explicit day_id field (not row
order), then solves a ridge-regularized WEIGHTED least-squares system for the
full rate vector using those information weights (ridge applied as a small
RELATIVE per-parameter loading, so it does not crush sources that are only
ever visible on one sparse, low-magnitude day), clipping to non-negative.
"""
import sys, math


def kernel_parts(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0):
    """Return (full_kernel, angular_only_kernel) -- the latter has the
    day-independent distance decay factored out, so it isolates what THIS
    day's wind direction alone can discriminate."""
    r = math.hypot(sx, sy)
    brg = math.degrees(math.atan2(-sy, -sx)) % 360.0
    d = abs(wd - brg) % 360.0
    delta = d if d <= 180.0 else 360.0 - d
    sigma = SIGMA_MAX / (1.0 + ALPHA * ws)
    ang = math.exp(-(delta * delta) / (2.0 * sigma * sigma))
    return ang * math.exp(-r / L0), ang


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

    rows = []          # design row a_d,i = dilution(d)*kernel_d,i
    targets = []        # conc - background(day_id)
    infos = []           # information weight of the day
    for (day_id, wd, ws, conc) in days:
        parts = [kernel_parts(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0) for (sx, sy) in sources]
        kvals = [f for (f, a) in parts]
        angs = [a for (f, a) in parts]
        top2 = sorted(angs, reverse=True)[:2]
        a1 = top2[0]
        a2 = top2[1] if len(top2) > 1 else 0.0
        info = (a1 - a2) / (a1 + a2 + 1e-9)  # 0 = tied bearings, 1 = one clear bearing
        dil = dilution(ws, BETA)
        a_row = [dil * kv for kv in kvals]
        target = conc - background(day_id, A0, A1, P)
        rows.append(a_row)
        targets.append(target)
        infos.append(info)

    mean_info = sum(infos) / len(infos) if infos else 1.0
    weights = [(inf + 0.05 * mean_info) for inf in infos]  # small floor keeps every day usable

    AtWA = [[0.0] * K for _ in range(K)]
    AtWy = [0.0] * K
    for a_row, t, w in zip(rows, targets, weights):
        for i in range(K):
            AtWy[i] += w * a_row[i] * t
            ai = a_row[i]
            if ai == 0.0:
                continue
            for j in range(K):
                AtWA[i][j] += w * ai * a_row[j]

    # RELATIVE (per-parameter) ridge: load each diagonal by a small fraction of
    # ITSELF, not of the K-wide average -- an additive average-based ridge would
    # be negligible for an always-visible cluster column but would swamp (crush
    # toward zero) a source that is only ever seen on one sparse, low-magnitude
    # anchor day, which is exactly the opposite of what we want.
    floor = 1e-9 * max((AtWA[i][i] for i in range(K)), default=1.0)
    for i in range(K):
        AtWA[i][i] = AtWA[i][i] * 1.06 + floor

    E = solve_linear(AtWA, AtWy)
    E = [max(0.0, v) for v in E]
    print(" ".join("%.6f" % v for v in E))


if __name__ == "__main__":
    main()
