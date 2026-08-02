# TIER: strong
"""The insight: don't claim more RC branches than the drive cycle you were
actually given can resolve. Look at the current profile's own pulse-duration
content (run-length segments of constant current), group those durations
into well-separated clusters (a plain 1-D gap-based clustering on log
duration), and place exactly ONE candidate branch per surviving cluster
(one with enough support to be real, not a stray single pulse). Two
timescales that never get excited by cleanly separated pulse widths in THIS
cycle collapse into one cluster -- and hence one branch -- instead of being
force-split into two collinear ones. Only after that data-driven candidate
set is fixed does it fit (R, C) values against the training data."""
import sys
import math


def gauss_solve(A, b):
    n = len(b)
    if n == 0:
        return []
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] if abs(M[col][col]) > 1e-12 else 1e-12
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[col][c] for c in range(n + 1)]
    return [M[i][n] for i in range(n)]


def cluster_durations(I, gap_log_ratio=math.log(1.8), min_support=3):
    runs = []
    cur = I[0]
    rl = 1
    for k in range(1, len(I)):
        if I[k] == cur:
            rl += 1
        else:
            runs.append(rl)
            cur = I[k]
            rl = 1
    runs.append(rl)
    logs = sorted(math.log(r) for r in runs)
    clusters = []
    cur_c = [logs[0]]
    for x in logs[1:]:
        if x - cur_c[-1] > gap_log_ratio:
            clusters.append(cur_c)
            cur_c = [x]
        else:
            cur_c.append(x)
    clusters.append(cur_c)
    reps = []
    for c in clusters:
        if len(c) >= min_support:
            reps.append(math.exp(sorted(c)[len(c) // 2]))
    return reps


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    nxt = lambda: next(it)

    n = int(nxt()); tid = int(nxt()); dt = float(nxt()); kmax = int(nxt())
    r0_lo = float(nxt()); r0_hi = float(nxt())
    r_lo = float(nxt()); r_hi = float(nxt())
    c_lo = float(nxt()); c_hi = float(nxt())
    tau_lo = float(nxt()); tau_hi = float(nxt())
    m_lo = float(nxt()); m_hi = float(nxt())
    tau_h = float(nxt()); cap_ah = float(nxt()); soc_init = float(nxt())
    n_ocv = int(nxt())
    ocv_pts = []
    for _ in range(n_ocv):
        s = float(nxt()); o = float(nxt())
        ocv_pts.append((s, o))
    I = [float(nxt()) for _ in range(n)]
    V = [float(nxt()) for _ in range(n)]

    def ocv(soc):
        if soc <= ocv_pts[0][0]:
            return ocv_pts[0][1]
        if soc >= ocv_pts[-1][0]:
            return ocv_pts[-1][1]
        for i in range(1, len(ocv_pts)):
            s0, o0 = ocv_pts[i - 1]
            s1, o1 = ocv_pts[i]
            if soc <= s1:
                f = (soc - s0) / (s1 - s0) if s1 != s0 else 0.0
                return o0 + f * (o1 - o0)
        return ocv_pts[-1][1]

    socs = []
    soc = soc_init
    for k in range(n):
        socs.append(soc)
        soc -= I[k] * dt / (3600.0 * cap_ah)
    target = [ocv(socs[k]) - V[k] for k in range(n)]

    reps = cluster_durations(I)
    reps = [t for t in reps if tau_lo <= t <= tau_hi]
    reps = reps[:kmax]
    if not reps:
        reps = [60.0]
    K = len(reps)

    a = [math.exp(-dt / t) for t in reps]
    ah = math.exp(-dt / tau_h)
    Zb = [[0.0] * n for _ in range(K)]
    Zh = [0.0] * n
    states = [0.0] * K
    sh = 0.0
    for k in range(n):
        for i in range(K):
            Zb[i][k] = states[i]
            states[i] = a[i] * states[i] + (1 - a[i]) * I[k]
        Zh[k] = sh
        sgn = 0.0 if I[k] == 0 else (1.0 if I[k] > 0 else -1.0)
        sh = ah * sh + (1 - ah) * sgn

    cols = [I] + Zb + [Zh]
    m = len(cols)
    ATA = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(m)] for i in range(m)]
    ATb = [sum(cols[i][k] * target[k] for k in range(n)) for i in range(m)]
    coef = gauss_solve(ATA, ATb)

    R0 = min(max(coef[0], r0_lo), r0_hi)
    Rs = [min(max(coef[1 + i], r_lo), r_hi) for i in range(K)]
    M = min(max(coef[-1], m_lo), m_hi)

    branches = []
    for tau, R in zip(reps, Rs):
        if R <= 0:
            R = r_lo
        C = tau / R
        C = min(max(C, c_lo), c_hi)
        branches.append((R, C))

    print("%.8f %d %.8f" % (R0, len(branches), M))
    for R, C in branches:
        print("%.8f %.4f" % (R, C))


if __name__ == "__main__":
    main()
