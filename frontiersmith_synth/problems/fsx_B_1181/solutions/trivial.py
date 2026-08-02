# TIER: trivial
"""Reproduces the checker's own naive reference: ONE generic RC branch at a
fixed, data-independent time constant (60s), plus hysteresis, fit by
ordinary least squares against the visible drive cycle. No attempt is made
to look at what the excitation actually contains."""
import sys
import math


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
        # linear interpolation over the given table
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

    T_GENERIC = 60.0
    a = math.exp(-dt / T_GENERIC)
    ah = math.exp(-dt / tau_h)
    Zb = [0.0] * n
    Zh = [0.0] * n
    sb, sh = 0.0, 0.0
    for k in range(n):
        Zb[k] = sb
        Zh[k] = sh
        sb = a * sb + (1 - a) * I[k]
        sgn = 0.0 if I[k] == 0 else (1.0 if I[k] > 0 else -1.0)
        sh = ah * sh + (1 - ah) * sgn

    cols = [I, Zb, Zh]
    ATA = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(3)] for i in range(3)]
    ATb = [sum(cols[i][k] * target[k] for k in range(n)) for i in range(3)]

    def solve3(A, b):
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
            M[col], M[piv] = M[piv], M[col]
            pv = M[col][col] if abs(M[col][col]) > 1e-12 else 1e-12
            M[col] = [x / pv for x in M[col]]
            for r in range(3):
                if r != col:
                    f = M[r][col]
                    M[r] = [M[r][c] - f * M[col][c] for c in range(4)]
        return [M[i][3] for i in range(3)]

    R0, Rb, M = solve3(ATA, ATb)
    R0 = min(max(R0, r0_lo), r0_hi)
    Rb = min(max(Rb, r_lo), r_hi)
    M = min(max(M, m_lo), m_hi)
    C = T_GENERIC / Rb if Rb > 1e-9 else c_lo

    print("%.8f 1 %.8f" % (R0, M))
    print("%.8f %.4f" % (Rb, C))


if __name__ == "__main__":
    main()
