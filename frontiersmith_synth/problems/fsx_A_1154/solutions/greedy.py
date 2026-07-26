# TIER: greedy
"""
Textbook effective-resistance / leverage-score sparsification (Spielman-Srivastava style),
the "obvious" first attempt: rank every edge by its GLOBAL leverage score

    tau_e = w_e * R_e ,   R_e = (e_u - e_v)^T L_G^+ (e_u - e_v)

(the classic quantity that would drive importance sampling over ALL of R^n), keep the top-s
by tau (ties -> original edge order, so provable bridges -- which always have tau_e == 1 --
sort ahead of everything else), and keep their original weight. This is spectrally sound
in general but is BLIND to the fact that only a specific, published, low-dimensional test
family is ever probed here.
"""
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    pos = 0
    n = int(data[pos]); m = int(data[pos + 1]); s = int(data[pos + 2]); K = int(data[pos + 3])
    pos += 4
    edges = []
    for _ in range(m):
        u = int(data[pos]); v = int(data[pos + 1]); w = float(data[pos + 2]); pos += 3
        edges.append((u, v, w))
    pos += K * n  # skip the published test vectors entirely -- textbook approach ignores them

    L = np.zeros((n, n), dtype=np.float64)
    for (u, v, w) in edges:
        du, dv = u - 1, v - 1
        L[du, du] += w
        L[dv, dv] += w
        L[du, dv] -= w
        L[dv, du] -= w

    Lp = np.linalg.pinv(L, hermitian=True)
    diag = np.diag(Lp)

    tau = np.empty(m, dtype=np.float64)
    for i, (u, v, w) in enumerate(edges):
        du, dv = u - 1, v - 1
        R = diag[du] + diag[dv] - 2.0 * Lp[du, dv]
        tau[i] = w * max(R, 0.0)

    order = sorted(range(m), key=lambda i: -tau[i])  # stable -> ties keep original edge order
    sel = [edges[i] for i in order[:s]]

    out = [str(len(sel))]
    for (u, v, w) in sel:
        out.append(f"{u} {v} {w:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
