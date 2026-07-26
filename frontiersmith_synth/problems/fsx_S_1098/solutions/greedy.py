# TIER: greedy
"""Flat (non-composed) Chebyshev-series polynomial filter, textbook style:
fit ONE degree-D polynomial to the step function and evaluate it via the
standard 3-term matvec recurrence. Degree is tuned to the per-instance
revealed gap (so it beats the oblivious 'trivial' baseline), but it is
still a single flat polynomial -- degree must grow like ~1/gap, so it pays
dearly on the small-gap trap cases instead of exploiting the two-sided
resolvent structure.
"""
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from numpy.polynomial import chebyshev as C

DEGREE_GRID = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256,
               320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]


def read_input():
    data = sys.stdin.read().split("\n")
    data = [t for t in data if t.strip() != ""]
    idx = 0
    n = int(data[idx]); idx += 1
    theta, eps, b_bw = data[idx].split(); idx += 1
    theta = float(theta); eps = float(eps)
    gap = float(data[idx]); idx += 1
    nnz = int(data[idx]); idx += 1
    A = np.zeros((n, n))
    for _ in range(nnz):
        i, j, v = data[idx].split(); idx += 1
        i = int(i); j = int(j); v = float(v)
        A[i, j] = v; A[j, i] = v
    return n, theta, eps, gap, A


def cheb_nodes(lo, hi, N):
    # Chebyshev (cosine-spaced) nodes: fitting a high-degree polynomial on
    # UNIFORM points is Runge-unstable near the domain edges; clustering
    # samples there keeps the fit (and its coefficients) well-conditioned.
    j = np.arange(1, N + 1)
    x = np.cos((2 * j - 1) * np.pi / (2 * N))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x


def find_degree(theta, gap, eps, Lam):
    m1 = theta - gap
    m2 = theta + gap
    target = (Lam > theta).astype(float)
    for D in DEGREE_GRID:
        N = 2 * D + 80
        xs_lo = cheb_nodes(0.0, m1, N)
        xs_hi = cheb_nodes(m2, 1.0, N)
        xs = np.concatenate([xs_lo, xs_hi])
        ys = (xs > theta).astype(float)
        u = 2 * xs - 1
        c = C.chebfit(u, ys, D)
        ueig = 2 * Lam - 1
        pred = C.chebval(ueig, c)
        err = np.max(np.abs(pred - target))
        if err <= eps * 0.5:
            return c
    return c  # best effort at the largest degree tried


def emit_program(c):
    D = len(c) - 1
    lines = ["NVEC 6"]
    lines.append("COPY 1 0")
    lines.append("MATVEC 3 0")
    lines.append("AXPBY 2 2.0 3 -1.0 0")
    lines.append("SCALE 4 %.17g 1" % c[0])
    if D >= 1:
        lines.append("AXPBY 4 1.0 4 %.17g 2" % c[1])
    for k in range(2, D + 1):
        lines.append("MATVEC 3 2")
        lines.append("AXPBY 5 4.0 3 -2.0 2")
        lines.append("AXPBY 5 1.0 5 -1.0 1")
        lines.append("COPY 1 2")
        lines.append("COPY 2 5")
        lines.append("AXPBY 4 1.0 4 %.17g 2" % c[k])
    lines.append("OUTPUT 4")
    sys.stdout.write("\n".join(lines) + "\n")


def main():
    n, theta, eps, gap, A = read_input()
    Lam = np.linalg.eigvalsh(A)
    c = find_degree(theta, gap, eps, Lam)
    emit_program(c)


if __name__ == "__main__":
    main()
