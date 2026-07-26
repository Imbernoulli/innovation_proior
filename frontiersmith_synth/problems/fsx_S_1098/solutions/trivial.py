# TIER: trivial
"""Flat Chebyshev-series filter at a FIXED, gap-oblivious safety degree.

Never looks at the revealed gap; always assumes a conservative worst-case
gap and always emits the same large degree D_TRIV. Safe (always accurate
enough) but wasteful -- the calibrated baseline the checker compares against.
"""
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from numpy.polynomial import chebyshev as C

D_TRIV = 350
WORST_GAP = 0.008  # hardcoded conservative assumption; input gap is IGNORED


def read_input():
    data = sys.stdin.read().split("\n")
    data = [t for t in data if t.strip() != ""]
    idx = 0
    n = int(data[idx]); idx += 1
    theta, eps, b_bw = data[idx].split(); idx += 1
    theta = float(theta); eps = float(eps)
    idx += 1  # gap line -- intentionally unused by trivial
    nnz = int(data[idx]); idx += 1
    idx += nnz  # skip matrix entries; trivial doesn't need matrix values
    return n, theta, eps


def cheb_nodes(lo, hi, N):
    # Chebyshev (cosine-spaced) nodes on [lo, hi]: fitting a HIGH-degree
    # polynomial on UNIFORM points is Runge-phenomenon-unstable near the
    # domain edges; clustering samples there (as real solvers do) keeps
    # the fit -- and hence the coefficient magnitudes -- well-conditioned.
    j = np.arange(1, N + 1)
    x = np.cos((2 * j - 1) * np.pi / (2 * N))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x


def chebyshev_coeffs(theta, gap, D):
    m1 = theta - gap
    m2 = theta + gap
    N = 2 * D + 80
    xs_lo = cheb_nodes(0.0, m1, N)
    xs_hi = cheb_nodes(m2, 1.0, N)
    xs = np.concatenate([xs_lo, xs_hi])
    ys = (xs > theta).astype(float)
    u = 2 * xs - 1
    c = C.chebfit(u, ys, D)
    return c


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
    n, theta, eps = read_input()
    c = chebyshev_coeffs(theta, WORST_GAP, D_TRIV)
    emit_program(c)


if __name__ == "__main__":
    main()
