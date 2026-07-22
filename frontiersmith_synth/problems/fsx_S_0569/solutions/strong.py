# TIER: strong
# Insight: exact inversion of an ill-conditioned / non-minimum-phase cascade explodes.
# Instead solve the REGULARIZED shaping problem -- Tikhonov normal equations
#   (H^T H + lam I) x = H^T y*
# where H is the (M x N) convolution matrix of the combined cascade.  This trades match
# against input energy (exactly the objective's structure) and stays bounded near the
# unit-circle zeros.  Then project onto the amplitude box.
#
# (Headroom left above this: the true optimum is the BOX-constrained QP, which a
#  projected-gradient / active-set solver can push further than a single clip.)
import sys
import numpy as np


def main():
    tk = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = tk[p]
        p += 1
        return v

    N = int(nxt())
    A = float(nxt())
    lam = float(nxt())
    K = int(nxt())
    filters = []
    for _ in range(K):
        Li = int(nxt())
        filters.append(np.array([float(nxt()) for _ in range(Li)], dtype=np.float64))
    M = int(nxt())
    ystar = np.array([float(nxt()) for _ in range(M)], dtype=np.float64)

    h = np.array([1.0])
    for f in filters:
        h = np.convolve(h, f)
    L = len(h)

    # convolution matrix H : (M x N), column j is h shifted down by j
    H = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        H[j:j + L, j] = h

    G = H.T @ H + lam * np.eye(N)
    b = H.T @ ystar
    x = np.linalg.solve(G, b)

    x = np.clip(x, -A, A)
    sys.stdout.write(" ".join("%.12g" % v for v in x) + "\n")


if __name__ == "__main__":
    main()
