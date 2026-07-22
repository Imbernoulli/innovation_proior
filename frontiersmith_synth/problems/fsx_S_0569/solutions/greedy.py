# TIER: greedy
# The "obvious" move: just make the output hit the target -- solve the UNREGULARIZED
# least-squares deconvolution  min_x ||H x - y*||^2  (ignore the energy term, ignore the
# conditioning), then clip to the amplitude bound.  Because the cascade has near-unit-circle
# and non-minimum-phase zeros, H^T H is ill-conditioned, so the small-singular-value
# directions get amplified into a huge-amplitude x; the clip to [-A,A] then wrecks the match.
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

    H = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        H[j:j + L, j] = h

    # exact (unregularized) least-squares deconvolution -- no energy penalty, no damping
    x, *_ = np.linalg.lstsq(H, ystar, rcond=1e-12)
    x = np.clip(x, -A, A)
    sys.stdout.write(" ".join("%.10g" % v for v in x) + "\n")


if __name__ == "__main__":
    main()
