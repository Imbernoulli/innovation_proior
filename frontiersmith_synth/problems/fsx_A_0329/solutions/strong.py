# TIER: strong
# Best rank-1 Korobov lattice in 3D. Sweep the generator g over 1..n-1 for the
# point set  p_i = ((i*[1, g, g^2 mod n] + 0.5)/n mod 1)  and keep the generator
# whose EXACT 3D star discrepancy is smallest. This real search reliably beats any
# single fixed construction (diagonal or Hammersley) and behaves differently per test.
import sys
import numpy as np


def star_discrepancy(pts, n):
    n_pts, d = pts.shape
    axes = []
    for j in range(d):
        vals = np.unique(np.concatenate([pts[:, j], np.array([1.0])]))
        axes.append(vals)
    grids = np.meshgrid(*axes, indexing="ij")
    corners = np.stack([g.ravel() for g in grids], axis=1)
    V = np.prod(corners, axis=1)
    best = 0.0
    C = corners.shape[0]
    chunk = max(1, 3_000_000 // max(1, n_pts * d))
    for s in range(0, C, chunk):
        cc = corners[s:s + chunk]
        Nc = np.all(cc[:, None, :] >= pts[None, :, :], axis=2).sum(axis=1)
        No = np.all(cc[:, None, :] > pts[None, :, :], axis=2).sum(axis=1)
        v = V[s:s + chunk]
        loc = float(np.maximum(Nc / n - v, v - No / n).max())
        if loc > best:
            best = loc
    return best


def lattice(n, g):
    i = np.arange(n)
    g2 = (g * g) % n
    x = ((i * 1) % n + 0.5) / n
    y = ((i * g) % n + 0.5) / n
    z = ((i * g2) % n + 0.5) / n
    return np.stack([x, y, z], axis=1)


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    best_g = 1
    best_d = None
    for g in range(1, n):
        d = star_discrepancy(lattice(n, g), n)
        if best_d is None or d < best_d:
            best_d = d
            best_g = g
    pts = lattice(n, best_g)
    out = ["%.10f %.10f %.10f" % (p[0], p[1], p[2]) for p in pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
