# TIER: strong
"""Minimum-enclosing-simplex insight: if no pixel is pure, the true endmembers lie OUTSIDE
the observed data cloud, so picking an extreme observed pixel (greedy) systematically
undershoots. Instead: even without pure pixels, the data still densely hugs each EDGE of
the true simplex (two-material mixtures). Assign points to the 3 angular sectors around
rough greedy corners, keep the boundary-hugging half of each sector (top-quantile by radial
distance from the centroid, after trimming a bit of each sector's ends to dodge any
corner-truncation contamination), fit a total-least-squares line through each edge's
surviving points, and INTERSECT consecutive edge-lines. Line intersections can and do land
outside the convex hull of the sample -- the inflate-outward move -- recovering vertices no
single observed pixel is anywhere near."""
import sys, math
import numpy as np
from scipy.optimize import nnls


def pick_extreme(Z, K):
    N = Z.shape[0]
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    i0, j0 = np.unravel_index(np.argmax(d), d.shape)
    chosen = [int(i0), int(j0)]
    while len(chosen) < K:
        best_idx, best_vol = None, -1.0
        for cand in range(N):
            if cand in chosen:
                continue
            pts = Z[chosen + [cand]]
            base = pts[0]
            edges = pts[1:] - base
            if edges.shape[0] == edges.shape[1]:
                vol = abs(np.linalg.det(edges))
            else:
                vol = float(np.linalg.norm(edges))
            if vol > best_vol:
                best_vol = vol
                best_idx = cand
        chosen.append(best_idx)
    return chosen


def nnls_abundance(Y, M_hat):
    K = M_hat.shape[0]
    N = Y.shape[0]
    A = np.zeros((N, K))
    BIG = 100.0
    Maug = np.vstack([M_hat.T, BIG * np.ones((1, K))])
    for j in range(N):
        yaug = np.concatenate([Y[j], [BIG]])
        sol, _ = nnls(Maug, yaug)
        s = sol.sum()
        A[j] = sol / s if s > 1e-9 else np.full(K, 1.0 / K)
    return A


def strong_solve(Y, K, trim_frac=0.05, quantile=0.5):
    mean_y = Y.mean(axis=0)
    Yc = Y - mean_y
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    basis = Vt[:K - 1]
    Z = Yc @ basis.T
    dim = K - 1

    chosen = pick_extreme(Z, K)
    G = Z[chosen].astype(float)

    if dim != 2:
        M_hat = (G @ basis) + mean_y
        return M_hat, nnls_abundance(Y, M_hat)

    centroid = Z.mean(axis=0)
    pt_ang = np.arctan2(Z[:, 1] - centroid[1], Z[:, 0] - centroid[0])
    radial = np.linalg.norm(Z - centroid, axis=1)
    g_ang = np.arctan2(G[:, 1] - centroid[1], G[:, 0] - centroid[0])
    order = np.argsort(g_ang)
    g_ang_sorted = g_ang[order]

    def ang_diff(a, b):
        return (a - b) % (2 * math.pi)

    lines = []
    for i in range(K):
        a0 = g_ang_sorted[i]
        a1 = g_ang_sorted[(i + 1) % K]
        span = ang_diff(a1, a0)
        rel = ang_diff(pt_ang, a0)
        in_arc = rel < span
        lo, hi = trim_frac * span, (1.0 - trim_frac) * span
        mask = in_arc & (rel >= lo) & (rel <= hi)
        pts, r = Z[mask], radial[mask]
        if pts.shape[0] >= 4:
            thresh = np.percentile(r, 100.0 * (1.0 - quantile))
            keep = r >= thresh
            if keep.sum() >= 2:
                pts = pts[keep]
        if pts.shape[0] < 2:
            p_on = G[order[i]]
            d_ = G[order[(i + 1) % K]] - G[order[i]]
        else:
            p_on = pts.mean(axis=0)
            pc = pts - p_on
            _, _, Vt2 = np.linalg.svd(pc, full_matrices=False)
            d_ = Vt2[0]
        lines.append((p_on, d_))

    def intersect(l1, l2):
        p1, d1 = l1
        p2, d2 = l2
        Amat = np.array([d1, -d2]).T
        rhs = p2 - p1
        try:
            ts = np.linalg.solve(Amat, rhs)
        except np.linalg.LinAlgError:
            return (p1 + p2) / 2.0
        return p1 + ts[0] * d1

    V = np.zeros((K, dim))
    for i in range(K):
        V[order[i]] = intersect(lines[(i - 1) % K], lines[i])

    M_hat = V @ basis + mean_y
    return M_hat, nnls_abundance(Y, M_hat)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    _t = int(next(it))
    R = int(next(it)); K = int(next(it)); N = int(next(it))
    Y = np.array([[float(next(it)) for _ in range(R)] for _ in range(N)])

    M_hat, A_hat = strong_solve(Y, K)
    # physical spectra can't be negative -- the extrapolated line intersection can overshoot
    # slightly past zero in a band; clip (a real sensor / physics constraint, not cheating).
    M_hat = np.clip(M_hat, 0.0, None)
    # re-fit abundances against the clipped endmembers so output stays self-consistent
    A_hat = nnls_abundance(Y, M_hat)

    out = []
    for k in range(K):
        out.append(" ".join("%.6f" % v for v in M_hat[k]))
    for j in range(N):
        out.append(" ".join("%.6f" % v for v in A_hat[j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
