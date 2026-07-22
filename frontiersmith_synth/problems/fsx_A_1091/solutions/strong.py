# TIER: strong
# The insight: fooling a spectral pipeline is a SUBSPACE-ROTATION problem.
# Degrees never change under 2-swaps, so the normalized-Laplacian
# perturbation is dL = -D^-1/2 dA D^-1/2 with FIXED D. By first-order
# perturbation theory (Davis-Kahan), the drift of informative eigenvector
# v_i under perturbation E is
#     dv_i = sum_{j!=i} v_j (v_j^T E v_i) / (lam_i - lam_j).
# Pick a decoy partition (phantom wards: phantom ward c = second half of
# true ward c glued to the first half of true ward c+1, cyclically -- a
# genuine partition) with orthonormal centered basis t_1..t_{k-1}.
# To pull v_i toward t_i we must MAXIMIZE  w_i^T E v_i  with
#     w_i = sum_{j!=i} (t_i . v_j)/(lam_i - lam_j) v_j.
# For one 2-swap removing (a,b),(c,d) and adding (a,d),(b,c) the gain is
# exactly separable:
#     g = (W'_a - W'_c)(V'_b - V'_d) + (W'_b - W'_d)(V'_a - V'_c),
# with W' = W/sqrt(deg), V' = V/sqrt(deg). Summed over informative i and
# evaluated over candidate boundary-straddling edges, applied greedily with
# the spectrum recomputed each round. Coherent alignment rotates the
# eigenvectors by O(B) where unstructured noise rotates them by O(sqrt(B)).
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


def read_instance():
    tok = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        t = int(tok[pos]); pos += 1
        return t

    n = nxt(); m = nxt(); k = nxt(); budget = nxt()
    labels = [nxt() for _ in range(n)]
    edges = [(nxt(), nxt()) for _ in range(m)]
    edges = sorted({(min(a, b), max(a, b)) for (a, b) in edges if a != b})
    return n, k, budget, labels, edges


def decoy(n, k, labels):
    """Phantom wards: D_c = second half of true ward c U first half of true
    ward (c+1) mod k. Returns (indicator matrix T (n x k), orthonormal
    centered basis Q (n x k-1), decoy group id per vertex)."""
    members = []
    for c in range(k):
        members.append(sorted([v for v in range(n) if labels[v] == c]))
    T = np.zeros((n, k), dtype=np.float64)
    grp = np.zeros(n, dtype=np.int64)
    for c in range(k):
        sc = members[c]
        nc = members[(c + 1) % k]
        part = set(sc[len(sc) // 2:]) | set(nc[: len(nc) // 2])
        for v in sorted(part):
            T[v, c] = 1.0
            grp[v] = c
    Tc = T - T.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(Tc)         # deterministic given Tc
    return T, Q[:, :k - 1], grp


def laplacian(n, edge_set):
    A = np.zeros((n, n), dtype=np.float64)
    for (a, b) in edge_set:
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    L = np.eye(n) - A * dinv[:, None] * dinv[None, :]
    return L, dinv


def main():
    n, k, budget, labels, edges = read_instance()
    es = set(edges)
    _, Q, grp = decoy(n, k, labels)
    lab = np.asarray(labels)

    swaps = []
    R = 6
    CMAX = 200
    JSPEC = None  # use the full spectrum in the w_i sum

    for rnd in range(R):
        remaining = budget - len(swaps)
        if remaining <= 0:
            break
        per_round = -(-remaining // (R - rnd))  # ceil
        L, dinv = laplacian(n, es)
        lam, V = np.linalg.eigh(L)
        J = n if JSPEC is None else min(n, JSPEC)
        W = np.zeros((n, k - 1), dtype=np.float64)
        VJ = V[:, :J]
        lamJ = lam[:J]
        for ii in range(1, k):
            t = Q[:, ii - 1]
            coeff = VJ.T @ t
            denom = lam[ii] - lamJ
            ok = np.abs(denom) > 1e-9
            if ii < J:
                ok[ii] = False
            c = np.where(ok, coeff / np.where(ok, denom, 1.0), 0.0)
            W[:, ii - 1] = VJ @ c
        Vp = V[:, 1:k] * dinv[:, None]     # primed informative eigenvectors
        Wp = W * dinv[:, None]

        # candidate edges: within-ward edges that straddle the decoy boundary
        cand = []
        for (a, b) in sorted(es):
            if labels[a] == labels[b] and grp[a] != grp[b]:
                pa = Wp[a] - Wp[b]
                qa = Vp[a] - Vp[b]
                cand.append((-float(np.abs(pa * qa).sum()), a, b))
        cand.sort()
        cand = [(a, b) for (_, a, b) in cand[:CMAX]]
        if len(cand) < 2:
            break
        C = len(cand)
        idx_a = [e[0] for e in cand]
        idx_b = [e[1] for e in cand]
        Wa = Wp[idx_a]; Wb = Wp[idx_b]
        Va = Vp[idx_a]; Vb = Vp[idx_b]
        # pair (i,j), i<j, remove (a_i,b_i) and (a_j,b_j):
        #   orient 1: add (a_i,b_j),(b_i,a_j)
        #     g = (Wa_i-Wa_j)(Vb_i-Vb_j) + (Wb_i-Wb_j)(Va_i-Va_j)
        #   orient 2: add (a_i,a_j),(b_i,b_j)
        #     g = (Wa_i-Wb_j)(Vb_i-Va_j) + (Wb_i-Wa_j)(Va_i-Vb_j)
        gains = []
        for i in range(C - 1):
            dWa = Wa[i][None, :] - Wa[i + 1:]
            dWb = Wb[i][None, :] - Wb[i + 1:]
            dVa = Va[i][None, :] - Va[i + 1:]
            dVb = Vb[i][None, :] - Vb[i + 1:]
            g1 = (dWa * dVb).sum(1) + (dWb * dVa).sum(1)
            dWa2 = Wa[i][None, :] - Wb[i + 1:]
            dWb2 = Wb[i][None, :] - Wa[i + 1:]
            g2 = (dWa2 * dVa).sum(1) + (dWb2 * dVb).sum(1)
            base = i + 1
            for off in range(g1.shape[0]):
                gains.append((float(g1[off]), i, base + off, 1))
                gains.append((float(g2[off]), i, base + off, 2))
        gains.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        applied = 0
        for (g, i, j, orient) in gains:
            if applied >= per_round:
                break
            if g <= 1e-12:
                break
            (a, b) = cand[i]
            (c, d) = cand[j]
            if len({a, b, c, d}) != 4:
                continue
            e_ab = (a, b) if a < b else (b, a)
            e_cd = (c, d) if c < d else (d, c)
            if e_ab not in es or e_cd not in es:
                continue
            if orient == 1:
                p, q = (a, d), (b, c)
            else:
                p, q = (a, c), (b, d)
            e1 = (min(p), max(p))
            e2 = (min(q), max(q))
            if e1 == e2 or e1 in es or e2 in es:
                continue
            es.discard(e_ab)
            es.discard(e_cd)
            es.add(e1)
            es.add(e2)
            if orient == 1:
                swaps.append((a, b, c, d))   # add (a,d),(b,c)
            else:
                swaps.append((a, b, d, c))   # add (a,c),(b,d)
            applied += 1
        if applied == 0:
            break

    out = [str(len(swaps))]
    out += ["%d %d %d %d" % s for s in swaps]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
