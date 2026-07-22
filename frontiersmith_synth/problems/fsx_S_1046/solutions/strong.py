# TIER: strong
# Strong: exploits the DtN-perturbation spectral structure directly.
#
# For a fixed defect k, the checker's own per-probe score is a quadratic form
# p^T M_k p on the zero-sum current subspace, where M_k = J_k^T J_k and J_k is
# the (linear) difference between the defect's and the pristine mesh's
# boundary response operators. The single current pattern that most strongly
# reveals defect k is therefore (approximately, after rounding to the
# integer/magnitude-bounded feasible set) the LEADING EIGENVECTOR of M_k
# restricted to that subspace -- i.e. detection power is the leading
# eigenvalue of a DtN perturbation operator, exactly the family's hook.
#
# Because the objective is a MIN over the whole defect family of the BEST
# probe for that defect, probe design becomes a covering problem over these
# per-defect leading eigen-directions: repeatedly aim a probe at whichever
# defect is currently worst-covered, using that defect's own eigen-focused
# pattern (falling back to a spread dipole only if it happens to score no
# worse). This both focuses current at the right depth AND diversifies
# across the family -- the two things a 2-electrode textbook probe cannot
# jointly do.
import sys
import numpy as np


def build_mesh(b, L, g_r, g_c, g_core):
    n = L * b + 1
    core = L * b

    def nid(l, i):
        return l * b + (i % b)

    W = np.zeros((n, n))
    for l in range(L - 1):
        for i in range(b):
            u, v = nid(l, i), nid(l + 1, i)
            W[u, v] += g_r
            W[v, u] += g_r
    for l in range(L):
        for i in range(b):
            u, v = nid(l, i), nid(l, i + 1)
            W[u, v] += g_c
            W[v, u] += g_c
    for i in range(b):
        u, v = nid(L - 1, i), core
        W[u, v] += g_core
        W[v, u] += g_core
    return n, core, W


def apply_defect(W, pos, layer, alpha, b):
    W2 = W.copy()

    def nid(l, i):
        return l * b + (i % b)

    node = nid(layer, pos)
    targets = [(nid(layer - 1, pos), node), (node, nid(layer + 1, pos)),
               (nid(layer, pos - 1), node), (node, nid(layer, pos + 1))]
    for (u, v) in targets:
        W2[u, v] *= alpha
        W2[v, u] *= alpha
    return W2


def laplacian(W):
    n = W.shape[0]
    Lap = -W.copy()
    for i in range(n):
        Lap[i, i] = W[i].sum()
    return Lap


def response_matrix(Lap, core, b):
    n = Lap.shape[0]
    idx = [i for i in range(n) if i != core]
    Lred = Lap[np.ix_(idx, idx)]
    Linv = np.linalg.inv(Lred)
    R = np.zeros((b, b))
    for j in range(b):
        Ivec = np.zeros(n)
        Ivec[j] = 1.0
        v_red = Linv @ Ivec[idx]
        v = np.zeros(n)
        v[idx] = v_red
        R[:, j] = v[:b]
    return R


def zero_sum_basis(b):
    ones = np.ones((1, b))
    _U, _S, Vt = np.linalg.svd(ones, full_matrices=True)
    return Vt[1:].T  # b x (b-1), orthonormal basis of {x: sum x = 0}


def leading_eigvec(M, P):
    Mp = P.T @ M @ P
    w, v = np.linalg.eigh(Mp)
    return P @ v[:, -1], w[-1]


def round_to_feasible(vec, b, I_max):
    if np.max(np.abs(vec)) < 1e-12:
        return np.zeros(b)
    scaled = vec / np.max(np.abs(vec)) * I_max
    rounded = np.clip(np.round(scaled), -I_max, I_max).astype(float)
    s = int(rounded.sum())
    order = np.argsort(-np.abs(rounded))
    i = 0
    while s != 0 and i < 4000:
        j = order[i % b]
        if s > 0 and rounded[j] > -I_max:
            rounded[j] -= 1; s -= 1
        elif s < 0 and rounded[j] < I_max:
            rounded[j] += 1; s += 1
        i += 1
    return rounded


def main():
    data = sys.stdin.read().split()
    ptr = 0

    def nxt():
        nonlocal ptr
        v = data[ptr]
        ptr += 1
        return v

    b = int(nxt()); L = int(nxt()); m = int(nxt()); q = int(nxt()); I_max = int(nxt())
    g_r = float(nxt()); g_c = float(nxt()); g_core = float(nxt())
    alpha = float(nxt())
    defects = []
    for _ in range(m):
        pos = int(nxt()); layer = int(nxt())
        defects.append((pos, layer))

    n, core, W = build_mesh(b, L, g_r, g_c, g_core)
    Lap = laplacian(W)
    R0 = response_matrix(Lap, core, b)

    P = zero_sum_basis(b)
    half = b // 2

    best_probes = []
    for pos, layer in defects:
        Wd = apply_defect(W, pos, layer, alpha, b)
        Lapd = laplacian(Wd)
        Rd = response_matrix(Lapd, core, b)
        J = Rd - R0
        M = J.T @ J

        vec, _ = leading_eigvec(M, P)
        pe = round_to_feasible(vec, b, I_max)

        a = pos
        c = (pos + half) % b
        if a == c:
            c = (a + 1) % b
        pd = np.zeros(b)
        pd[a] = I_max
        pd[c] = -I_max

        val_e = float(pe @ M @ pe)
        val_d = float(pd @ M @ pd)
        best_probes.append((pe if val_e >= val_d else pd, M))

    covered = [0.0] * m
    chosen = []
    for _ in range(q):
        w = min(range(m), key=lambda i: covered[i])
        cand, _ = best_probes[w]
        chosen.append(cand)
        for i, (_p, M) in enumerate(best_probes):
            val = float(cand @ M @ cand)
            if val > covered[i]:
                covered[i] = val

    lines = []
    for p in chosen:
        row = [int(round(x)) for x in p]
        s = sum(row)
        if s != 0:
            # safety net: nudge the largest-magnitude entry (should not
            # trigger in practice since round_to_feasible already balances)
            idx = max(range(b), key=lambda i: abs(row[i]))
            row[idx] -= s
            row[idx] = max(-I_max, min(I_max, row[idx]))
        lines.append(" ".join(map(str, row)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
