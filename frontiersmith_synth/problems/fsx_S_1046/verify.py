#!/usr/bin/env python3
# Deterministic checker for "Bell-Sounding: Worst-Case Crack Probes" (fsx_S_1046, format C).
# CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys
import numpy as np

STRADDLE_FRAC = 5  # baseline flanking span = max(1, b // STRADDLE_FRAC)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def build_mesh(b, L, g_r, g_c, g_core):
    """Cylindrical bell mesh: layers 0..L-1 (layer 0 = rim/boundary, ring of b
    nodes), each ring node radially wired to the next layer's ring node, each
    ring circumferentially wired to its neighbours, and the innermost ring
    (layer L-1) all wired to one extra CORE (crown) node. Returns
    (n, core_index, W) with W the symmetric conductance (weight) matrix."""
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
    """Scale the 4 edges incident to interior node (layer,pos) by alpha (a
    localized conductance drop -- the hidden crack)."""
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
    """b x b linear map: current pattern on the rim (zero-sum) -> rim
    potentials, gauge-fixed by v[core] = 0."""
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


def score_of(probes, Ms):
    """min over defects of max over probes of p^T M p (worst-case detectability)."""
    if not probes or not Ms:
        return 0.0
    vals = []
    for M in Ms:
        best = max(float(p @ M @ p) for p in probes)
        vals.append(best)
    return min(vals)


def baseline_probes(defects, b, q, I_max):
    """Checker's own simple reference: a TIGHT flanking dipole straddling
    (electrodes at pos-s and pos+s) each of the first q listed defects,
    cycling through the defect list if q > m."""
    span = max(1, b // STRADDLE_FRAC)
    m = len(defects)
    probes = []
    for j in range(q):
        pos, _layer = defects[j % m]
        a = (pos - span) % b
        c = (pos + span) % b
        if a == c:
            c = (a + 1) % b
        p = np.zeros(b)
        p[a] = I_max
        p[c] = -I_max
        probes.append(p)
    return probes


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    ptr = 0

    def nxt():
        nonlocal ptr
        v = toks[ptr]
        ptr += 1
        return v

    b = int(nxt()); L = int(nxt()); m = int(nxt()); q = int(nxt()); I_max = int(nxt())
    g_r = float(nxt()); g_c = float(nxt()); g_core = float(nxt())
    alpha = float(nxt())
    defects = []
    for _ in range(m):
        pos = int(nxt()); layer = int(nxt())
        defects.append((pos, layer))

    # --- read participant output strictly: EXACTLY q non-blank lines, each
    # with EXACTLY b whitespace-separated tokens (matches the stated format). ---
    try:
        with open(out_path) as f:
            raw_lines = f.read().splitlines()
    except OSError:
        fail("cannot read output")

    out_lines = [ln for ln in raw_lines if ln.strip() != ""]
    if len(out_lines) != q:
        fail(f"expected {q} non-blank output lines (one per probe), got {len(out_lines)}")

    probes_int = []
    for i in range(q):
        row_toks = out_lines[i].split()
        if len(row_toks) != b:
            fail(f"probe {i+1} has {len(row_toks)} entries, expected {b}")
        row = []
        for j, tok in enumerate(row_toks):
            try:
                if any(c in tok for c in ('.', 'e', 'E', 'n', 'N', 'inf', 'Inf', 'nan', 'NaN')):
                    raise ValueError
                val = int(tok)
            except ValueError:
                fail(f"non-integer token '{tok}' at probe {i+1} entry {j+1}")
            if val < -I_max or val > I_max:
                fail(f"entry {val} out of range [-{I_max},{I_max}] at probe {i+1} entry {j+1}")
            row.append(val)
        if sum(row) != 0:
            fail(f"probe {i+1} entries sum to {sum(row)}, must sum to 0 (current conservation)")
        probes_int.append(row)

    probes = [np.array(r, dtype=float) for r in probes_int]

    # --- physics ---
    n, core, W = build_mesh(b, L, g_r, g_c, g_core)
    Lap = laplacian(W)
    R0 = response_matrix(Lap, core, b)

    Ms = []
    for pos, layer in defects:
        Wd = apply_defect(W, pos, layer, alpha, b)
        Lapd = laplacian(Wd)
        Rd = response_matrix(Lapd, core, b)
        J = Rd - R0
        Ms.append(J.T @ J)

    F = score_of(probes, Ms)
    if not np.isfinite(F):
        fail("non-finite objective")

    Bp = baseline_probes(defects, b, q, I_max)
    B = score_of(Bp, Ms)
    if not np.isfinite(B) or B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.8f B=%.8f Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
