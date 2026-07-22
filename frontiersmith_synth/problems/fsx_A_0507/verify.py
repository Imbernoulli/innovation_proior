import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def is_wall(r, c, H, W, vents_set):
    # entire outer border is an insulating wall EXCEPT the vent cells
    if r == 0 or r == H - 1 or c == 0 or c == W - 1:
        return (r, c) not in vents_set
    return False


def solve_maxtemp(H, W, KHI, vents_set, sources, upgraded):
    """Exact steady state of the Dirichlet heat problem on the open (interior) cells.
    Edge conductance = harmonic mean of the two cell conductivities (series resistance).
    Vents are grounded at T=0; walls carry no flux. Returns max interior temperature."""
    idx = {}
    unk = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            idx[(r, c)] = len(unk)
            unk.append((r, c))
    n = len(unk)
    if n == 0:
        return 0.0
    P = np.zeros(n)
    for (r, c, p) in sources:
        P[idx[(r, c)]] += p

    def kof(rc):
        return KHI if rc in upgraded else 1.0

    def g(a, b):
        return 2.0 * a * b / (a + b)

    rows = []
    cols = []
    data = []
    for u, (r, c) in enumerate(unk):
        diag = 0.0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if is_wall(nr, nc, H, W, vents_set):
                continue
            gg = g(kof((r, c)), kof((nr, nc)))
            diag += gg
            if (nr, nc) in idx:  # interior neighbor -> off-diagonal
                rows.append(u)
                cols.append(idx[(nr, nc)])
                data.append(-gg)
            # else neighbor is a grounded vent: only adds to the diagonal
        rows.append(u)
        cols.append(u)
        data.append(diag)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    T = spla.spsolve(A, P)
    if not np.all(np.isfinite(T)):
        return float("inf")
    return float(T.max())


def read_ints(path):
    with open(path) as f:
        toks = f.read().split()
    return toks


def main():
    inp = read_ints(sys.argv[1])
    # ---- parse instance ----
    try:
        it = iter(inp)
        H = int(next(it)); W = int(next(it)); KHI = int(next(it)); K = int(next(it))
        S = int(next(it))
        sources = []
        for _ in range(S):
            r = int(next(it)); c = int(next(it)); p = int(next(it))
            sources.append((r, c, p))
        NV = int(next(it))
        vents = []
        for _ in range(NV):
            r = int(next(it)); c = int(next(it))
            vents.append((r, c))
    except Exception:
        fail("bad instance")
    vents_set = set(vents)

    # ---- parse participant output (strict, bounded) ----
    try:
        with open(sys.argv[2]) as f:
            otoks = f.read().split()
    except Exception:
        fail("no output")
    if len(otoks) == 0:
        fail("empty output")
    # reject any non-integer / non-finite token up front
    for t in otoks:
        if t.lower() in ("nan", "inf", "-inf", "+inf", "infinity"):
            fail("non-finite token")
    try:
        kk = int(otoks[0])
    except Exception:
        fail("bad count token")
    if kk < 0 or kk > K:
        fail("count out of range")
    if len(otoks) != 1 + 2 * kk:
        fail("token count mismatch")
    upgraded = set()
    for j in range(kk):
        try:
            r = int(otoks[1 + 2 * j]); c = int(otoks[2 + 2 * j])
        except Exception:
            fail("bad coordinate")
        if not (1 <= r <= H - 2 and 1 <= c <= W - 2):
            fail("cell (%d,%d) is not an open interior cell" % (r, c))
        if (r, c) in vents_set:
            fail("cannot upgrade a vent")
        if (r, c) in upgraded:
            fail("duplicate cell (%d,%d)" % (r, c))
        upgraded.add((r, c))

    # ---- baseline B = do-nothing max temperature (checker's own trivial construction) ----
    B = solve_maxtemp(H, W, KHI, vents_set, sources, set())
    if not math.isfinite(B) or B <= 0:
        fail("degenerate instance")

    # ---- participant objective F ----
    F = solve_maxtemp(H, W, KHI, vents_set, sources, upgraded)
    if not math.isfinite(F) or F <= 0:
        fail("degenerate solution temperature")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("B=%.4f F=%.4f Ratio: %.6f" % (B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
