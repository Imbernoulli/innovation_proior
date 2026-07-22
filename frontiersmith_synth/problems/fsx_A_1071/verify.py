#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the weighing-design problem.

Reads n,k from <in>; reads the participant's n x n {-1,0,1} matrix from <out>;
validates feasibility strictly; scores the defect F = sum_{i!=j} |(WW^T)_{ij}|
against the checker's own sliding-window baseline B (minimization, raised cap
so a perfect F=0 still leaves headroom)."""
import sys


def bail(msg):
    print("INVALID:", msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        bail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        toks = f.read().split()
    if len(toks) < 2:
        bail("bad input file")
    n, k = int(toks[0]), int(toks[1])
    if not (4 <= n <= 200 and 1 <= k < n):
        bail("bad instance parameters")

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception:
        bail("cannot read output")

    lines = out_text.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != n:
        bail(f"expected {n} lines, got {len(lines)}")

    W = [[0] * n for _ in range(n)]
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != n:
            bail(f"row {i}: expected {n} tokens, got {len(parts)}")
        nz = 0
        for j, tok in enumerate(parts):
            try:
                v = int(tok)
            except ValueError:
                bail(f"row {i} col {j}: non-integer token {tok!r}")
            if v not in (-1, 0, 1):
                bail(f"row {i} col {j}: value {v} not in {{-1,0,1}}")
            W[i][j] = v
            if v != 0:
                nz += 1
        if nz != k:
            bail(f"row {i}: has {nz} nonzero entries, need exactly {k}")

    # ---- objective: defect of W W^T vs k*I ----
    import numpy as np
    Wm = np.array(W, dtype=np.int64)
    G = Wm @ Wm.T
    off = G.copy()
    np.fill_diagonal(off, 0)
    F = int(np.abs(off).sum())

    # ---- checker's own baseline: sliding-window circulant support, all +1 ----
    Bm = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for t in range(k):
            Bm[i, (i + t) % n] = 1
    Gb = Bm @ Bm.T
    offb = Gb.copy()
    np.fill_diagonal(offb, 0)
    B = int(np.abs(offb).sum())
    if B <= 0:
        bail("degenerate baseline (should not happen for 1<=k<n)")

    sc = min(880.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"defect F={F} baseline B={B}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
