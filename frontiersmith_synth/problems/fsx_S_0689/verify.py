#!/usr/bin/env python3
# Deterministic checker for "Mirror Relay: Tuning a Spin Chain to Teleport a Pulse Intact"
# (format C, maximize continuous-time quantum-walk transfer fidelity).
# CLI: python3 verify.py <in> <out> <ans>   (ans is an empty placeholder, ignored).
# Prints "... Ratio: <r>" with r in [0,1]; ANY feasibility violation -> "Ratio: 0.0".
import sys, math
import numpy as np

BSCALE = 1.4   # safety margin baked into the internal baseline so `strong` never saturates


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def build_matrix(N, defects, couplings):
    A = np.zeros((N, N), dtype=np.float64)
    for s, v in defects.items():
        A[s - 1, s - 1] = v
    for e, v in couplings.items():
        A[e - 1, e] = v
        A[e, e - 1] = v
    return A


def fidelity(N, defects, couplings, T):
    """|<N| exp(-i A T) |1>|, computed via a real symmetric eigendecomposition.
    Basis-independent (invariant to eigenvector sign / degenerate-subspace choice),
    numpy.linalg.eigh is deterministic for a fixed input matrix -> reproducible score."""
    A = build_matrix(N, defects, couplings)
    w, V = np.linalg.eigh(A)
    amp = np.sum(V[0, :] * V[N - 1, :] * np.exp(-1j * w * T))
    return abs(complex(amp))


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        it = iter(itoks)
        N = int(next(it))
        T = float(next(it))
        J_LO = float(next(it)); J_HI = float(next(it))
        D = int(next(it))
        defects = {}
        for _ in range(D):
            s = int(next(it)); v = float(next(it))
            defects[s] = v
        K = int(next(it))
        frozen = {}
        frozen_edges = set()
        for _ in range(K):
            e = int(next(it)); v = float(next(it))
            frozen[e] = v
            frozen_edges.add(e)
    except Exception:
        fail("bad input")

    if N < 2:
        fail("bad N")
    free_edges = sorted(e for e in range(1, N) if e not in frozen_edges)
    Fcount = len(free_edges)

    # ---- parse participant output ----
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not otoks:
        fail("empty output")
    try:
        k = int(otoks[0])
    except Exception:
        fail("bad count token")
    if k != Fcount:
        fail("expected count %d, got %d" % (Fcount, k))
    vals = otoks[1:1 + k]
    if len(vals) != k:
        fail("expected %d values, got %d" % (k, len(vals)))

    EPS = 1e-6
    couplings = dict(frozen)
    for edge_idx, tok in zip(free_edges, vals):
        try:
            x = float(tok)
        except Exception:
            fail("unparsable value %r" % tok)
        if not math.isfinite(x):
            fail("non-finite value %r" % tok)
        if x < J_LO - EPS or x > J_HI + EPS:
            fail("value %.6f out of bounds [%.6f,%.6f]" % (x, J_LO, J_HI))
        couplings[edge_idx] = x

    F = fidelity(N, defects, couplings, T)

    # ---- internal baseline B: uniform mid-bound coupling on all free edges, +40% safety margin ----
    mid = 0.5 * (J_LO + J_HI)
    triv_couplings = dict(frozen)
    for e in free_edges:
        triv_couplings[e] = mid
    B_raw = fidelity(N, defects, triv_couplings, T)
    B = BSCALE * max(1e-9, B_raw)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = max(0.0, min(1.0, sc / 1000.0))
    print("F=%.8f B_raw=%.8f Ratio: %.6f" % (F, B_raw, ratio))


if __name__ == "__main__":
    main()
