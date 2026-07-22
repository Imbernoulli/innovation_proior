#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for nodal-line-gardening.

Feasibility: the participant artifact must be exactly N*N whitespace-separated
integer tokens (row-major mass loads), each in [0, cap], all finite, summing to
<= budget. Any violation -> "Ratio: 0.0".

Score: solve the generalized eigenproblem K v = omega^2 (M_base + load) v for the
clamped N x N membrane, take the eigenvector ranked k-th by omega^2 ascending,
normalize it to unit peak amplitude, and reward target cells whose amplitude
sits below tau=0.2 of that peak (a "nodal line" garden). Normalized against the
checker's own uniform-spread baseline construction.
"""
import sys
import math
import numpy as np

EPS = 1e-6
TAU = 0.2


def build_K(N):
    Nc = N * N
    K = np.zeros((Nc, Nc))
    for r in range(N):
        for c in range(N):
            i = r * N + c
            K[i, i] = 4.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    K[i, rr * N + cc] = -1.0
    return K


def score_for_loads(K, N, k, loads):
    Nc = N * N
    m = np.array([1.0 + i * EPS + loads[i] for i in range(Nc)])
    s = 1.0 / np.sqrt(m)
    L = (s[:, None] * K) * s[None, :]
    w, U = np.linalg.eigh(L)
    u = U[:, k - 1]
    v = s * u
    maxabs = float(np.max(np.abs(v)))
    if maxabs < 1e-12:
        return None
    vhat = v / maxabs
    return vhat


def gardening_score(vhat, targets):
    tot = 0.0
    for c in targets:
        a = abs(vhat[c])
        tot += max(0.0, 1.0 - a / TAU)
    return tot


def uniform_baseline(N, cap, budget):
    Nc = N * N
    base = budget // Nc
    rem = budget - base * Nc
    loads = [min(cap, base) for _ in range(Nc)]
    for i in range(rem):
        loads[i] = min(cap, loads[i] + 1)
    return loads


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    p = 0
    N = int(toks[p]); p += 1
    k = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1
    budget = int(toks[p]); p += 1
    t = int(toks[p]); p += 1
    targets = [int(toks[p + i]) for i in range(t)]
    p += t
    Nc = N * N

    with open(outf) as f:
        raw = f.read().split()

    if len(raw) != Nc:
        fail(f"expected {Nc} tokens, got {len(raw)}")

    loads = []
    for tok in raw:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token '{tok}'")
        if not math.isfinite(v):
            fail("non-finite token")
        loads.append(v)

    for i, v in enumerate(loads):
        if v < 0 or v > cap:
            fail(f"load[{i}]={v} out of range [0,{cap}]")
    if sum(loads) > budget:
        fail(f"sum(loads)={sum(loads)} > budget {budget}")

    K = build_K(N)
    vhat = score_for_loads(K, N, k, loads)
    if vhat is None:
        fail("degenerate eigenvector (unreachable in a valid instance)")
    F = gardening_score(vhat, targets)

    base_loads = uniform_baseline(N, cap, budget)
    vhat_b = score_for_loads(K, N, k, base_loads)
    B = gardening_score(vhat_b, targets) if vhat_b is not None else 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"F={F:.6f} B={B:.6f}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
