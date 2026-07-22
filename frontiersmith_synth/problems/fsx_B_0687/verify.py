#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for surge-roster-boundary-stagger.

Reads the instance (<in>), a roster of W shift-start hours (<out>), validates
feasibility, runs a deterministic discrete queue recursion for every published
surge profile, takes the worst (max) peak queue length over the sweep, and
scores against an internal baseline (a fully-clustered roster).
"""
import sys, math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    W = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    g = int(toks[idx]); idx += 1
    r = int(toks[idx]); idx += 1
    base = [int(toks[idx + i]) for i in range(24)]; idx += 24
    K = int(toks[idx]); idx += 1
    profiles = []
    for _ in range(K):
        sk = int(toks[idx]); dk = int(toks[idx + 1]); ak = int(toks[idx + 2])
        idx += 3
        profiles.append((sk, dk, ak))
    return W, L, C, g, r, base, K, profiles


def coverage(starts, L):
    cov = [0] * 24
    for s in starts:
        for k in range(L):
            cov[(s + k) % 24] += 1
    return cov


def worst_case_peak(starts, L, C, base, profiles, days=3):
    cov = coverage(starts, L)
    cap = [C * c for c in cov]
    worst = 0
    T = days * 24
    for (sk, dk, ak) in profiles:
        Q = 0
        peak = 0
        for t in range(T):
            h = t % 24
            surge = ak if ((h - sk) % 24) < dk else 0
            lam = base[h] + surge
            Q = Q + lam - cap[h]
            if Q < 0:
                Q = 0
            if t >= 24 and Q > peak:
                peak = Q
        if peak > worst:
            worst = peak
    return worst


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    W, L, C, g, r, base, K, profiles = read_instance(in_path)
    allowed = set(h for h in range(24) if h % g == r)

    try:
        with open(out_path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(toks) != W:
        fail(f"expected exactly {W} tokens, got {len(toks)}")

    starts = []
    for t in toks:
        try:
            v = int(t)
        except Exception:
            fail(f"non-integer token '{t}'")
        if not math.isfinite(v):
            fail("non-finite token")
        if v < 0 or v > 23:
            fail(f"start hour {v} out of range [0,23]")
        if v not in allowed:
            fail(f"start hour {v} violates shift rule (must be h%{g}=={r})")
        starts.append(v)

    F = worst_case_peak(starts, L, C, base, profiles)

    # internal baseline: fully-clustered roster (all workers start at the
    # first allowed hour) -- the naive "just staff everyone the same" default.
    r0 = min(allowed)
    baseline_starts = [r0] * W
    B = worst_case_peak(baseline_starts, L, C, base, profiles)
    B = max(B, 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"F={F} B={B} Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
