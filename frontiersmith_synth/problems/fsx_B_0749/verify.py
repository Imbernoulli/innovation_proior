#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Vesting Lock-in Allocator.

Simulates the commitment schedule exactly as specified, validates feasibility
strictly (no over-commit, no negative commit, all-finite), computes terminal
wealth F, and normalizes against the trivial "keep everything in cash" baseline
B = C0.
"""
import sys, math


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        toks = f.read().split()
    p = 0
    T = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    C0 = int(toks[p]); p += 1
    L = [int(toks[p + i]) for i in range(K)]; p += K
    rate = []
    for _t in range(T):
        row = [int(toks[p + i]) for i in range(K)]
        p += K
        rate.append(row)

    try:
        with open(outf) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("no output file")

    if len(out_toks) != T * K:
        fail("expected %d commit values (T*K), got %d" % (T * K, len(out_toks)))

    commits = []
    for i, s in enumerate(out_toks):
        try:
            v = float(s)
        except ValueError:
            fail("token %d ('%s') is not a number" % (i, s))
        if not math.isfinite(v):
            fail("token %d is not finite" % i)
        if v < -1e-6:
            fail("negative commit amount %.6f at token %d" % (v, i))
        commits.append(max(0.0, v))

    EPS = 1e-6
    free_cash = [0.0] * (T + 2)  # buckets 1..T+1 (T+1 = terminal, no more reinvestment)
    free_cash[1] = float(C0)

    for t in range(1, T + 1):
        avail = free_cash[t]
        row = commits[(t - 1) * K:(t - 1) * K + K]
        total = sum(row)
        if total > avail + EPS * max(1.0, avail):
            fail("period %d commits %.6f but only %.6f free" % (t, total, avail))
        leftover = avail - total
        if leftover < 0:
            leftover = 0.0
        for k in range(K):
            amt = row[k]
            if amt <= 0:
                continue
            Lk = L[k]
            window_end = min(t + Lk - 1, T)
            mult = 1.0
            for s in range(t, window_end + 1):
                mult *= (1.0 + rate[s - 1][k] / 10000.0)
                if not math.isfinite(mult):
                    fail("overflow while compounding instrument %d" % k)
            val = amt * mult
            bucket = min(t + Lk, T + 1)
            free_cash[bucket] += val
        free_cash[t + 1] += leftover

    F = free_cash[T + 1]
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative terminal wealth")

    B = float(C0)  # trivial feasible construction: never commit anything
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("terminal_wealth=%.6f baseline=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
