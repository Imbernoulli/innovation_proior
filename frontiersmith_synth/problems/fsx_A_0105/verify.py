#!/usr/bin/env python3
# verify.py <in> <out> <ans>
# Deterministic scorer for the discrete first-autocorrelation constant C1 (minimization).
# Feasibility: exactly n finite numbers in [0, 1e9] with strictly positive sum.
# Score: Ratio = min(1000, 100*B/C1(f))/1000, B = C1 of the naive half-fill baseline.
import sys, math

CAP = 1e9

def c1(f):
    n = len(f)
    S = sum(f)
    if S <= 0:
        return None
    g = [0.0] * (2 * n - 1)
    for i in range(n):
        fi = f[i]
        if fi == 0.0:
            continue
        base = i
        for j in range(n):
            g[base + j] += fi * f[j]
    return 2.0 * n * max(g) / (S * S)

def fail(msg):
    print("reason: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as fh:
        n = int(fh.read().split()[0])

    try:
        with open(outp) as fh:
            toks = fh.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)

    if len(toks) != n:
        fail("expected %d numbers, got %d" % (n, len(toks)))

    f = []
    for tk in toks:
        try:
            v = float(tk)
        except ValueError:
            fail("non-numeric token %r" % tk)
        if not math.isfinite(v):
            fail("non-finite value")
        if v < -1e-12 or v > CAP:
            fail("value %r out of range [0, %g]" % (v, CAP))
        f.append(max(0.0, v))

    if sum(f) <= 0.0:
        fail("total energy must be strictly positive")

    F = c1(f)
    if F is None or F <= 0.0:
        fail("degenerate profile")

    # internal naive baseline: fill first half uniformly, silent second half
    L = n // 2
    base = [1.0] * L + [0.0] * (n - L)
    B = c1(base)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("C1: %.6f  B: %.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
