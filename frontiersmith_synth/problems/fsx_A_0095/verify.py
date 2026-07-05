#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the bakery self-interference (first-autocorrelation)
minimization problem.

Instance (<in>):  "n V"
Artifact (<out>): n integers a_0..a_{n-1}, each in [0, V], not all zero.

Objective (MINIMIZE):
    S    = sum_i a_i
    conv[k] = sum_i a_i * a_{k-i}          (full self-convolution, length 2n-1)
    peak = max_k conv[k]
    c    = 2 * n * peak / S^2

Feasibility (any violation -> "Ratio: 0.0"):
    * exactly n whitespace-separated integer tokens
    * every token in [0, V]
    * S > 0

Scoring (minimization, per AGENT_BRIEF calibration):
    B  = c of the checker's own baseline schedule (a triangular "ramp-up /
         ramp-down" plan the checker builds itself)
    sc = min(1000.0, 100.0 * B / max(1e-9, c))
    print "Ratio: sc/1000"
The baseline (triangular) schedule scores exactly 0.1; a schedule 10x better
than the baseline would cap at 1.0 (unreachable here -> genuinely open-ended).
All arithmetic is exact integer/rational, so the score is bit-for-bit
reproducible.
"""
import sys


def conv_peak(a):
    n = len(a)
    best = 0
    for k in range(2 * n - 1):
        s = 0
        lo = k - n + 1
        if lo < 0:
            lo = 0
        hi = k if k < n - 1 else n - 1
        for i in range(lo, hi + 1):
            s += a[i] * a[k - i]
        if s > best:
            best = s
    return best


def c_value(a):
    """Return c = 2*n*peak/S^2 as an exact float via Fraction, or None if S<=0."""
    from fractions import Fraction
    n = len(a)
    S = sum(a)
    if S <= 0:
        return None
    peak = conv_peak(a)
    return Fraction(2 * n * peak, S * S)


def baseline_schedule(n, V):
    """Triangular ramp-up/ramp-down plan (deterministic)."""
    a = []
    for i in range(n):
        # frac in [0,1], peak in the middle
        frac = 1.0 - abs(2.0 * i / (n - 1) - 1.0)
        a.append(int(round(V * frac)))
    if sum(a) == 0:
        a[0] = V
    return a


def fail(reason):
    sys.stderr.write("reason: %s\n" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    with open(sys.argv[1]) as f:
        n, V = map(int, f.read().split())
    with open(sys.argv[2]) as f:
        toks = f.read().split()

    if len(toks) != n:
        fail("expected %d tokens, got %d" % (n, len(toks)))
    a = []
    for tk in toks:
        try:
            x = int(tk)
        except ValueError:
            fail("non-integer token %r" % tk)
        if x < 0 or x > V:
            fail("token %d out of range [0,%d]" % (x, V))
        a.append(x)
    if sum(a) <= 0:
        fail("total production is zero")

    c = c_value(a)
    if c is None:
        fail("total production is zero")

    B = c_value(baseline_schedule(n, V))
    # both exact Fractions; ratio = 100 * B / c
    sc = 100.0 * float(B) / max(1e-9, float(c))
    if sc > 1000.0:
        sc = 1000.0
    sys.stderr.write("c=%.6f baseline=%.6f\n" % (float(c), float(B)))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
