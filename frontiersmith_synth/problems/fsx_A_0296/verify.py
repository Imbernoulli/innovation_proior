#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans is an empty placeholder -- ignored)

Deterministic scorer for the "subway signal polarity" max-determinant problem.

The participant submits an N x N matrix of +/-1 (a station-signalling scheme).
It must respect the fixed track polarities from <in>.  We compute the EXACT
integer determinant via Bareiss elimination (no floats in the arithmetic), then
score the log-determinant EXCESS of the submission over an internal deterministic
baseline the checker rebuilds itself:

    q(M)  = log2(|det M|)
    L0    = q(baseline) - DELTA          (a fixed floor)
    F     = q(submission) - L0
    B     = q(baseline)   - L0  = DELTA
    sc    = min(1000, 100 * F / B)       (clamped to [0,1000])
    Ratio = sc / 1000

A submission that merely reproduces the baseline scores exactly 0.1.  Because the
score grows with the LOG of the determinant, the exponentially-large gap between a
random matrix and a good construction is compressed into a graded, non-saturating
scale -- there is no known polynomial-time optimum for these odd orders, so the
metric stays genuinely open-ended.
"""
import sys
import math

DELTA = 4.0


def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            piv = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    piv = i
                    break
            if piv < 0:
                return 0
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        mkk = M[k][k]
        Mk = M[k]
        for i in range(k + 1, n):
            Mi = M[i]
            mik = Mi[k]
            for j in range(k + 1, n):
                Mi[j] = (Mi[j] * mkk - mik * Mk[j]) // prev
        prev = mkk
    return sign * M[n - 1][n - 1]


class LCG:
    __slots__ = ("x",)

    def __init__(self, seed):
        self.x = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def nxt(self):
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.x

    def bit(self):
        return 1 if (self.nxt() >> 33) & 1 else -1


def build_baseline(n, seed, fixed):
    """Deterministic feasible reference matrix: fixed cells as given, free cells
    from an LCG stream; reseed on the (astronomically rare) singular draw."""
    for attempt in range(64):
        rng = LCG(seed + attempt * 7919)
        M = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if (i, j) in fixed:
                    M[i][j] = fixed[(i, j)]
                else:
                    M[i][j] = rng.bit()
        if bareiss_det(M) != 0:
            return M
    return M


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    seed = int(next(it))
    k = int(next(it))
    fixed = {}
    for _ in range(k):
        i = int(next(it))
        j = int(next(it))
        v = int(next(it))
        fixed[(i, j)] = v
    return n, seed, fixed


def read_submission(path, n):
    """Strictly parse exactly N*N integer tokens, each in {-1,+1}.  Any deviation
    (wrong count, non-integer, nan/inf, out of range) -> infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) != n * n:
        return None
    vals = []
    for t in toks:
        try:
            x = int(t)
        except ValueError:
            return None
        if x != 1 and x != -1:
            return None
        vals.append(x)
    return [vals[r * n:(r + 1) * n] for r in range(n)]


def main():
    if len(sys.argv) < 3:
        fail("usage")
    n, seed, fixed = read_instance(sys.argv[1])
    M = read_submission(sys.argv[2], n)
    if M is None:
        fail("malformed_or_out_of_range_matrix")
    # honour every fixed track polarity
    for (i, j), v in fixed.items():
        if M[i][j] != v:
            fail("fixed_cell_violation")

    det_sub = bareiss_det(M)
    if det_sub == 0:
        fail("singular_submission")

    base = build_baseline(n, seed, fixed)
    det_base = bareiss_det(base)
    if det_base == 0:
        # unreachable safety net
        fail("baseline_degenerate")

    q_sub = math.log2(abs(det_sub))
    q_base = math.log2(abs(det_base))
    L0 = q_base - DELTA
    F = q_sub - L0
    B = DELTA
    if F <= 0.0:
        sys.stdout.write("|det|=%d below floor\nRatio: 0.0\n" % abs(det_sub))
        sys.exit(0)
    sc = 100.0 * F / max(1e-9, B)
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    sys.stdout.write("|det_sub|=%d |det_base|=%d gain_bits=%.4f Ratio: %.6f\n"
                     % (abs(det_sub), abs(det_base), q_sub - q_base, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
