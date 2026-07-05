#!/usr/bin/env python3
# counter.py <in> <out> <ans>
# Format D (flops / op-count) checker for a CP (rank) decomposition of a survey tensor.
# 1) parse the submitted decomposition strictly (schema, token count, finiteness);
# 2) verify EXACT rational reconstruction of the integer tensor (any mismatch -> 0);
# 3) score = min(1, 0.1 * B / R), B = number of nonzero entries (per-entry baseline).
import sys
from fractions import Fraction

MAX_TOKENS = 5_000_000
MAX_R = 20000


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(next(it))
    return I, J, K, T


def read_tokens(path):
    toks = []
    with open(path) as f:
        for line in f:
            for t in line.split():
                toks.append(t)
                if len(toks) > MAX_TOKENS:
                    return toks, True
    return toks, False


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inp, out = sys.argv[1], sys.argv[2]
    I, J, K, T = read_instance(inp)

    B = sum(1 for i in range(I) for j in range(J) for k in range(K) if T[i][j][k] != 0)
    if B == 0:
        fail("degenerate instance")

    toks, overflow = read_tokens(out)
    if overflow:
        fail("output too large")
    if not toks:
        fail("empty output")

    # ---- R ----
    try:
        R = int(toks[0])
    except Exception:
        fail("R not an integer")
    if R < 1:
        fail("R must be >= 1")
    if R > MAX_R:
        fail("R exceeds cap")

    per = I + J + K
    need = R * per
    if len(toks) - 1 != need:
        fail("expected %d factor tokens, got %d" % (need, len(toks) - 1))

    # ---- parse factors exactly; reject non-finite / malformed ----
    vals = toks[1:]
    idx = 0
    terms = []
    for _ in range(R):
        try:
            a = [Fraction(vals[idx + t]) for t in range(I)]; idx += I
            b = [Fraction(vals[idx + t]) for t in range(J)]; idx += J
            c = [Fraction(vals[idx + t]) for t in range(K)]; idx += K
        except (ValueError, ZeroDivisionError, OverflowError):
            fail("non-rational / non-finite token")
        terms.append((a, b, c))

    # ---- exact reconstruction ----
    recon = [[[Fraction(0) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    for (a, b, c) in terms:
        for i in range(I):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(J):
                aibj = ai * b[j]
                if aibj == 0:
                    continue
                row = recon[i][j]
                for k in range(K):
                    ck = c[k]
                    if ck != 0:
                        row[k] += aibj * ck

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if recon[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    F = R
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write("B(nonzeros)=%d R=%d Ratio: %.6f\n" % (B, R, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
