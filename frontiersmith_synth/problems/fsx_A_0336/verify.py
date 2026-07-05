#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic checker for the drone flight-signature problem.

Reads the instance (N, beacon r) from <in> and the participant matrix from <out>.
Feasibility (any violation -> `Ratio: 0.0`):
  * output parses to EXACTLY N*N integer tokens (bounded read; rejects nan/inf/floats/garbage),
  * every entry is exactly -1 or +1,
  * row 0 equals the beacon pattern r.
Objective: exact integer D = |det(A)| via fraction-free (Bareiss) elimination.
Score (maximization, log2 of D against baseline B = N-1 = log2(2^{N-1})):
  F = log2(D) (0 if D==0);  sc = min(1000, 100*F/B);  Ratio = sc/1000.
Bit-for-bit deterministic on reruns (no floats in the det; only the final log2).
"""
import sys
import math

MAX_OUT_BYTES = 5_000_000  # adversarial-flood guard


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def bareiss_det(M):
    """Exact integer determinant, fraction-free Bareiss elimination with pivoting."""
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            Ai = A[i]
            aik = Ai[k]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    # --- instance ---
    with open(inf) as f:
        toks = f.read().split()
    if not toks:
        fail("empty instance")
    N = int(toks[0])
    r = [int(x) for x in toks[1:1 + N]]
    if len(r) != N:
        fail("bad instance")

    # --- participant output (bounded, strict integer parse) ---
    with open(outf, "rb") as f:
        raw = f.read(MAX_OUT_BYTES + 1)
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    try:
        text = raw.decode("utf-8", "strict")
    except Exception:
        fail("non-utf8 output")

    parts = text.split()
    if len(parts) != N * N:
        fail("expected %d tokens, got %d" % (N * N, len(parts)))

    vals = []
    for p in parts:
        # strict integer only: rejects nan, inf, 1.0, hex, etc.
        try:
            v = int(p)
        except ValueError:
            fail("non-integer token %r" % p)
        if v != -1 and v != 1:
            fail("entry not in {-1,+1}: %r" % p)
        vals.append(v)

    A = [vals[i * N:(i + 1) * N] for i in range(N)]

    # row 0 must equal the beacon pattern
    if A[0] != r:
        fail("row 0 does not match beacon pattern")

    # --- objective ---
    D = abs(bareiss_det(A))
    F = math.log2(D) if D > 0 else 0.0
    B = float(N - 1)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("D_bits=%.4f baseline_bits=%.1f det_nonzero=%d" % (F, B, 1 if D > 0 else 0))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
