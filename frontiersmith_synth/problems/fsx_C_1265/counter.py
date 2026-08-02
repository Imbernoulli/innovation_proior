#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the tsv-thermal-placement problem.

1. Parses the instance: M dies, N columns, area budget A, baseline/via per-layer resistances
   R0 > Rv > 0, per-column via area costs a[1..N], and M power maps p[m][1..N] (m=1 nearest
   the heat sink ... m=M farthest / top of stack).
2. Parses the participant's artifact: N tokens x[1..N] in {0,1} (which columns get a via).
   Validates STRICTLY: exactly N tokens, each an exact integer token equal to 0 or 1
   (non-integer / nan / inf / out-of-range / missing / extra -> Ratio: 0.0). Total via area
   cost sum(a[c] for x[c]=1) must be <= A, else Ratio: 0.0.
3. Computes, for every column c, the depth-weighted STACKED heat profile
       W[c] = sum_{m=1}^{M} m * p[m][c]
   (heat generated on die m crosses m layer-boundaries of resistance to reach the sink, and
   every die sharing column c shares that same vertical path). Per-column resistance is
   Rv if a via is placed at c, else R0. F = max_c ( R(c) * W[c] ), the realized peak
   temperature (up to the fixed R0/Rv scale) of the whole stack.
4. Baseline B = the checker's own trivial construction: NO vias placed anywhere (always
   feasible, since it uses 0 of the budget) -> B = R0 * max_c W[c]. Minimization ratio:
       sc = min(1000, 100*B/max(1e-9,F)); print("Ratio: %.6f" % (sc/1000)).
"""
import sys


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        M = int(inext())
        N = int(inext())
        A = int(inext())
        if M <= 0 or N <= 0 or A < 0:
            fail("bad instance header")
        R0 = int(inext())
        Rv = int(inext())
        if not (0 < Rv < R0):
            fail("bad resistances")
        a = [int(inext()) for _ in range(N)]
        if any(v <= 0 for v in a):
            fail("bad area costs")
        P = []
        for _m in range(M):
            row = [int(inext()) for _ in range(N)]
            if any(v < 0 for v in row):
                fail("bad power map")
            P.append(row)
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if len(otoks) != N:
        fail(f"expected exactly {N} tokens, got {len(otoks)}")

    x = []
    for tok in otoks:
        # strict integer parse: rejects "nan"/"inf"/floats/garbage outright
        try:
            iv = int(tok)
        except ValueError:
            fail("non-integer token (garbage/nan/inf/float)")
        if iv not in (0, 1):
            fail("token not in {0,1}")
        x.append(iv)

    total_cost = sum(a[c] for c in range(N) if x[c] == 1)
    if total_cost > A:
        fail(f"area budget exceeded ({total_cost} > {A})")

    # ---- depth-weighted stacked hotspot profile ----
    W = [0] * N
    for c in range(N):
        s = 0
        for m in range(M):
            s += (m + 1) * P[m][c]
        W[c] = s

    F = 0
    for c in range(N):
        Rc = Rv if x[c] == 1 else R0
        Tc = Rc * W[c]
        if Tc > F:
            F = Tc

    B = R0 * max(W)
    if B <= 0:
        fail("degenerate instance (non-positive baseline)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
