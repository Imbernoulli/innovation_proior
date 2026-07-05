#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>   (ans is an empty placeholder -- ignored)

Format-D checker for the substation phasor-coupling decomposition problem.

  1. Read the target tensor T (B x L x H, integer) from <in>.
  2. Parse the participant's rank-R multiplier program from <out>:
         R
         then R primitives, each 3 lines:
             a  (B rationals)   -- bus  weights
             b  (L rationals)   -- line weights
             c  (H rationals)   -- harmonic weights
     Rationals accepted as int / decimal / "p/q". NaN/Inf or any parse error -> reject.
  3. EXACT feasibility gate: sum_r a_r[i]*b_r[j]*c_r[k] must equal T[i][j][k] for ALL
     (i,j,k), in exact rational arithmetic. Any mismatch / malformed output -> Ratio: 0.0.
  4. Objective F = R = number of hardware scalar multiplies (fewer is better -> minimize).
     Internal baseline B0 = number of non-zero harmonic (mode-3) fibers = a trivial feasible
     rank the checker can always build itself.
         sc = min(1000, 100 * B0 / F);   Ratio = sc / 1000     (trivial R == B0 -> 0.1)

Deterministic, exact, O(size). Emits exactly one 'Ratio:' line as the last line.
"""
import sys
from fractions import Fraction


def emit(r, reason=""):
    if reason:
        sys.stdout.write("reason: %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % r)
    sys.exit(0)


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def main():
    if len(sys.argv) < 3:
        emit(0.0, "usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- parse instance ----
    tok = read_tokens(in_path)
    try:
        it = iter(tok)
        B = int(next(it)); L = int(next(it)); H = int(next(it))
        if B <= 0 or L <= 0 or H <= 0:
            raise ValueError
        T = [[[0] * H for _ in range(L)] for _ in range(B)]
        for k in range(H):
            for i in range(B):
                for j in range(L):
                    T[i][j][k] = int(next(it))
    except Exception:
        emit(0.0, "bad instance")

    # baseline B0 = number of non-zero harmonic fibers (over k), for a given (bus,line)
    B0 = 0
    for i in range(B):
        for j in range(L):
            if any(T[i][j][k] != 0 for k in range(H)):
                B0 += 1
    if B0 <= 0:
        B0 = 1  # all-zero tensor guard

    # ---- parse participant output ----
    try:
        with open(out_path) as f:
            otok = f.read().split()
    except Exception:
        emit(0.0, "no output")
    if not otok:
        emit(0.0, "empty output")

    try:
        R = int(otok[0])
    except Exception:
        emit(0.0, "R not an integer")
    if R < 1:
        emit(0.0, "R < 1")

    per = B + L + H
    cap = 3 * B * L * H + 50           # reject absurd / DoS-sized programs
    if R > cap:
        emit(0.0, "R exceeds cap")
    need = 1 + R * per
    if len(otok) != need:
        emit(0.0, "token count %d != expected %d" % (len(otok), need))

    def parse_frac(s):
        low = s.lower()
        if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            raise ValueError("non-finite")
        return Fraction(s)

    prims = []
    idx = 1
    try:
        for _ in range(R):
            a = [parse_frac(otok[idx + t]) for t in range(B)]; idx += B
            b = [parse_frac(otok[idx + t]) for t in range(L)]; idx += L
            c = [parse_frac(otok[idx + t]) for t in range(H)]; idx += H
            prims.append((a, b, c))
    except Exception:
        emit(0.0, "bad rational / non-finite in primitive")

    # ---- EXACT reconstruction gate ----
    That = [[[Fraction(0) for _ in range(H)] for _ in range(L)] for _ in range(B)]
    for (a, b, c) in prims:
        for i in range(B):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(L):
                aibj = ai * b[j]
                if aibj == 0:
                    continue
                row = That[i][j]
                for k in range(H):
                    ck = c[k]
                    if ck != 0:
                        row[k] += aibj * ck

    for i in range(B):
        for j in range(L):
            for k in range(H):
                if That[i][j][k] != T[i][j][k]:
                    emit(0.0, "reconstruction mismatch at bus %d line %d band %d" % (i, j, k))

    F = R
    sc = min(1000.0, 100.0 * B0 / max(1e-9, F))
    emit(sc / 1000.0)


if __name__ == "__main__":
    main()
