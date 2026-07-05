#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>   (ans is an empty placeholder -- ignored)

Format-D checker for the e-sports synergy-tensor decomposition problem.

  1. Reads the target tensor T (I x J x K, integer) from <in>.
  2. Parses the participant's rank-R trilinear decomposition from <out>:
         R
         then R gadgets, each 3 lines: a (I rationals), b (J rationals), c (K rationals)
     Rationals accepted as int / decimal / "p/q"; NaN/Inf and any parse error -> reject.
  3. EXACT feasibility gate: sum_r a_r[i]*b_r[j]*c_r[k] must equal T[i][j][k] for ALL i,j,k
     (exact rational arithmetic). Any mismatch / malformed output -> Ratio: 0.0.
  4. Objective F = R = number of scalar multiplications (fewer is better -> minimization).
     Internal baseline B = # non-zero frontal (mode-3) fibers  (a trivial feasible rank).
         sc = min(1000, 100 * B / F);   Ratio = sc / 1000     (trivial R==B -> 0.1)

Deterministic, exact, O(size). Prints exactly one 'Ratio:' line (last line).
"""
import sys
from fractions import Fraction


def emit(r, reason=""):
    if reason:
        sys.stdout.write("reason: %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % r)
    sys.exit(0)


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def main():
    if len(sys.argv) < 3:
        emit(0.0, "usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- parse instance ----
    tok = read_ints(in_path)
    try:
        it = iter(tok)
        I = int(next(it)); J = int(next(it)); K = int(next(it))
        T = [[[0] * K for _ in range(J)] for _ in range(I)]
        for k in range(K):
            for i in range(I):
                for j in range(J):
                    T[i][j][k] = int(next(it))
    except Exception:
        emit(0.0, "bad instance")

    # baseline B = number of non-zero mode-3 fibers (frontal fibers over k)
    B = 0
    for i in range(I):
        for j in range(J):
            if any(T[i][j][k] != 0 for k in range(K)):
                B += 1
    if B <= 0:
        B = 1  # degenerate all-zero tensor guard

    # ---- parse participant output ----
    try:
        with open(out_path) as f:
            otok = f.read().split()
    except Exception:
        emit(0.0, "no output")
    if not otok:
        emit(0.0, "empty output")

    # first token = R
    try:
        R = int(otok[0])
    except Exception:
        emit(0.0, "R not an integer")
    if R < 1:
        emit(0.0, "R < 1")
    per = I + J + K
    cap = 3 * I * J * K + 50  # reject absurdly large / DoS decompositions
    if R > cap:
        emit(0.0, "R exceeds cap")
    need = 1 + R * per
    if len(otok) != need:
        emit(0.0, "token count %d != expected %d" % (len(otok), need))

    def parse_frac(s):
        # reject non-finite explicitly; Fraction() would raise on nan/inf anyway
        low = s.lower()
        if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            raise ValueError("non-finite")
        return Fraction(s)

    gadgets = []
    idx = 1
    try:
        for _ in range(R):
            a = [parse_frac(otok[idx + t]) for t in range(I)]; idx += I
            b = [parse_frac(otok[idx + t]) for t in range(J)]; idx += J
            c = [parse_frac(otok[idx + t]) for t in range(K)]; idx += K
            gadgets.append((a, b, c))
    except Exception:
        emit(0.0, "bad rational / non-finite in gadget")

    # ---- EXACT reconstruction gate ----
    That = [[[Fraction(0) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    for (a, b, c) in gadgets:
        for i in range(I):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(J):
                aibj = ai * b[j]
                if aibj == 0:
                    continue
                row = That[i][j]
                for k in range(K):
                    ck = c[k]
                    if ck != 0:
                        row[k] += aibj * ck

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if That[i][j][k] != T[i][j][k]:
                    emit(0.0, "reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    F = R
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    emit(sc / 1000.0)


if __name__ == "__main__":
    main()
