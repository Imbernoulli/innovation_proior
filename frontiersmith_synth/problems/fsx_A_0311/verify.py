#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the resonance-free
traffic-signal deployment (cap set in AG(n,3)) problem.

Feasibility: distinct tokens, each length n over {0,1,2}, and NO three distinct
schedules x,y,z with x_i+y_i+z_i == 0 (mod 3) at every intersection i.

Score (maximization): sc = min(1000, 100*F/B), Ratio = sc/1000, with internal
baseline B = 2^(n-2) (a trivial resonance-free family). Any violation -> Ratio 0.0.
Exact integer arithmetic over F_3 -> bit-for-bit deterministic."""
import sys


def emit(r, msg=""):
    if msg:
        print(msg)
    print("Ratio: %.6f" % r)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        emit(0.0, "usage error")
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    try:
        with open(inf) as f:
            toks = f.read().split()
        n = int(toks[0]); q = int(toks[1])
    except Exception:
        emit(0.0, "bad instance")
    if q != 3 or n < 1 or n > 20:
        emit(0.0, "bad instance params")

    B = 2 ** (n - 2)  # trivial baseline family size

    # ---- read participant output ----
    try:
        with open(outf) as f:
            data = f.read()
    except Exception:
        emit(0.0, "no output")

    tokens = data.split()
    # bound: cannot exceed the whole space; anything larger is malformed
    if len(tokens) > 3 ** n:
        emit(0.0, "too many schedules (> 3^n)")

    seen = set()
    S = []
    for t in tokens:
        if len(t) != n:
            emit(0.0, "bad token length: %r" % t[:24])
        for c in t:
            if c not in "012":
                emit(0.0, "bad phase character: %r" % t[:24])
        if t in seen:
            emit(0.0, "duplicate schedule: %r" % t[:24])
        seen.add(t)
        S.append(tuple(ord(c) - 48 for c in t))

    k = len(S)

    # ---- feasibility: no grid-lock resonance (no line in AG(n,3)) ----
    # For distinct x,y the unique completing point is z_i = (-(x_i+y_i)) mod 3;
    # since x != y, z is automatically distinct from both. Resonance iff z in S.
    setS = seen
    rng = range(n)
    for i in range(k):
        xi = S[i]
        for j in range(i + 1, k):
            xj = S[j]
            z = "".join(chr(48 + ((-(xi[c] + xj[c])) % 3)) for c in rng)
            if z in setS:
                emit(0.0, "grid-lock resonance among 3 schedules")

    F = k
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    emit(sc / 1000.0, "feasible cap: F=%d baseline_B=%d" % (F, B))


if __name__ == "__main__":
    main()
