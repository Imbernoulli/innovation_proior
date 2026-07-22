#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for fsx_B_0663 (cayley-ball-growth).

Reads p,k,r (+ the registry's suggested couriers, unused by scoring) from <in>.
Validates the participant's k courier matrices, runs BFS to radius r in the Cayley
graph they generate, and normalizes against the checker's own subgroup-trapped
baseline flood.
"""
import sys

MAX_ABS = 10 ** 9


def mat_inv(M, p):
    a, b, c, d = M
    return (d % p, (-b) % p, (-c) % p, a % p)


def mat_mul(A, B, p):
    a, b, c, d = A
    e, f, g, h = B
    return ((a * e + b * g) % p, (a * f + b * h) % p,
            (c * e + d * g) % p, (c * f + d * h) % p)


def ball_size(gens, r, p):
    S = set()
    for M in gens:
        S.add(M)
        S.add(mat_inv(M, p))
    ident = (1, 0, 0, 1)
    visited = {ident}
    frontier = {ident}
    for _ in range(r):
        nxt = set()
        for M in frontier:
            for s in S:
                nm = mat_mul(M, s, p)
                if nm not in visited:
                    visited.add(nm)
                    nxt.add(nm)
        frontier = nxt
    return len(visited)


def baseline_couriers(k, p, r):
    """The checker's own trivial, always subgroup-trapped construction: k upper-
    triangular unipotent shears with pairwise-independent shift amounts, still all
    commuting (they generate a subgroup of the p-order unipotent radical)."""
    R = 2 * r + 3
    shifts = [pow(R, i, p) for i in range(k)]
    return [(1, s % p, 0, 1) for s in shifts]


def fail(reason):
    print("infeasible:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        in_lines = f.read().split("\n")
    header = in_lines[0].split()
    p, k, r = int(header[0]), int(header[1]), int(header[2])

    try:
        with open(outf) as f:
            raw = f.read()
    except Exception as e:
        fail(f"cannot read output: {e}")
        return

    lines = [ln for ln in raw.split("\n") if ln.strip() != ""]
    if len(lines) != k:
        fail(f"expected exactly {k} lines, got {len(lines)}")
        return

    mats = []
    for ln in lines:
        toks = ln.split()
        if len(toks) != 4:
            fail(f"line does not have exactly 4 tokens: {ln!r}")
            return
        vals = []
        for t in toks:
            try:
                v = int(t)
            except ValueError:
                fail(f"non-integer token: {t!r}")
                return
            if not (-MAX_ABS <= v <= MAX_ABS):
                fail(f"token out of range: {t!r}")
                return
            vals.append(v)
        a, b, c, d = vals
        aa, bb, cc, dd = a % p, b % p, c % p, d % p
        det = (aa * dd - bb * cc) % p
        if det != 1:
            fail(f"determinant {det} != 1 (mod {p}) for matrix {vals}")
            return
        mats.append((aa, bb, cc, dd))

    F = ball_size(mats, r, p)
    B = ball_size(baseline_couriers(k, p, r), r, p)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"F={F} B={B}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
