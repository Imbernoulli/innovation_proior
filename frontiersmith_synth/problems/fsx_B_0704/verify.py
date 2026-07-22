#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the tempering-chart problem.

Reads the billet instance from <in>, the candidate schedule from <out>, strictly
validates feasibility, replays the deterministic per-grain kinetics, and prints
the normalized score as 'Ratio: <float in [0,1]>' on its own final line.
"""
import sys


def fail(msg):
    print(f"{msg} Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args.")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    p = 0
    L = int(itoks[p]); p += 1
    Tmax = int(itoks[p]); p += 1
    C0 = int(itoks[p]); p += 1
    n_max = int(itoks[p]); p += 1
    B = int(itoks[p]); p += 1
    d0 = [int(itoks[p + i]) for i in range(L)]; p += L
    theta = [int(itoks[p + i]) for i in range(L)]; p += L

    D_init = sum(d0)

    try:
        with open(out_path) as f:
            raw = f.read()
    except FileNotFoundError:
        fail("no output file.")

    otoks = raw.split()
    if len(otoks) == 0:
        fail("empty output.")

    ptr = 0
    ntok = otoks[ptr]; ptr += 1
    try:
        if not (ntok.lstrip("-").isdigit()):
            raise ValueError
        n = int(ntok)
    except ValueError:
        fail("n is not an integer.")

    if n < 0 or n > n_max:
        fail(f"n={n} out of range [0,{n_max}].")

    Ts = []
    if n > 0:
        if len(otoks) - ptr < n:
            fail("missing temperature tokens.")
        for _ in range(n):
            tok = otoks[ptr]; ptr += 1
            if not (tok.lstrip("-").isdigit()):
                fail(f"non-integer temperature token '{tok}'.")
            v = int(tok)
            if v < 0 or v > Tmax:
                fail(f"temperature {v} out of range [0,{Tmax}].")
            Ts.append(v)

    if ptr != len(otoks):
        fail("trailing tokens after schedule.")

    cost = sum(C0 + t for t in Ts)
    if cost > B:
        fail(f"fuel cost {cost} exceeds budget {B}.")

    d = list(d0)
    for T in Ts:
        mob = T - 2 if T > 2 else 0
        for i in range(L):
            di = d[i]
            heal = di if di < mob else mob
            nuc = T - theta[i] if T > theta[i] else 0
            d[i] = di - heal + nuc

    F = sum(d)
    baseline = max(1e-9, float(D_init))
    sc = min(1000.0, 100.0 * baseline / max(1e-9, float(F)))
    print("final_defects=%d init_defects=%d steps=%d Ratio: %.6f" % (F, D_init, n, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
