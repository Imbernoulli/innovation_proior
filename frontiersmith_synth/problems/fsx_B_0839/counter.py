#!/usr/bin/env python3
"""
counter.py -- Format D checker for the alchemist's lossy decanting puzzle.

Input  <in>:  N M / caps[N] / V0 target Nb / Lo_num Lo_den Hi_num Hi_den /
              Nb backbone valve ids / M lines "u v num den" (valve id =
              line order, 0-indexed). Valve ids and non-start tank labels
              are shuffled by the generator, so the backbone (a guaranteed
              feasible plan) is given EXPLICITLY as a list of ids -- there
              is no positional convention (e.g. "last two ids") to exploit.
Output <out>: R
              e_1 e_2 ... e_R   (valve ids applied in order, starting at tank 0)

Validation (exact rational arithmetic throughout, fractions.Fraction):
  - well-formed integers, 1 <= R <= MAXR, every id in [0,M-1]
  - each op's valve must originate at the tank the batch currently occupies
  - each transfer must not exceed the destination tank's capacity
  - the batch must finish in `target` with volume in [Lo, Hi]
Any violation -> "Ratio: 0.0" and exit 0.

Baseline B: the checker itself replays valve ids 0..Nb-1 (the backbone chain
that the generator always plants) through the SAME simulate() used to grade
the participant, and uses its length as the naive/trivial reference.
Objective is minimization (fewest valve ops): Ratio = min(1, 0.1*B/R).
"""
import sys
from fractions import Fraction

MAXR = 500


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    try:
        N = int(next(it)); M = int(next(it))
        caps = [int(next(it)) for _ in range(N)]
        V0 = int(next(it)); target = int(next(it)); Nb = int(next(it))
        Lo = Fraction(int(next(it)), int(next(it)))
        Hi = Fraction(int(next(it)), int(next(it)))
        backbone_ids = [int(next(it)) for _ in range(Nb)]
        edges = []
        for _ in range(M):
            u = int(next(it)); v = int(next(it))
            num = int(next(it)); den = int(next(it))
            edges.append((u, v, num, den))
    except Exception:
        fail("malformed instance (should not happen)")
    return N, M, caps, V0, target, Nb, Lo, Hi, backbone_ids, edges


def simulate(edges, caps, start_tank, start_vol, ops):
    """Replay ops from (start_tank, start_vol). Returns ((tank, vol), None)
    on success or (None, reason) on the first violated constraint."""
    cur_tank = start_tank
    cur_vol = start_vol
    for e in ops:
        if not (0 <= e < len(edges)):
            return None, "op %d: edge id out of range" % e
        u, v, num, den = edges[e]
        if u != cur_tank:
            return None, "edge %d starts at tank %d but batch is in tank %d" % (e, u, cur_tank)
        new_vol = cur_vol * Fraction(num, den)
        if new_vol > caps[v]:
            return None, "edge %d overflows tank %d capacity" % (e, v)
        cur_tank = v
        cur_vol = new_vol
    return (cur_tank, cur_vol), None


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    N, M, caps, V0, target, Nb, Lo, Hi, backbone_ids, edges = parse_input(inp)

    out_toks = open(outp).read().split()
    if not out_toks:
        fail("empty output")
    it = iter(out_toks)
    try:
        R = int(next(it))
    except Exception:
        fail("bad/non-integer R")
    if R < 1:
        fail("R < 1")
    if R > MAXR:
        fail("R too large (> %d)" % MAXR)

    ops = []
    try:
        for _ in range(R):
            ops.append(int(next(it)))
    except Exception:
        fail("bad/missing op tokens (need %d integers)" % R)
    if list(it):
        fail("trailing tokens after the declared plan")

    result, err = simulate(edges, caps, 0, Fraction(V0), ops)
    if err:
        fail(err)
    end_tank, end_vol = result
    if end_tank != target:
        fail("plan ends in tank %d, not target tank %d" % (end_tank, target))
    if not (Lo <= end_vol <= Hi):
        fail("final volume outside target window [Lo,Hi]")

    # ---- checker's own trivial baseline: replay the planted backbone
    #      (its edge ids are given explicitly in the input; no positional
    #      convention like "ids 0..Nb-1" is assumed post-shuffle) ----
    bres, berr = simulate(edges, caps, 0, Fraction(V0), backbone_ids)
    if berr or bres[0] != target or not (Lo <= bres[1] <= Hi):
        fail("internal: backbone baseline invalid (generator bug)")
    B = Nb

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))


if __name__ == "__main__":
    main()
