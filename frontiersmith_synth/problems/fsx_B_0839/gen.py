#!/usr/bin/env python3
"""
gen.py <testId> -- alchemist's lossy decanting puzzle (valve-plan-reachability).

Builds a directed graph of tanks connected by "valves". Each valve fully drains
its source tank and deposits the volume into its destination tank after
multiplying it by a FIXED positive rational factor (numerator/denominator,
exact). Starting with V0 units in tank 0, the goal is a minimum-length
sequence of valve operations that lands the (single, ever-moving) batch of
fluid in a designated target tank with volume inside a narrow window
[Lo, Hi].

Every instance contains:
  - a "backbone" chain of Nb valves (tank i -> tank i+1 in the UNPERMUTED
    numbering) whose factors are chosen so their PRODUCT lands exactly on
    the target value; walking the whole backbone is always a valid (but
    long) plan. Its edge ids (after shuffling, see below) are listed
    explicitly in the input so the checker can replay it without assuming
    any positional convention.
  - a 2-hop "shortcut" pair of valves (tank 0 -> X -> target) whose factors'
    product also lands exactly on the target value.

All factors are monomials in the primes {2,3,5} (never revealed to the
solver): reachable volumes live in a small multiplicative lattice, and the
shortcut's *individual* hop may look numerically far from the target even
though it is exactly 2 ops from goal -- rewarding solvers that reason about
the exponent lattice instead of chasing numeric proximity one step at a time.

Anti-shortcut-leak hardening: valve ids and non-start tank labels are
deterministically shuffled (seeded by testId) so neither "the shortcut is
always the last two edges" nor "the target is always tank M-2" holds --
a submission MUST read and simulate the actual edge list; it cannot recover
the answer from N/M arithmetic alone.

Deterministic: every parameter is a pure function of testId (1..10).
"""
import sys
import random
from fractions import Fraction

# (d2, d3, d5, trap) per testId. trap=True instances plant a shortcut whose
# first hop overshoots badly in raw numeric terms (a myopic nearest-value
# search avoids it and is forced down the long backbone); trap=False
# instances have a "clean" shortcut (zero overshoot) that a nearest-value
# search also finds easily.
PARAMS = {
    1:  (4, -1,  1, False),
    2:  (5,  1, -1, False),
    3:  (6, -2,  1, True),
    4:  (7,  2, -2, True),
    5:  (5, -1,  2, False),
    6:  (8, -3,  1, True),
    7:  (6,  3, -1, True),
    8:  (7, -1,  1, False),
    9:  (9, -2,  2, True),
    10: (10, 2, -3, True),
}

CAP = 10 ** 18


def frac_to_num_den(f):
    return f.numerator, f.denominator


def build(t):
    d2, d3, d5, trap = PARAMS[t]
    Nb = 4 + t                 # backbone edge count (also the checker baseline B)
    target0 = Nb                # target tank index, PRE-shuffle numbering
    X0 = Nb + 1                 # shortcut relay tank, PRE-shuffle numbering
    N = Nb + 2                  # total tanks
    V0 = 1

    # ---- choose overshoot K on the shortcut's first hop ----
    if not trap:
        K = 0
    else:
        Q = Fraction(3) ** d3 * Fraction(5) ** d5
        K = 1
        while True:
            lhs = Q * abs(Fraction(2) ** K - 1)
            rhs = abs(1 - Q)
            if lhs > rhs * 3:      # safety margin so the trap is unambiguous
                break
            K += 1
            if K > 60:
                break

    def monomial(e2, e3, e5):
        num = 2 ** max(e2, 0) * 3 ** max(e3, 0) * 5 ** max(e5, 0)
        den = 2 ** max(-e2, 0) * 3 ** max(-e3, 0) * 5 ** max(-e5, 0)
        return num, den

    # ---- backbone edges: PRE-shuffle id 0..Nb-1, tank i -> tank i+1 ----
    deltas = [(0, 0, 0)] * Nb
    deltas[0] = (d2, 0, 0)
    deltas[1] = (0, d3, 0)
    deltas[2] = (0, 0, d5)

    edges0 = []  # (u, v, num, den) in PRE-shuffle tank numbering, PRE-shuffle id order
    for i in range(Nb):
        e2, e3, e5 = deltas[i]
        num, den = monomial(e2, e3, e5)
        edges0.append((i, i + 1, num, den))

    # ---- shortcut edges: PRE-shuffle id Nb (tank0 -> X, overshoot by K),
    #                      PRE-shuffle id Nb+1 (X -> target, cancels it) ----
    num, den = monomial(d2 + K, d3, d5)
    edges0.append((0, X0, num, den))
    num2, den2 = monomial(-K, 0, 0)
    edges0.append((X0, target0, num2, den2))

    M = len(edges0)
    assert N == Nb + 2 and M == Nb + 2
    backbone_ids0 = list(range(Nb))   # pre-shuffle backbone edge ids

    # ---- deterministic shuffle: break the "shortcut is always the last two
    #      edges / target is always tank N-2" positional leak. Tank 0 (the
    #      fixed start, per the statement) is NOT relabeled; every other
    #      tank and every edge id is permuted. ----
    rng = random.Random(0x5A17 * t + 101)

    others = list(range(1, N))
    shuffled_others = others[:]
    rng.shuffle(shuffled_others)
    tank_map = {0: 0}
    for old, new in zip(others, shuffled_others):
        tank_map[old] = new

    edge_order = list(range(M))       # edge_order[new_id] = old_id
    rng.shuffle(edge_order)
    old_to_new_edge = {old: new for new, old in enumerate(edge_order)}

    edges = [None] * M
    for new_id, old_id in enumerate(edge_order):
        u, v, num, den = edges0[old_id]
        edges[new_id] = (tank_map[u], tank_map[v], num, den)

    target = tank_map[target0]
    backbone_ids = [old_to_new_edge[old] for old in backbone_ids0]

    Vstar = Fraction(2) ** d2 * Fraction(3) ** d3 * Fraction(5) ** d5
    Lo = Vstar * Fraction(999, 1000)
    Hi = Vstar * Fraction(1001, 1000)

    caps = [CAP] * N

    lines = []
    lines.append(f"{N} {M}")
    lines.append(" ".join(str(c) for c in caps))
    lines.append(f"{V0} {target} {Nb}")
    lon, lod = frac_to_num_den(Lo)
    hin, hid = frac_to_num_den(Hi)
    lines.append(f"{lon} {lod} {hin} {hid}")
    lines.append(" ".join(str(e) for e in backbone_ids))
    for (u, v, num, den) in edges:
        lines.append(f"{u} {v} {num} {den}")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > 10:
        t = ((t - 1) % 10) + 1
    sys.stdout.write(build(t))


if __name__ == "__main__":
    main()
