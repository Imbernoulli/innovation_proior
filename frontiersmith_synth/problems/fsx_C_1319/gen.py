#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE self-assembly-scheduling instance to stdout.

Instance layout:
    N M T theta0
    u_1 v_1 s_1 type_1      # type in {T, D}
    ...
    u_M v_M s_M type_M

`T`-type bonds form a node-disjoint perfect matching on a subset of the N monomers
(the target structure).  `D`-type bonds ("decoys") are extra candidate pairs that
compete for the same monomer sites but are not part of the target; their strength
is drawn from a strictly higher range than the target bonds' so that "strongest
bond first" scheduling means "decoys first".  testIds 3, 6, 9 are engineered
TRAP cases with dense decoy contention against the target's monomers.
"""
import sys
import random

TRAP_IDS = {3, 6, 9}
TRAP_MULT = 1.3
THETA0 = 14
TARGET_LO, TARGET_HI = 1, 4     # weak / reversible-until-late
DECOY_LO, DECOY_HI = 8, 12      # strong / locks in early -- strictly above THETA0's
                                  # crossing of the target range, strictly below THETA0


def build(test_id):
    rng = random.Random(2_000_003 + test_id * 97 + 13)
    K = 4 + test_id                      # 5..14 target bonds
    N = 2 * K
    perm = list(range(N))
    rng.shuffle(perm)
    target_edges = [(perm[2 * i], perm[2 * i + 1]) for i in range(K)]
    target_strength = [rng.randint(TARGET_LO, TARGET_HI) for _ in range(K)]

    is_trap = test_id in TRAP_IDS
    if is_trap:
        Dmin, Dmax = K, max(K + 1, int(TRAP_MULT * K))
    else:
        Dmin, Dmax = max(1, K // 4), max(2, K // 2)
    D = rng.randint(Dmin, Dmax)

    # Cap decoys to at most one per monomer: this keeps the per-node contention
    # a clean 1-vs-1 race (decoy vs. target) instead of letting several decoys
    # pile onto the same site, which would make the checker's own reference
    # baseline collapse to near zero on the densest trap cases and blow the
    # score ratio's headroom.
    seen = set(tuple(sorted(e)) for e in target_edges)
    node_decoy_count = [0] * N
    decoy_edges = []
    attempts = 0
    while len(decoy_edges) < D and attempts < 60 * D + 600:
        u = rng.randrange(N)
        v = rng.randrange(N)
        attempts += 1
        if u == v:
            continue
        e = tuple(sorted((u, v)))
        if e in seen:
            continue
        if node_decoy_count[u] >= 1 or node_decoy_count[v] >= 1:
            continue
        seen.add(e)
        node_decoy_count[u] += 1
        node_decoy_count[v] += 1
        decoy_edges.append((u, v))
    D = len(decoy_edges)
    decoy_strength = [rng.randint(DECOY_LO, DECOY_HI) for _ in range(D)]

    bonds = []
    for i, (u, v) in enumerate(target_edges):
        bonds.append([u, v, target_strength[i], 'T'])
    for i, (u, v) in enumerate(decoy_edges):
        bonds.append([u, v, decoy_strength[i], 'D'])

    # Shuffle presentation order so the fixed processing order used by the
    # simulator (input order) carries no bias toward target or decoy bonds.
    order = list(range(len(bonds)))
    rng.shuffle(order)
    bonds = [bonds[i] for i in order]

    M = len(bonds)
    Tmax = max(15, M + rng.randint(5, 15))
    return N, M, Tmax, THETA0, bonds


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    N, M, Tmax, theta0, bonds = build(test_id)
    out = [f"{N} {M} {Tmax} {theta0}"]
    for (u, v, s, typ) in bonds:
        out.append(f"{u} {v} {s} {typ}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
