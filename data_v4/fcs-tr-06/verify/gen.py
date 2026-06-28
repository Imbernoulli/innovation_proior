#!/usr/bin/env python3
"""Random small-case generator for rooted-forest isomorphism.

Usage: gen.py <seed>

Produces two small rooted forests in sol.cpp's input format. To exercise BOTH
YES and NO answers heavily, with some probability we build the second forest as a
random relabeling + sibling-shuffle of the first (guaranteed isomorphic), and
otherwise we build it independently at random (usually non-isomorphic, but the
brute oracle is the ground truth either way).
"""
import sys
import random


def random_forest(rng, n):
    """Return par[1..n]; par[i] in {0} U {1..i-1 chosen as an earlier node}.

    We pick parents only among already-emitted nodes (or 0 = root) so the result
    is always a valid forest with no cycles.
    """
    par = [0] * (n + 1)  # 1-indexed; par[0] unused
    for i in range(1, n + 1):
        # parent is 0 (root) or some earlier node 1..i-1
        choices = [0] + list(range(1, i))
        par[i] = rng.choice(choices)
    return par[1:]


def relabel_isomorphic(rng, par):
    """Build a forest isomorphic to `par` by (1) randomly permuting node ids
    while preserving parent structure, and (2) implicitly shuffling sibling order
    (the output order of nodes already varies). Returns new par list."""
    n = len(par)
    # children adjacency including virtual root 0
    children = {0: []}
    for i in range(1, n + 1):
        children.setdefault(i, [])
    for i in range(1, n + 1):
        children[par[i - 1]].append(i)

    # Choose a random bijection of old ids 1..n -> new ids 1..n.
    new_ids = list(range(1, n + 1))
    rng.shuffle(new_ids)
    old_to_new = {0: 0}
    for old, new in zip(range(1, n + 1), new_ids):
        old_to_new[old] = new

    new_par = [0] * (n + 1)
    for old in range(1, n + 1):
        new_par[old_to_new[old]] = old_to_new[par[old - 1]]
    return new_par[1:]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n1 = rng.randint(0, 8)
    par1 = random_forest(rng, n1)

    mode = rng.random()
    if mode < 0.4 and n1 > 0:
        # force an isomorphic pair
        par2 = relabel_isomorphic(rng, par1)
        n2 = n1
    else:
        n2 = rng.randint(0, 8)
        par2 = random_forest(rng, n2)

    out = []
    out.append(str(n1))
    if par1:
        out.append(" ".join(map(str, par1)))
    out.append(str(n2))
    if par2:
        out.append(" ".join(map(str, par2)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
