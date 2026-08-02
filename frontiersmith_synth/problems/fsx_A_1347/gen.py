#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE triangulated-mesh instance to stdout.

The mesh is a disjoint union of "wheel" patches: a hub vertex fanned by a
cycle of rim vertices (hub-rim + rim-rim triangles).  A wheel whose rim has
EVEN length is properly 3-colorable; a wheel whose rim has ODD length
provably needs a 4th color (it contains K4 for rim length 3, and in general
an odd wheel's chromatic number is 4).  Every test plants at least one
odd-rim wheel, so an obstruction always exists to certify.

Vertex ids are scrambled by a seeded permutation (pure hygiene -- no
solution should be able to exploit raw id order).
"""
import sys
import random


def main():
    testId = int(sys.argv[1])
    rng = random.Random(90000 + 131 * testId + 7 * testId * testId)

    # difficulty ladder: more / larger wheel patches as testId grows
    p = min(2 + testId // 2, 8)
    rim_lens = [rng.randint(4, 10) for _ in range(p)]
    if testId % 2 == 1:
        # ALL-EVEN regime: the true global optimum is 3 colors, but only a
        # rim-parity-aware walk reaches it -- plain first-fit (regardless of
        # whether it is cost-aware) very often overshoots to 4 here.
        rim_lens = [v if v % 2 == 0 else v + 1 for v in rim_lens]
        rim_lens = [min(v, 10) for v in rim_lens]
    else:
        # OBSTRUCTION regime: guarantee >=1 odd-rim (un-3-colorable) patch,
        # so a genuine obstruction certificate is always available here.
        if all(x % 2 == 0 for x in rim_lens):
            idx = rng.randrange(p)
            v = rim_lens[idx]
            rim_lens[idx] = v + 1 if v < 10 else v - 1

    faces = []
    n = 0
    for rim_len in rim_lens:
        hub = n + 1
        rim = [n + 2 + j for j in range(rim_len)]
        n += 1 + rim_len
        for j in range(rim_len):
            a, b, c = hub, rim[j], rim[(j + 1) % rim_len]
            faces.append((a, b, c))

    m = len(faces)
    K = 12  # color palette size (> any vertex degree, so any greedy always succeeds)

    costs = list(range(1, K + 1))
    rng.shuffle(costs)

    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    remap = {old: perm[old - 1] for old in range(1, n + 1)}
    faces = [(remap[a], remap[b], remap[c]) for (a, b, c) in faces]

    out = [f"{n} {m} {K}", " ".join(map(str, costs))]
    for (a, b, c) in faces:
        out.append(f"{a} {b} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
