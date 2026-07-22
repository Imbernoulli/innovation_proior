#!/usr/bin/env python3
"""gen.py <testId> -- emits one "Welding a Frame That Flinches at Every Joint" instance.

Deterministic: all randomness is seeded from testId only (random.Random(testId)).

The frame is a disjoint union of small gadgets:
  - a "ring" gadget: a simple cycle of length L (every node degree 2).
  - a "wheel" gadget: a hub joined to R rim nodes by spokes, with the rim
    nodes also wired into a cycle (hub degree R, rim degree 3).
No gadget has a degree-1 node, so no joint's final displacement is pinned
to a fixed magnitude regardless of strategy -- every joint's residual is
genuinely a function of the chosen weld order and sides.

Each gadget is built in one of two modes:
  - "uniform": every strut in the gadget carries the SAME pull rating D.
    This is the classic "equal and opposite" setup that tempts a solver
    into assuming mirrored struts cancel outright -- they don't, once
    stiffness has drifted between the two welds (the trap).
  - "noise": each strut gets an independently drawn rating, so a solver
    cannot special-case "all ratings equal" and must handle the general
    stiffness-discounted accumulation.

testId selects a manifest of gadgets (growing in count/size with testId);
manifests for testId 2,3,4,5,6,7,9,10 include >=1 "uniform" gadget of size
>=4 specifically to trap a stiffness-blind greedy.
"""
import random
import sys


def make_ring(rng, start, L, mode):
    edges = []
    if mode == "uniform":
        d = rng.randint(20, 80)
        for i in range(L):
            edges.append((start + i, start + (i + 1) % L, d))
    else:
        for i in range(L):
            mag = rng.randint(10, 60)
            sign = rng.choice((1, -1))
            edges.append((start + i, start + (i + 1) % L, sign * mag))
    return L, edges


def make_wheel(rng, start, R, mode):
    hub = start
    rim = [start + 1 + i for i in range(R)]
    edges = []
    if mode == "uniform":
        d = rng.randint(20, 80)
        for r in rim:
            edges.append((hub, r, d))
        for i in range(R):
            edges.append((rim[i], rim[(i + 1) % R], d))
    else:
        for r in rim:
            mag = rng.randint(10, 60)
            sign = rng.choice((1, -1))
            edges.append((hub, r, sign * mag))
        for i in range(R):
            mag = rng.randint(10, 60)
            sign = rng.choice((1, -1))
            edges.append((rim[i], rim[(i + 1) % R], sign * mag))
    return R + 1, edges


MANIFEST = {
    1: [("ring", 4, "noise"), ("ring", 3, "noise")],
    2: [("wheel", 3, "uniform"), ("ring", 5, "noise")],
    3: [("wheel", 4, "noise"), ("wheel", 4, "uniform")],
    4: [("wheel", 4, "uniform"), ("wheel", 5, "uniform"), ("ring", 4, "noise")],
    5: [("wheel", 5, "uniform"), ("wheel", 4, "noise"), ("wheel", 3, "uniform")],
    6: [("wheel", 5, "uniform"), ("wheel", 4, "uniform"), ("ring", 5, "noise"), ("wheel", 4, "noise")],
    7: [("wheel", 6, "uniform"), ("wheel", 5, "uniform"), ("wheel", 4, "uniform"), ("ring", 4, "noise")],
    8: [("ring", 4, "noise"), ("ring", 4, "noise"), ("ring", 4, "noise"),
        ("wheel", 5, "noise"), ("wheel", 5, "noise"), ("wheel", 5, "uniform")],
    9: [("wheel", 5, "uniform"), ("wheel", 5, "uniform"), ("wheel", 5, "uniform"), ("wheel", 4, "uniform"),
        ("ring", 4, "noise"), ("ring", 4, "noise")],
    10: [("wheel", 8, "uniform"), ("wheel", 7, "uniform"), ("wheel", 6, "uniform"),
         ("wheel", 5, "noise"), ("ring", 5, "noise")],
}


def build(testId):
    rng = random.Random(testId)
    manifest = MANIFEST[testId]
    n = 0
    edges = []
    for kind, size, mode in manifest:
        if kind == "ring":
            cnt, es = make_ring(rng, n, size, mode)
        else:
            cnt, es = make_wheel(rng, n, size, mode)
        n += cnt
        edges.extend(es)
    return n, edges


def main():
    testId = int(sys.argv[1])
    n, edges = build(testId)
    out = [f"{n} {len(edges)}"]
    for u, w, eff in edges:
        out.append(f"{u} {w} {eff}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
