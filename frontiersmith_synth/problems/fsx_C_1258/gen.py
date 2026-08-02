#!/usr/bin/env python3
"""gen.py <testId> -- prints one Bloom-cascade-allocation instance to stdout.

Deterministic: every random draw is seeded ONLY from testId via
random.Random(testId*1_000_003 + role), so re-running with the same testId
reproduces byte-identical output forever. counter.py re-derives the exact
same S / hash coefficients / hot keys with the identical formulas (and also
draws a held-out tail sample the participant never sees) -- see counter.py.
"""
import sys
import random

L = 4          # number of cascade layers (fixed)
KMAX = 6       # max hash functions any layer may use
P = (1 << 31) - 1   # prime modulus (2^31-1, Mersenne prime)
COST = [1, 3, 9, 27]  # lookup cost of visiting layer i (1-indexed -> COST[i-1])
DPEN = 2500    # penalty charged when a non-member survives (leaks through) all L layers


def rng(test_id, role):
    return random.Random(test_id * 1_000_003 + role)


def build_instance(test_id):
    n = 500 + 100 * test_id
    universe = 25 * n
    bits_per_key = 20
    budget = bits_per_key * n

    r_mem = rng(test_id, 11)
    members = sorted(r_mem.sample(range(universe), n))
    mset = set(members)

    coeffs = []
    for i in range(L):
        r_c = rng(test_id, 100 + i)
        layer_coeffs = [(r_c.randint(1, P - 1), r_c.randint(0, P - 1)) for _ in range(KMAX)]
        coeffs.append(layer_coeffs)

    skewed = test_id >= 4
    hot = []
    if skewed:
        r_hot = rng(test_id, 21)
        h_count = 5
        chosen = set()
        while len(chosen) < h_count:
            k = r_hot.randrange(universe)
            if k not in mset:
                chosen.add(k)
        for k in sorted(chosen):
            w = r_hot.randint(30, 80)
            hot.append((k, w))

    def gen_tail(role_tail, count):
        r_t = rng(test_id, role_tail)
        tail = []
        got = 0
        while got < count:
            k = r_t.randrange(universe)
            if k not in mset:
                tail.append((k, 1))
                got += 1
        return tail

    tail_count = 400 + 80 * test_id
    tail_vis = gen_tail(31, tail_count)
    tail_hid = gen_tail(41, tail_count)

    return dict(test_id=test_id, universe=universe, n=n, budget=budget,
                members=members, coeffs=coeffs, hot=hot,
                tail_vis=tail_vis, tail_hid=tail_hid)


def emit(inst):
    out = []
    out.append(f"{inst['test_id']}")
    out.append(f"{inst['n']} {inst['universe']} {L} {KMAX} {inst['budget']}")
    out.append(" ".join(map(str, COST)))
    out.append(f"{DPEN}")
    out.append(" ".join(map(str, inst['members'])))
    for layer_coeffs in inst['coeffs']:
        for (a, b) in layer_coeffs:
            out.append(f"{a} {b}")
    out.append(f"{len(inst['hot'])}")
    for (k, w) in inst['hot']:
        out.append(f"{k} {w}")
    out.append(f"{len(inst['tail_vis'])}")
    for (k, w) in inst['tail_vis']:
        out.append(f"{k} {w}")
    print("\n".join(out))


def main():
    test_id = int(sys.argv[1])
    inst = build_instance(test_id)
    emit(inst)


if __name__ == "__main__":
    main()
