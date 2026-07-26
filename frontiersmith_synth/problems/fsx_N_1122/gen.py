#!/usr/bin/env python3
"""
gen.py <testId> -> prints one instance of decaying-cascade-timing to stdout.

Instance = a DAG of "reinforcement gadgets": each gadget has two external
sources S1,S2, two relay chains of (possibly different) length feeding a
join node J. A chain relays a single pulse instantly (chain edge weight >=
chain node threshold), so activating a source at time t0 makes the far end
of a length-c chain active at exactly time t0+c. J's two incoming edges each
carry a weight strictly below theta_J, but their SUM meets theta_J -- J can
only ever be crossed by receiving BOTH pulses in the exact same simulation
step (nodes emit to neighbours only in the single step right after they
activate; unspent partial credit decays sharply and never recovers).

Deterministic: all randomness seeded from testId only.
"""
import sys
import random


def new_node(theta_list, theta):
    nid = len(theta_list)
    theta_list.append(theta)
    return nid


def build_instance(test_id):
    rng = random.Random(1000003 * test_id + 17)

    gadget_counts = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    idx = min(max(test_id, 1), 10) - 1
    g_total = gadget_counts[idx]
    # a few extra gadgets on the very last (adversarial) tier
    if test_id > 10:
        g_total += (test_id - 10) * 2

    trap_fraction = 0.75

    DECAY_NUM, DECAY_DEN = 1, 3
    THETA_SOURCE = 3
    THETA_CHAIN = 5
    CHAIN_W = 5
    EXT_BOOST = 50

    theta_list = []
    edges = []
    max_chain_len = 0

    for _g in range(g_total):
        is_trap = rng.random() < trap_fraction
        c1 = rng.randint(2, 9)
        if is_trap:
            c2 = rng.randint(2, 9)
            while c2 == c1:
                c2 = rng.randint(2, 9)
        else:
            c2 = c1

        a1 = rng.randint(3, 9)
        a2 = rng.randint(3, 9)
        theta_j = a1 + a2  # individually insufficient, combined exactly sufficient

        s1 = new_node(theta_list, THETA_SOURCE)
        s2 = new_node(theta_list, THETA_SOURCE)

        prev = s1
        for _ in range(c1):
            nd = new_node(theta_list, THETA_CHAIN)
            edges.append((prev, nd, CHAIN_W))
            prev = nd
        p1 = prev

        prev = s2
        for _ in range(c2):
            nd = new_node(theta_list, THETA_CHAIN)
            edges.append((prev, nd, CHAIN_W))
            prev = nd
        p2 = prev

        j = new_node(theta_list, theta_j)
        edges.append((p1, j, a1))
        edges.append((p2, j, a2))

        max_chain_len = max(max_chain_len, c1, c2)

    n = len(theta_list)
    m = len(edges)
    horizon = max_chain_len + 10

    out = []
    out.append(f"{n} {m}")
    out.append(f"{DECAY_NUM} {DECAY_DEN} {EXT_BOOST} {horizon}")
    for th in theta_list:
        out.append(str(th))
    for (u, v, w) in edges:
        out.append(f"{u} {v} {w}")
    return "\n".join(out) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(build_instance(test_id))


if __name__ == "__main__":
    main()
