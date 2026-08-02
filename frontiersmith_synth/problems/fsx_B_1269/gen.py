#!/usr/bin/env python3
"""gen.py <testId> -- generator for fsx_B_1269 (cross-border-tax-route).

Builds a jurisdiction graph with TWO parallel routes from source (id 0) to
target (id n-1):

  1. A dedicated "backbone" chain: one fixed-rate, fixed-substance,
     fixed-instrument-type real jurisdiction per layer, connected only to
     its own neighbours in the chain. This is always structurally simple,
     always compliant, and deliberately priced at a middling rate (it is
     the checker's own reference route, and also what solutions/trivial.py
     reproduces) -- it is not rate-optimized.

  2. A layered "candidate" network: layer_1 -> layer_2 -> ... -> layer_L,
     fully connected between consecutive layers (and to source/target),
     where a routing plan picks exactly one node per layer. Each layer
     mixes REAL jurisdictions (high economic substance, moderate treaty
     rate) and, on TRAP test ids, SHELL jurisdictions (near-zero substance,
     very low "treaty-shopped" rate).

On trap test ids every candidate layer has exactly one real node and the
rest are shells: the all-shell candidate route is by far the cheapest hop-
by-hop, but it fails the path-level anti-conduit substance test and/or the
timing window -- a per-hop-rate shortest-path search cannot see this, only
a search that evaluates whole routes can. On easy test ids every candidate
node is real, so the candidate search is unconstrained by compliance and a
plain cheapest-route search reliably beats the fixed backbone.

All node ids increase along every edge (source < layer_1 < ... < layer_L <
backbone chain < target), so the graph is a DAG in id order.

Determinism: all randomness is seeded from testId only.
"""
import sys, random

# (L, widths, is_trap) per testId, 1..10 -- small -> large/adversarial ladder
CONFIG = {
    1: (1, [2], False),
    2: (1, [2], True),
    3: (2, [2, 2], False),
    4: (2, [3, 3], False),
    5: (3, [3, 3, 3], True),
    6: (3, [3, 3, 3], False),
    7: (3, [4, 3, 4], False),
    8: (4, [3, 4, 3, 4], True),
    9: (4, [3, 4, 3, 4], False),
    10: (4, [4, 4, 4, 4], False),
}

BASELINE_RATE_BP = 3000     # 30% statutory (non-treaty) withholding reference
GAMMA = 10                  # substance-required-per-benefit constant (see statement)
V0 = 1_000_000              # principal amount routed

BB_RATE_BP = 2900           # fixed backbone hop rate (plain vanilla, not rate-optimal)
BB_HOLD = 2
BB_SUBSTANCE = 70
BB_TYPE = 0


def gen(test_id):
    L, widths, trap = CONFIG[test_id]
    rnd = random.Random(20260 + 97 * test_id)

    # ---- node id layout ----
    # 0 = source; candidate layer blocks; then L backbone nodes; then target.
    layer_start = []
    nxt = 1
    for w in widths:
        layer_start.append(nxt)
        nxt += w
    bb_start = nxt
    bb_ids = list(range(bb_start, bb_start + L))
    target = bb_start + L
    n = target + 1

    substance = [0] * n
    role = [None] * n
    substance[0] = 100
    substance[target] = 100
    for bb in bb_ids:
        substance[bb] = BB_SUBSTANCE
        role[bb] = 'R'

    for li, w in enumerate(widths):
        start = layer_start[li]
        num_real = w if not trap else 1     # trap layers: exactly 1 real, rest shell
        for j in range(w):
            nid = start + j
            if j < num_real:
                role[nid] = 'R'
                substance[nid] = rnd.randint(40, 90)
            else:
                role[nid] = 'S'
                substance[nid] = rnd.randint(0, 6)

    def candidate_layer_nodes(li):
        if li == 0:
            return [0]
        if li == L + 1:
            return [target]
        start = layer_start[li - 1]
        return list(range(start, start + widths[li - 1]))

    edges = []   # (u, v, rate_bp, hold, itype)

    # candidate network: fully connect consecutive layers
    for li in range(0, L + 1):
        us = candidate_layer_nodes(li)
        vs = candidate_layer_nodes(li + 1)
        for u in us:
            for v in vs:
                r = 'R' if v == target else role[v]
                if r == 'S':
                    rate_bp = rnd.randint(50, 200)
                    hold = 1
                else:
                    rate_bp = rnd.randint(700, 2400)
                    hold = rnd.randint(2, 3)
                itype = rnd.randint(0, 1)
                edges.append((u, v, rate_bp, hold, itype))

    # backbone chain: source -> bb_1 -> ... -> bb_L -> target, fixed pricing
    chain = [0] + bb_ids + [target]
    for u, v in zip(chain, chain[1:]):
        edges.append((u, v, BB_RATE_BP, BB_HOLD, BB_TYPE))

    # ---- timing window ----
    T_min = L + 2
    T_max = 4 * (L + 1)

    backbone = chain

    m = len(edges)
    out = []
    out.append(f"{n} {m}")
    out.append(f"{V0} {BASELINE_RATE_BP} {GAMMA} {T_min} {T_max}")
    out.append(" ".join(str(s) for s in substance))
    for (u, v, rate_bp, hold, itype) in edges:
        out.append(f"{u} {v} {rate_bp} {hold} {itype}")
    out.append(str(len(backbone)))
    out.append(" ".join(str(x) for x in backbone))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    tid = int(sys.argv[1])
    sys.stdout.write(gen(tid))
