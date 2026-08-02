#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE "kinetics-from-endpoint" instance to stdout.

Story: a source S feeds flux into N intermediate nodes (branch points of a
reaction/pathway network). Node i carries a KNOWN flux weight w_i (>=1,
integer). Node i then splits its own flux among a fixed subset of L_i final
products (endpoints) with a HIDDEN branching-ratio vector r_i (a probability
distribution over its own product subset). The instance also reports the
BASELINE product distribution actually observed under normal conditions --
the flux-weighted sum of every node's contribution to each product. Because
several DIFFERENT nodes can share the exact same product subset, many
different choices of their individual r_i reproduce the identical baseline
(endpoint-degeneracy): baseline data alone cannot tell you which node
produced how much of a shared product.

A "perturbation query" on node i means: isolate node i (route ALL flux
through it alone, i.e. knock out every other node) and re-measure the
product distribution -- this reveals r_i exactly. The solver has a budget of
Q such queries. gen.py never prints r_i or which nodes are informationally
redundant -- the solver must read the topology (which nodes share which
product subset) and the flux weights, and decide where to spend the budget.

Instance format (stdout):
  line 1: testId N L Q W
  line 2: N integers w_1..w_N (flux weight of node 1..N)
  next N lines: "deg e_1 .. e_deg" -- the sorted 1-indexed product ids node i
      (1-indexed, in order) feeds into. Two nodes with the IDENTICAL sorted
      product-id list are "confounded" (indistinguishable from baseline
      alone: their individual r's cannot be separated without a query).
  last line: L floats -- the observed BASELINE product distribution
      (flux-weighted sum over all nodes), 10 decimal digits.

Difficulty ladder (testId 1..10): testId 1-3 are warm-ups with no adversarial
structure (any reasonable budget spend works about as well as any other).
testId 4-10 each plant ONE oversized confounded cluster whose full
separation costs strictly MORE than the entire remaining budget (after its
few decoy singleton nodes, which need zero queries and are already fully
determined by baseline alone) -- while several smaller, fully-resolvable
clusters sit elsewhere, together needing about the same total budget. A
per-EDGE / per-node "highest flux first" scan (ignoring which node belongs
to which cluster and what finishing a cluster costs) either burns its whole
budget on already-determined decoys, or dumps it into the oversized cluster
and never finishes separating anything.

All randomness is seeded ONLY from testId -> fully deterministic.
"""
import sys
import random
from fractions import Fraction as Fr

DECOY_W = 6
BAIT_W = 5
VALUE_W = 4
DENOM = 10000
P_LO, P_HI = 9700, 9900   # a node's own-product share: 97.00%-99.00% of its flux


def make_ladder():
    # (Q, #decoy singleton nodes, sizes of the resolvable clusters, easy?)
    specs = [
        (2, 0, [3],          True),
        (3, 0, [4],          True),
        (5, 0, [3, 3],       True),
        (6, 1, [4, 4],       False),
        (8, 1, [4, 4, 3],    False),
        (10, 1, [5, 4, 4],   False),
        (11, 1, [5, 5, 4],   False),
        (13, 2, [5, 5, 4, 4], False),
        (14, 1, [5, 5, 5, 4], False),
        (17, 2, [5, 5, 5, 5, 4], False),
    ]
    ladder = []
    for Q, d, vals, easy in specs:
        groups = [(1, 'decoy')] * d
        if not easy:
            remaining = Q - d
            bigbait_size = remaining + 3   # cost = remaining+2, STRICTLY > remaining
            groups.append((bigbait_size, 'bait'))
        groups += [(v, 'value') for v in vals]
        ladder.append((groups, Q))
    return ladder


LADDER = make_ladder()


def build_instance(testId):
    """Deterministic, seeded ONLY by testId. Shared verbatim with verify.py so the
    checker can regenerate the identical hidden branching ratios without them ever
    being written to disk."""
    rng = random.Random(700003 * testId + 91)
    groups_spec, Q = LADDER[testId - 1]
    ep_base = 0
    groups = []
    for g, tier in groups_spec:
        w = {'decoy': DECOY_W, 'bait': BAIT_W, 'value': VALUE_W}[tier]
        ws = [w] * g
        rs = []
        if g == 1:
            rs.append([Fr(DENOM, DENOM)])
        else:
            for k in range(g):
                primary = rng.randint(P_LO, P_HI)
                remaining = DENOM - primary
                base = remaining // (g - 1)
                rem = remaining % (g - 1)
                vec = [0] * g
                vec[k] = primary
                others = [e for e in range(g) if e != k]
                for idx, e in enumerate(others):
                    vec[e] = base + (1 if idx < rem else 0)
                assert sum(vec) == DENOM
                rs.append([Fr(x, DENOM) for x in vec])
        groups.append({'g': g, 'ep_base': ep_base, 'ws': ws, 'rs': rs, 'tier': tier})
        ep_base += g
    hub_id = 1
    hubs = []
    for grp in groups:
        grp['hub_ids'] = []
        for k in range(grp['g']):
            hubs.append({'w': grp['ws'][k], 'k': k, 'g': grp['g'], 'r': grp['rs'][k], 'grp': grp})
            grp['hub_ids'].append(hub_id)
            hub_id += 1
    N = hub_id - 1
    L = ep_base
    W = sum(h['w'] for h in hubs)
    return {'groups': groups, 'hubs': hubs, 'N': N, 'L': L, 'W': W, 'Q': Q}


def baseline_vector(inst):
    L = inst['L']
    baseline = [Fr(0)] * L
    for h in inst['hubs']:
        grp = h['grp']
        base = grp['ep_base']
        w = Fr(h['w'], inst['W'])
        for e in range(grp['g']):
            baseline[base + e] += w * h['r'][e]
    return baseline


def main():
    testId = int(sys.argv[1])
    testId = max(1, min(10, testId))
    inst = build_instance(testId)
    N, L, Q, W = inst['N'], inst['L'], inst['Q'], inst['W']

    lines = [f"{testId} {N} {L} {Q} {W}"]
    lines.append(" ".join(str(h['w']) for h in inst['hubs']))
    for h in inst['hubs']:
        grp = h['grp']
        base = grp['ep_base']
        eps = [str(base + e + 1) for e in range(grp['g'])]
        lines.append(f"{grp['g']} " + " ".join(eps))
    bl = baseline_vector(inst)
    lines.append(" ".join("%.10f" % float(x) for x in bl))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
