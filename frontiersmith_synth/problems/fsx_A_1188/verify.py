#!/usr/bin/env python3
"""
Deterministic checker for fsx_A_1188 -- "Kinetics From the Endpoint: A
Perturbation Budget".

Reads:
  <in>  : testId N L Q W ; then a line of N flux weights; then N lines
          "deg e_1..e_deg" (the product subset each node feeds); then a line
          of L baseline product-distribution values (context only).
  <out> : M  then M integers -- the set of node ids the solver chooses to
          isolate-and-probe (a perturbation query per id), |set| <= Q.

The HIDDEN branching-ratio vectors r_i are never written to disk. This
checker regenerates them deterministically from testId alone, via the exact
same seeded construction gen.py used (build_instance below is byte-for-byte
the same function) -- so a submitted program never has access to them and
cannot "solve" the instance by guessing the ground truth; its only lever is
WHICH nodes to query.

Reconstruction rule (this IS the scoring mechanism -- stated in full in
statement.md, nothing hidden here beyond the numeric constants):
  Partition nodes into clusters by their (sorted) product subset -- two nodes
  in the same cluster are indistinguishable from baseline data alone
  (endpoint-degeneracy). A cluster of size g is fully SEPARATED once >= g-1
  of its nodes have been queried (the last one is then pinned down exactly
  by subtracting the known nodes' contribution from the cluster's baseline
  total) -- every node in a separated cluster counts as exactly identified.
  A cluster that is NOT fully separated (0..g-2 nodes queried) counts as
  UNSEPARATED regardless of partial queries spent on it: its identification
  level is exactly the "do nothing" level (the flux-weighted AVERAGE of its
  members compared against each true member) -- partial information about a
  cluster does not, by itself, tell you which member produced what.

Per-node identification accuracy = 1 - 0.5 * L1(estimate, truth) in [0,1]
(both are probability vectors over the cluster's own product subset, so L1
in [0,2]). Objective F = flux-weighted average of per-node accuracy over ALL
N nodes (in [0,1]). Internal baseline B = F with an EMPTY query set (the
checker's own trivial "read baseline, query nothing" construction).
Score (maximization): sc = min(1000, 100*F/B), Ratio = sc/1000.
"""
import sys
import random
from fractions import Fraction as Fr

DECOY_W = 6
BAIT_W = 5
VALUE_W = 4
DENOM = 10000
P_LO, P_HI = 9700, 9900


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def make_ladder():
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
            bigbait_size = remaining + 3
            groups.append((bigbait_size, 'bait'))
        groups += [(v, 'value') for v in vals]
        ladder.append((groups, Q))
    return ladder


LADDER = make_ladder()


def build_instance(testId):
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


def group_baseline_acc(g, ws, rs, Wtot):
    """Flux-weighted identification accuracy of a cluster if NO node in it is
    (sufficiently) queried: every member is estimated as the flux-weighted
    average of the whole cluster."""
    r_hat = [Fr(0)] * g
    tot = Fr(sum(ws), Wtot)
    for k in range(g):
        qk = Fr(ws[k], Wtot)
        for e in range(g):
            r_hat[e] += qk * rs[k][e]
    if tot > 0:
        r_hat = [x / tot for x in r_hat]
    acc_sum = Fr(0)
    for k in range(g):
        l1 = sum(abs(r_hat[e] - rs[k][e]) for e in range(g))
        acc = max(Fr(0), 1 - l1 * Fr(1, 2))
        acc_sum += Fr(ws[k], Wtot) * acc
    return acc_sum


def score_query_set(inst, query_ids):
    query_set = set(query_ids)
    total = Fr(0)
    for grp in inst['groups']:
        g = grp['g']
        hub_ids = grp['hub_ids']
        ws = grp['ws']
        rs = grp['rs']
        group_flux = Fr(sum(ws), inst['W'])
        if g == 1:
            total += group_flux * 1
            continue
        queried_local = [k for k, hid in enumerate(hub_ids) if hid in query_set]
        if len(queried_local) >= g - 1:
            total += group_flux * 1
        else:
            total += group_baseline_acc(g, ws, rs, inst['W'])
    return total


def read_instance_header(in_path):
    with open(in_path) as fh:
        toks = fh.read().split()
    if len(toks) < 5:
        fail("truncated instance")
    testId, N, L, Q, W = (int(x) for x in toks[:5])
    if not (1 <= testId <= 10):
        fail("generator invariant violated: bad testId in instance")
    return testId, N, L, Q, W


def parse_output(out_path, N, Q):
    with open(out_path) as fh:
        raw = fh.read().split()
    if len(raw) == 0:
        fail("empty output")
    # reject non-finite / non-integer tokens explicitly before int() (which would
    # itself raise on "nan"/"inf" -- but be defensive and explicit)
    for tok in raw:
        low = tok.lower()
        if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            fail("non-finite token in output")
    try:
        vals = [int(t) for t in raw]
    except ValueError:
        fail("non-integer token in output")
    M = vals[0]
    if M < 0 or M > Q:
        fail("query count out of range (0 <= M <= Q)")
    rest = vals[1:]
    if len(rest) != M:
        fail("expected %d query ids, token count mismatch" % M)
    if len(set(rest)) != len(rest):
        fail("duplicate query id")
    for hid in rest:
        if not (1 <= hid <= N):
            fail("query id out of range [1,%d]" % N)
    return rest


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    testId, N, L, Q, W = read_instance_header(in_path)
    inst = build_instance(testId)
    if inst['N'] != N or inst['L'] != L or inst['Q'] != Q or inst['W'] != W:
        fail("instance/testId mismatch (corrupted input)")

    query_ids = parse_output(out_path, N, Q)

    F = score_query_set(inst, query_ids)
    B = score_query_set(inst, [])

    sc = min(1000.0, 100.0 * float(F) / max(1e-9, float(B)))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (float(F), float(B), ratio))


if __name__ == "__main__":
    main()
