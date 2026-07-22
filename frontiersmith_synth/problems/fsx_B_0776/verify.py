#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for hydraulic-trunk-sizing (fsx_B_0776).

Reads the instance, validates the participant's discrete diameter assignment
(feasibility: right count, in-range indices, within budget, finite), simulates the
resulting per-node head loss (Weymouth-style, HL ~ K*L*Q^2/D^5, Q fixed by demand-
conservation along the tree), scores each node's demand satisfaction on a per-node
normalized scale between the WORST possible loss (every pipe at min diameter) and the
BEST possible loss (every pipe at max diameter) on that node's own root path, sums the
unmet demand, and reports Ratio = 100*B/F (minimization) against the checker's own
uniform-diameter baseline construction B.
"""
import sys
import math

DIAMS = list(range(1, 9))


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split('\n')
    except Exception as e:
        fail(f"cannot read input: {e}")
    toks = [t for t in toks if t.strip() != ""]
    if len(toks) < 3:
        fail("truncated input")
    n = int(toks[0].split()[0])
    line2 = toks[1].split()
    ndiam = int(line2[0])
    diams = [int(x) for x in line2[1:1 + ndiam]]
    C = int(toks[2].split()[0])
    if len(toks) < 3 + (n - 1):
        fail("truncated instance rows")
    parent = [-1] * n
    demand = [0] * n
    length = [0] * n
    unit_cost = [0] * n
    K = [0.0] * n
    for i in range(1, n):
        row = toks[2 + i].split()
        p1 = int(row[0])
        parent[i] = p1 - 1  # convert to 0-indexed
        demand[i] = int(row[1])
        length[i] = int(row[2])
        unit_cost[i] = int(row[3])
        K[i] = float(row[4])
    return dict(n=n, diams=diams, C=C, parent=parent, demand=demand,
                length=length, unit_cost=unit_cost, K=K)


def children_of(inst):
    n = inst['n']; parent = inst['parent']
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    return children


def subtree_flow(inst, children):
    n = inst['n']
    order = []
    st = [(0, False)]
    while st:
        node, processed = st.pop()
        if processed:
            order.append(node); continue
        st.append((node, True))
        for c in children[node]:
            st.append((c, False))
    S = [0] * n
    for v in order:
        S[v] = inst['demand'][v] + sum(S[c] for c in children[v])
    return S


def edge_cost(inst, v, D):
    return inst['unit_cost'][v] * inst['length'][v] * D * D


def uniform_total_cost(inst, D):
    return sum(edge_cost(inst, v, D) for v in range(1, inst['n']))


def head_loss(inst, v, S, D):
    Q = S[v]
    return inst['K'][v] * inst['length'][v] * Q * Q / (D ** 5)


def path_loss_uniform(inst, S, children, D_const):
    n = inst['n']
    L = [0.0] * n
    st = [0]
    while st:
        v = st.pop()
        for c in children[v]:
            L[c] = L[v] + head_loss(inst, c, S, D_const)
            st.append(c)
    return L


def path_loss_actual(inst, S, children, diam_idx):
    n = inst['n']
    L = [0.0] * n
    st = [0]
    while st:
        v = st.pop()
        for c in children[v]:
            D = inst['diams'][diam_idx[c]]
            L[c] = L[v] + head_loss(inst, c, S, D)
            st.append(c)
    return L


def total_unmet(inst, diam_idx, S, children, bestL, worstL):
    n = inst['n']
    L = path_loss_actual(inst, S, children, diam_idx)
    unmet = 0.0
    for v in range(1, n):
        denom = worstL[v] - bestL[v]
        if denom < 1e-9:
            f = 1.0
        else:
            f = (worstL[v] - L[v]) / denom
            f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        unmet += (1 - f) * inst['demand'][v]
    return unmet


def uniform_max_index(inst, C):
    best = 0
    for i, D in enumerate(inst['diams']):
        if uniform_total_cost(inst, D) <= C:
            best = i
    return best


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    inf, outf = sys.argv[1], sys.argv[2]
    inst = read_instance(inf)
    n = inst['n']
    ndiam = len(inst['diams'])

    try:
        raw = open(outf).read().split()
    except Exception as e:
        fail(f"cannot read output: {e}")

    if len(raw) != n - 1:
        fail(f"expected {n-1} diameter indices, got {len(raw)}")

    idx = [0] * n
    for i in range(1, n):
        tok = raw[i - 1]
        try:
            val = int(tok)
        except ValueError:
            fail(f"token '{tok}' is not an integer (nan/inf/garbage rejected)")
        if not (0 <= val < ndiam):
            fail(f"diameter index {val} out of range [0,{ndiam-1}] at edge {i}")
        idx[i] = val

    total_cost = sum(edge_cost(inst, v, inst['diams'][idx[v]]) for v in range(1, n))
    if total_cost > inst['C']:
        fail(f"total cost {total_cost} exceeds budget {inst['C']}")

    children = children_of(inst)
    S = subtree_flow(inst, children)
    for v in range(1, n):
        if not math.isfinite(S[v]):
            fail("non-finite flow (should be unreachable)")
    bestL = path_loss_uniform(inst, S, children, inst['diams'][-1])
    worstL = path_loss_uniform(inst, S, children, inst['diams'][0])
    for v in range(1, n):
        if not (math.isfinite(bestL[v]) and math.isfinite(worstL[v])):
            fail("non-finite head loss (should be unreachable)")

    F = total_unmet(inst, idx, S, children, bestL, worstL)
    if not math.isfinite(F):
        fail("non-finite objective")

    # checker's own baseline: largest uniform diameter (same for every pipe) affordable
    base_idx = [uniform_max_index(inst, inst['C'])] * n
    B = total_unmet(inst, base_idx, S, children, bestL, worstL)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"total_unmet={F:.6f} baseline_unmet={B:.6f} total_cost={total_cost} budget={inst['C']}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
