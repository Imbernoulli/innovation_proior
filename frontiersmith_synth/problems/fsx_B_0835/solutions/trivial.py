# TIER: trivial
"""
Full non-adaptive scan: run every test in the given input order 0,1,...,T-1
on every patient, ignoring outcomes for routing until the very end, then use
the observed T-bit outcome vector to pick the (unique, precomputed) label.
This exactly reproduces the checker's own baseline construction.
"""
import sys


def resolve(tok, F, vals):
    if tok[0] == 'F':
        return F[int(tok[1:])]
    if tok[0] == 'I':
        return vals[int(tok[1:])]
    return int(tok[1:])


def eval_all(instrs, F):
    vals = []
    for (op, a, b) in instrs:
        va, vb = resolve(a, F, vals), resolve(b, F, vals)
        if op == 'ADD':
            v = va + vb
        elif op == 'SUB':
            v = va - vb
        else:
            v = va * vb
        vals.append(v)
    return vals


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v

    K = int(nxt()); M = int(nxt()); T = int(nxt()); N = int(nxt())
    instrs = []
    for _ in range(M):
        instrs.append((nxt(), nxt(), nxt()))
    tests = []
    for _ in range(T):
        tests.append((int(nxt()), int(nxt())))
    patients = []
    for _ in range(N):
        F = [int(nxt()) for _ in range(K)]
        w = int(nxt()); lab = int(nxt())
        patients.append((F, w, lab))

    # build a complete binary tree of depth T: level t asks test t regardless
    # of outcome so far; leaves are indexed by the T-bit path (0=lo,1=hi).
    # leaf label = the (unique) true label of any patient whose full outcome
    # vector matches that path (there always is at least the patient(s) that
    # produced it, by construction of the input).
    leaf_label = {}
    for (F, w, lab) in patients:
        vals = eval_all(instrs, F)
        bits = tuple(1 if vals[fin] >= thr else 0 for (fin, thr) in tests)
        leaf_label[bits] = lab  # consistent by construction (same bits -> same label)

    # node numbering: complete binary tree, node 0 = root testing test0.
    # internal node at depth d (0-indexed, d=0..T-1), 2^d nodes at that depth,
    # global index = (2^d - 1) + local_index_within_depth.
    numnodes = (1 << (T + 1)) - 1  # T levels of TEST + one level of LEAF (2^T leaves)
    # We'll lay out: nodes 0 .. (2^T - 2) are TEST nodes (T levels, indices
    # 0..2^T-2), nodes (2^T - 1) .. (2^(T+1) - 2) are LEAF nodes (2^T of them).
    out_lines = []
    test_start = 0
    leaf_start = (1 << T) - 1
    total_nodes = leaf_start + (1 << T)
    node_desc = [None] * total_nodes

    def test_node_index(depth, local):
        return (1 << depth) - 1 + local

    for depth in range(T):
        for local in range(1 << depth):
            idx = test_node_index(depth, local)
            if depth + 1 < T:
                lo = test_node_index(depth + 1, 2 * local)
                hi = test_node_index(depth + 1, 2 * local + 1)
            else:
                lo = leaf_start + 2 * local
                hi = leaf_start + 2 * local + 1
            node_desc[idx] = ('TEST', depth, lo, hi)

    for local in range(1 << T):
        idx = leaf_start + local
        bits = tuple((local >> (T - 1 - d)) & 1 for d in range(T))
        lab = leaf_label.get(bits, 0)  # unreachable combos: label irrelevant
        node_desc[idx] = ('LEAF', lab)

    out_lines.append(str(total_nodes))
    for nd in node_desc:
        if nd[0] == 'LEAF':
            out_lines.append("LEAF %d" % nd[1])
        else:
            _, depth, lo, hi = nd
            out_lines.append("TEST %d %d %d" % (depth, lo, hi))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
