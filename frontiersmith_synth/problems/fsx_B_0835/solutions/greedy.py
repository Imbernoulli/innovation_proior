# TIER: greedy
"""
Textbook decision-tree recipe (ID3-style): at every node, among tests not yet
used along this root-to-node path, pick the test that maximizes weighted
information gain on the remaining candidate patients (ties -> lowest test
index). Recurse until every leaf is pure. This is exactly the "read off the
most statistically informative question first" heuristic -- it is blind to
how much each test's *preparation* costs, and to whether that cost overlaps
with what has already been computed along the path.
"""
import sys
import math


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


def entropy(weights):
    tot = sum(weights)
    if tot <= 0:
        return 0.0
    h = 0.0
    for w in weights:
        if w > 0:
            pr = w / tot
            h -= pr * math.log2(pr)
    return h


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
    for i in range(N):
        F = [int(nxt()) for _ in range(K)]
        w = int(nxt()); lab = int(nxt())
        patients.append((F, w, lab))

    # per patient: outcome bits for every test, and label
    outcomes = []
    labels = []
    weights = []
    for (F, w, lab) in patients:
        vals = eval_all(instrs, F)
        bits = [1 if vals[fin] >= thr else 0 for (fin, thr) in tests]
        outcomes.append(bits)
        labels.append(lab)
        weights.append(w)

    nodes = []  # filled by build(); each entry finalized as tuple

    def build(idxs, used):
        labs = set(labels[i] for i in idxs)
        if len(labs) == 1:
            nid = len(nodes)
            nodes.append(('LEAF', next(iter(labs))))
            return nid
        wts = [weights[i] for i in idxs]
        base_h = entropy(wts)
        best_t, best_gain = None, -1.0
        for t in range(T):
            if t in used:
                continue
            g0 = [weights[i] for i in idxs if outcomes[i][t] == 0]
            g1 = [weights[i] for i in idxs if outcomes[i][t] == 1]
            tot = sum(g0) + sum(g1)
            if tot == 0:
                continue
            h0 = entropy(g0)
            h1 = entropy(g1)
            cond = (sum(g0) / tot) * h0 + (sum(g1) / tot) * h1
            gain = base_h - cond
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_t = t
        if best_t is None:
            # should not happen given the input's construction; fall back to
            # majority label to stay well-formed.
            nid = len(nodes)
            from collections import Counter
            maj = Counter(labels[i] for i in idxs).most_common(1)[0][0]
            nodes.append(('LEAF', maj))
            return nid
        idxs0 = [i for i in idxs if outcomes[i][best_t] == 0]
        idxs1 = [i for i in idxs if outcomes[i][best_t] == 1]
        nid = len(nodes)
        nodes.append(None)  # placeholder to reserve id
        lo = build(idxs0, used | {best_t}) if idxs0 else build(idxs, used | {best_t})
        hi = build(idxs1, used | {best_t}) if idxs1 else build(idxs, used | {best_t})
        nodes[nid] = ('TEST', best_t, lo, hi)
        return nid

    root = build(list(range(N)), frozenset())
    assert root == 0

    out_lines = [str(len(nodes))]
    for nd in nodes:
        if nd[0] == 'LEAF':
            out_lines.append("LEAF %d" % nd[1])
        else:
            _, ti, lo, hi = nd
            out_lines.append("TEST %d %d %d" % (ti, lo, hi))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
