# TIER: strong
# Cancellation insight + sharing. Reformulate each DENSE row via the all-ones sum:
#   build S = x_0 ^ ... ^ x_{n-1} ONCE, then realise row r as  r = S ^ (fold of r's MISSING
#   bits).  S contains bits r does not, so this relies on intermediate cancellation
#   (x ^ x = 0) -- provably unreachable by any cancellation-free / monotone sharing scheme,
#   because a shared sub-sum reused by r must be a subset of r's support.
# For dense complement-structured rows the MISSING sets are sparse, so the shared folds are
# tiny; the monotone (greedy) heuristic instead pays to share the large DIRECT supports.
import sys
from collections import Counter

def main():
    tok = sys.stdin.read().split()
    p = 0
    def nx():
        nonlocal p
        v = int(tok[p]); p += 1; return v
    m = nx(); n = nx()
    rows = []
    for _ in range(m):
        bits = tuple(j for j in range(n) if nx())
        rows.append(bits)

    gates = []
    def emit(a, b):
        gates.append((a, b))
        return n + len(gates) - 1

    # classify rows
    plan = []            # per row: ('d', tuple bits)  or  ('c', frozenset missing)
    use_S = False
    for bits in rows:
        w = len(bits)
        cw = n - w
        if cw < w - 1 or cw == 0:
            miss = frozenset(j for j in range(n) if j not in bits)
            plan.append(("c", miss)); use_S = True
        else:
            plan.append(("d", bits))

    S_node = None
    if use_S:
        node = 0
        for j in range(1, n):
            node = emit(node, j)
        S_node = node

    # Paar-share the sparse complement folds among the cancellation rows.
    # working sets over signal ids (start as input indices); returns final node per set.
    idx_c = [i for i, pl in enumerate(plan) if pl[0] == "c" and len(pl[1]) >= 1]
    sets = [set(plan[i][1]) for i in idx_c]
    while True:
        cnt = Counter()
        for s in sets:
            sig = sorted(s); L = len(sig)
            for x in range(L):
                for y in range(x + 1, L):
                    cnt[(sig[x], sig[y])] += 1
        if not cnt:
            break
        (bi, bj), c = cnt.most_common(1)[0]
        if c < 2:
            break
        node = emit(bi, bj)
        for s in sets:
            if bi in s and bj in s:
                s.discard(bi); s.discard(bj); s.add(node)
    comp_node = {}
    for local, i in enumerate(idx_c):
        sig = sorted(sets[local])
        node = sig[0]
        for b in sig[1:]:
            node = emit(node, b)
        comp_node[i] = node

    outputs = [None] * m
    for i, pl in enumerate(plan):
        if pl[0] == "d":
            bits = pl[1]
            node = bits[0]
            for b in bits[1:]:
                node = emit(node, b)
            outputs[i] = node
        else:  # cancellation row
            miss = pl[1]
            if len(miss) == 0:
                outputs[i] = S_node                 # row == all-ones == S
            else:
                outputs[i] = emit(S_node, comp_node[i])

    out = [str(len(gates))]
    for a, b in gates:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outputs)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
