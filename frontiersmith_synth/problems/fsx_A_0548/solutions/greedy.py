# TIER: greedy
# Paar-style monotone common-subexpression sharing (the natural "share the most frequent
# pair" heuristic). Cancellation-free: every shared intermediate is a subset of the rows
# that use it, so it can never form the all-ones sum -> it leaves the cancellation win on
# the table for dense (complement-structured) matrices.
import sys
from collections import Counter

def main():
    tok = sys.stdin.read().split()
    p = 0
    def nx():
        nonlocal p
        v = int(tok[p]); p += 1; return v
    m = nx(); n = nx()
    row_sets = []
    for _ in range(m):
        s = set(j for j in range(n) if nx())
        row_sets.append(s)

    gates = []
    def emit(a, b):
        gates.append((a, b))
        return n + len(gates) - 1

    # iterative most-frequent-pair sharing
    while True:
        cnt = Counter()
        for s in row_sets:
            sig = sorted(s)
            L = len(sig)
            for x in range(L):
                for y in range(x + 1, L):
                    cnt[(sig[x], sig[y])] += 1
        if not cnt:
            break
        (bi, bj), c = cnt.most_common(1)[0]
        if c < 2:
            break
        node = emit(bi, bj)
        for s in row_sets:
            if bi in s and bj in s:
                s.discard(bi); s.discard(bj); s.add(node)

    # fold whatever remains in each row (unshared final combine)
    outputs = []
    for s in row_sets:
        sig = sorted(s)
        node = sig[0]
        for b in sig[1:]:
            node = emit(node, b)
        outputs.append(node)

    out = [str(len(gates))]
    for a, b in gates:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outputs)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
