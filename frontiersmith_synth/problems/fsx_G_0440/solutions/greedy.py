# TIER: greedy
# Paar's greedy cancellation-free common-subexpression elimination (Algorithm 1),
# deterministic min tie-break: repeatedly create the XOR of the signal pair that
# co-occurs in the most output rows, replacing it everywhere.  Beats the naive
# per-row baseline by sharing shared gadgets across rows.
import sys
from collections import Counter

def paar(m, n, rows_bits, pick):
    # sig[id] not needed for emission; ids just track creation order.
    rows = [set(b) for b in rows_bits]     # each row = set of signal ids
    gates = []
    next_id = n
    while True:
        cnt = Counter()
        for rs in rows:
            rl = sorted(rs)
            L = len(rl)
            for a in range(L):
                ia = rl[a]
                for b in range(a + 1, L):
                    cnt[(ia, rl[b])] += 1
        if not cnt:
            break
        mx = max(cnt.values())
        cands = [p for p, c in cnt.items() if c == mx]
        pair = pick(cands, mx)
        a, b = pair
        nid = next_id; next_id += 1
        gates.append((a, b))
        for rs in rows:
            if a in rs and b in rs:
                rs.discard(a); rs.discard(b); rs.add(nid)
    outs = [next(iter(rs)) for rs in rows]
    return gates, outs

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    rows_bits = []
    for _ in range(m):
        row = [int(next(it)) for _ in range(n)]
        rows_bits.append([i for i in range(n) if row[i]])

    gates, outs = paar(m, n, rows_bits, lambda cands, mx: min(cands))

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    lines.append(" ".join(str(o) for o in outs))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
