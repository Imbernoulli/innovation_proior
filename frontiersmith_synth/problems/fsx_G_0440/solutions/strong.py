# TIER: strong
# Multi-restart randomized Paar: run the greedy CSE many times with randomized
# tie-breaking (seeded, deterministic), keep the schedule with the fewest gates.
# Escapes the single deterministic tie-break's local optimum, so it matches or
# beats the plain greedy on most instances while staying well below the optimum.
import sys, random
from collections import Counter

def paar_run(n, rows_bits, rng):
    rows = [set(b) for b in rows_bits]
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
        pair = rng.choice(cands) if rng is not None else min(cands)
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

    # deterministic seed derived from the instance
    seed = (m * 1000003 + n) ^ sum((i + 1) * (v + 1)
                                   for i, r in enumerate(rows_bits) for v in r)
    best = None
    # include the deterministic run, then randomized restarts
    for k in range(81):
        rng = None if k == 0 else random.Random(seed + 7919 * k)
        gates, outs = paar_run(n, rows_bits, rng)
        if best is None or len(gates) < len(best[0]):
            best = (gates, outs)
    gates, outs = best

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    lines.append(" ".join(str(o) for o in outs))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
