# TIER: strong
"""The insight: never let a prefix become invalid in the first place.

Instead of chasing value order and rolling back whatever blocks it, choose
UP FRONT a subset S of flags that is (a) closed under REQUIRES (every
prerequisite of a chosen flag is also chosen) and (b) independent in the
CONFLICTS graph (no two chosen flags ever conflict). Every such S can be
enabled with zero rollbacks, in some order respecting REQUIRES -- so every
prefix of that order is automatically a valid configuration.

Within a fixed valid S, ordering only affects HOW LONG each flag stays
active before the horizon ends, so the best order enables, at each step,
the highest-value flag that is currently ready (a safe exchange-argument
optimal rule once no eviction can ever happen).

The instance is small, so we search S exhaustively over subsets (2^N,
N <= 12) rather than special-casing any one algorithm -- the reformulation
into a constraint-respecting *ordering* problem is the actual insight, not
raw enumeration power."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    N = int(nxt())
    values = [int(nxt()) for _ in range(N)]
    R = int(nxt())
    req_of = {i: [] for i in range(1, N + 1)}
    for _ in range(R):
        c = int(nxt())
        p = int(nxt())
        req_of[c].append(p)
    C = int(nxt())
    conf_of = {i: set() for i in range(1, N + 1)}
    for _ in range(C):
        a = int(nxt())
        b = int(nxt())
        conf_of[a].add(b)
        conf_of[b].add(a)

    flags = list(range(1, N + 1))

    def eval_subset(mask):
        S = [f for f in flags if mask & (1 << (f - 1))]
        Sset = set(S)
        for f in S:
            for p in req_of[f]:
                if p not in Sset:
                    return None
        for f in S:
            if conf_of[f] & Sset:
                return None
        active = set()
        order = []
        remaining = set(S)
        while remaining:
            ready = [f for f in remaining if all(p in active for p in req_of[f])]
            if not ready:
                return None  # cannot happen: requires graph is acyclic + S is closed
            f = max(ready, key=lambda z: (values[z - 1], -z))
            order.append(f)
            active.add(f)
            remaining.discard(f)
        total = 0
        for i, f in enumerate(order, start=1):
            total += values[f - 1] * (N - i + 1)
        return total, order

    best_total = -1
    best_order = []
    for mask in range(1 << N):
        res = eval_subset(mask)
        if res is None:
            continue
        total, order = res
        if total > best_total:
            best_total = total
            best_order = order

    out = [f"E {f}" for f in best_order]
    while len(out) < N:
        out.append("P")
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
