# TIER: strong
# Value-aware multi-restart greedy: build the raid-free set several times under
# different priorities (pure value-descending plus seeded perturbations), and
# keep whichever set has the largest total surveillance value.
import sys, itertools, random


def main():
    raw = sys.stdin.read().splitlines()
    ptr = 0
    n = int(raw[ptr].split()[0]); ptr += 1
    b = int(raw[ptr].split()[0]); ptr += 1
    blocked = set()
    for _ in range(b):
        blocked.add(raw[ptr].strip()); ptr += 1
    weights = list(map(int, raw[ptr].split())); ptr += 1

    allv = list(itertools.product(range(3), repeat=n))
    strs = [''.join(map(str, v)) for v in allv]
    wt = {v: weights[i] for i, v in enumerate(allv)}
    allowed = [v for v, s in zip(allv, strs) if s not in blocked]

    def build(order):
        S = []
        Sset = set()
        for v in order:
            ok = True
            for x in S:
                w = tuple((-(x[k] + v[k])) % 3 for k in range(n))
                if w in Sset:
                    ok = False
                    break
            if ok:
                S.append(v)
                Sset.add(v)
        return S

    orders = [sorted(allowed, key=lambda v: (-wt[v], v))]
    for sd in range(4):
        rr = random.Random(1234 + sd)
        jitter = {v: rr.random() * 3.0 for v in allowed}
        orders.append(sorted(allowed, key=lambda v: -(wt[v] + jitter[v])))

    best = None
    best_val = -1
    for order in orders:
        S = build(order)
        val = sum(wt[v] for v in S)
        if val > best_val:
            best_val = val
            best = S

    sys.stdout.write('\n'.join(''.join(map(str, v)) for v in best) + '\n')


if __name__ == "__main__":
    main()
