# TIER: strong
"""Multi-start priority greedy. Several deterministic orderings (lexicographic, reversed,
a low-weight/balanced priority, and seeded shuffles) each grow a cap avoiding protected
signatures; keep the largest manifest. Routing around the protected set with different
orders beats a single fixed sweep."""
import sys, random


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    pw = [3 ** i for i in range(n)]
    forb = set()
    for _ in range(m):
        code = 0
        for i in range(n):
            code += int(next(it)) * pw[i]
        forb.add(code)

    N = 3 ** n
    digs = [None] * N
    for v in range(N):
        x = v; dd = []
        for _ in range(n):
            dd.append(x % 3); x //= 3
        digs[v] = dd

    def third(a, b):
        da = digs[a]; db = digs[b]; c = 0
        for i in range(n):
            c += ((-da[i] - db[i]) % 3) * pw[i]
        return c

    cand = [v for v in range(N) if v not in forb]

    def grow(order):
        S = set(); out = []
        for v in order:
            good = True
            for a in S:
                if third(a, v) in S:
                    good = False; break
            if good:
                S.add(v); out.append(v)
        return out

    # priority orderings
    orders = []
    orders.append(cand[:])                    # lexicographic
    orders.append(cand[::-1])                 # reversed
    # balanced priority: prefer signatures using few '2' buckets, tie-break by code
    orders.append(sorted(cand, key=lambda v: (sum(1 for d in digs[v] if d == 2), v)))
    orders.append(sorted(cand, key=lambda v: (-sum(1 for d in digs[v] if d == 2), v)))

    starts = 20 if n <= 5 else (14 if n == 6 else 8)
    rng = random.Random(9973)
    for _ in range(starts):
        o = cand[:]; rng.shuffle(o)
        orders.append(o)

    best = []
    for o in orders:
        s = grow(o)
        if len(s) > len(best):
            best = s

    out = []
    for v in best:
        out.append(" ".join(str(d) for d in digs[v]))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
