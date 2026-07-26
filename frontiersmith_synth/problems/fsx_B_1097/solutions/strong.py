# TIER: strong
"""The insight: caps make every length scarce long before its full a**l layer is used
up, so committing the WHOLE alphabet to one length (greedy's move) wastes alphabet
that other lengths could still cash in. If two lengths use pairwise DISJOINT digit
subsets, no string from one can ever be a scattered subsequence of a string from the
other -- regardless of length order -- so several length-"layers" can be stacked at
once (a Dilworth-style chain-cover/partition argument, not a single antichain slab).

We DP over how many of the `a` digits to dedicate to each length (0..remaining) and
how much of the global budget T to spend there, maximizing total weight; a length
given `al` digits can hold up to min(cap[l], al**l) of its strings. Then we
reconstruct the winning allocation and materialize the strings."""
import sys
import itertools


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it))
    Lmax = int(next(it))
    T = int(next(it))
    weight = [int(next(it)) for _ in range(Lmax)]   # weight[l-1] = weight of length l
    cap = [int(next(it)) for _ in range(Lmax)]

    NEG = -1
    dp = [[NEG] * (T + 1) for _ in range(a + 1)]
    dp[0][0] = 0
    choice = {}  # (l, u, b) -> (al, count, prev_u, prev_b)

    for l in range(1, Lmax + 1):
        wl = weight[l - 1]
        capl = cap[l - 1]
        ndp = [[NEG] * (T + 1) for _ in range(a + 1)]
        for u in range(a + 1):
            row = dp[u]
            for b in range(T + 1):
                base = row[b]
                if base < 0:
                    continue
                for al in range(0, a - u + 1):
                    maxcount = min(capl, al ** l)
                    cand = []
                    for c in (0, maxcount, min(maxcount, T - b)):
                        if c not in cand:
                            cand.append(c)
                    for count in cand:
                        if count < 0:
                            continue
                        nb = b + count
                        if nb > T:
                            continue
                        nu = u + al
                        if nu > a:
                            continue
                        nv = base + wl * count
                        if nv > ndp[nu][nb]:
                            ndp[nu][nb] = nv
                            choice[(l, nu, nb)] = (al, count, u, b)
        dp = ndp

    bu, bb, bv = 0, 0, -1
    for u in range(a + 1):
        for b in range(T + 1):
            if dp[u][b] > bv:
                bv, bu, bb = dp[u][b], u, b

    picks = {}
    u, b = bu, bb
    for l in range(Lmax, 0, -1):
        al, count, pu, pb = choice[(l, u, b)]
        picks[l] = (al, count)
        u, b = pu, pb

    out = []
    next_free = 0
    for l in range(1, Lmax + 1):
        al, count = picks[l]
        if al == 0 or count == 0:
            continue
        digits = list(range(next_free, next_free + al))
        next_free += al
        cnt = 0
        for tup in itertools.product(digits, repeat=l):
            if cnt >= count:
                break
            out.append("".join(str(d) for d in tup))
            cnt += 1

    print(len(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
