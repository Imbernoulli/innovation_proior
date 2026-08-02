# TIER: greedy
"""The obvious first attempt: match the AVERAGE composition to the Tg target
using the textbook linear mixing rule on the pure-component (diagonal) Tg
values only -- completely ignoring the dyad/interaction matrix -- then just
react the monomers in that ratio (a seeded-random, i.e. statistically
uncontrolled, sequence). This nails the composition but is blind to
blockiness vs. alternation, so it misses whenever the interaction terms
matter."""
import sys, random


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    tg = [int(next(it)) for _ in range(K)]
    for _ in range(K * K):
        next(it)  # M -- deliberately not used
    caps = [int(next(it)) for _ in range(K)]
    target = int(next(it))

    best = None
    # exact grid over (n1, n2), n3 = N - n1 - n2, minimizing the naive
    # composition-weighted average distance to target
    for n1 in range(0, min(caps[0], N) + 1):
        rem_after1 = N - n1
        if rem_after1 > caps[1] + caps[2]:
            continue
        lo2 = max(0, rem_after1 - caps[2])
        hi2 = min(caps[1], rem_after1)
        if lo2 > hi2:
            continue
        # closed-form best n2 in [lo2,hi2] for the linear objective, then
        # check the two integer neighbors of the unconstrained optimum
        # avg = (n1*tg0 + n2*tg1 + n3*tg2)/N, n3 = rem_after1-n2
        # avg = target  =>  n2*(tg1-tg2) = N*target - n1*tg0 - rem_after1*tg2
        denom = (tg[1] - tg[2])
        cands = [lo2, hi2]
        if denom != 0:
            n2_star = (N * target - n1 * tg[0] - rem_after1 * tg[2]) / denom
            cands += [int(n2_star), int(n2_star) + 1]
        else:
            cands += [(lo2 + hi2) // 2]
        for n2 in cands:
            if n2 < lo2 or n2 > hi2:
                continue
            n3 = rem_after1 - n2
            avg = (n1 * tg[0] + n2 * tg[1] + n3 * tg[2]) / N
            err = abs(avg - target)
            if best is None or err < best[0]:
                best = (err, n1, n2, n3)

    if best is None:
        # fallback: fill caps to reach N
        n = [0] * K
        rem = N
        for i in range(K):
            take = min(caps[i], rem)
            n[i] = take
            rem -= take
    else:
        _, n1, n2, n3 = best
        n = [n1, n2, n3]

    multiset = []
    for t in range(K):
        multiset.extend([t + 1] * n[t])
    rng = random.Random(424242 + N * 7 + sum(tg))
    rng.shuffle(multiset)
    print(" ".join(map(str, multiset)))


if __name__ == "__main__":
    main()
