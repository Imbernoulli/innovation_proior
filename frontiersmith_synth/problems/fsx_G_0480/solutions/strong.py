# TIER: strong
# Combine several conflict-free (Sidon) code constructions and keep the largest:
#   1) increasing-order greedy (Mian-Chowla) -- a strong maximal set;
#   2) an Erdos-Turan quadratic-residue construction, then greedy augmentation;
#   3) a deterministic remove-and-reaugment local search seeded from (1).
# Fully deterministic (fixed RNG seed). No known optimal construction exists for
# general n, so this still leaves headroom.
import sys
import random


def is_prime(p):
    if p < 2:
        return False
    i = 2
    while i * i <= p:
        if p % i == 0:
            return False
        i += 1
    return True


def sums_of(S):
    s = set()
    m = len(S)
    for i in range(m):
        ai = S[i]
        for j in range(i, m):
            s.add(ai + S[j])
    return s


def augment(S, sums, Sset, n, order):
    for x in order:
        if x in Sset:
            continue
        c2 = 2 * x
        if c2 in sums:
            continue
        ok = True
        cand = []
        for y in S:
            c = x + y
            if c in sums:
                ok = False
                break
            cand.append(c)
        if ok:
            S.append(x)
            Sset.add(x)
            sums.add(c2)
            sums.update(cand)
    return S


def mian_chowla(n):
    S = []
    sums = set()
    for x in range(1, n + 1):
        c2 = 2 * x
        if c2 in sums:
            continue
        ok = True
        cand = []
        for y in S:
            c = x + y
            if c in sums:
                ok = False
                break
            cand.append(c)
        if ok:
            S.append(x)
            sums.add(c2)
            sums.update(cand)
    return S


def erdos_turan(n):
    best = []
    p = 2
    while 2 * p * p <= n:
        if is_prime(p):
            s = [2 * p * i + ((i * i) % p) + 1 for i in range(p)]
            if s and max(s) <= n and len(s) > len(best):
                best = s
        p += 1
    return best


def local_search(base, n, T, seed):
    rng = random.Random(seed)
    best = sorted(base)
    order = list(range(1, n + 1))
    for it in range(T):
        cur = list(best)
        r = 1 + (it % 3)
        if len(cur) <= r:
            continue
        rem = set(rng.sample(cur, r))
        cur = [e for e in cur if e not in rem]
        cset = set(cur)
        sm = sums_of(cur)
        rng.shuffle(order)
        augment(cur, sm, cset, n, order)
        if len(cur) > len(best):
            best = sorted(cur)
    return best


def main():
    n = int(sys.stdin.read().split()[0])

    cands = []

    mc = mian_chowla(n)
    cands.append(mc)

    et = erdos_turan(n)
    et = sorted(set(et))
    et_sums = sums_of(et)
    et_set = set(et)
    augment(et, et_sums, et_set, n, list(range(1, n + 1)))
    cands.append(et)

    # Bound local-search iterations so runtime stays well under the limit.
    T = max(10, min(50, 200000 // max(1, n)))
    ls = local_search(mc, n, T, 20260703)
    cands.append(ls)

    best = max(cands, key=len)
    best = sorted(set(best))
    sys.stdout.write(" ".join(map(str, best)) + "\n")


if __name__ == "__main__":
    main()
