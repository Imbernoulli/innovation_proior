# TIER: strong
# The insight: f is MULTIPLICATIVE, so the coordinate system the law actually lives in is not
# "tag number n in increasing order" -- it is "prime power, indexed by which residue class
# (mod 4) the prime belongs to".  Re-basing onto that coordinate system turns a wildly
# non-smooth-in-n function into a THREE-PARAMETER lookup table.
#
# Any training tag n that is itself a pure prime power p**k (n = 2, 3, 4, 5, 7, 8, 9, 11, ...)
# gives a DIRECT, uncontaminated read of the local law: obs ~= p**k - a*p**(k-1) + noise, so
#     a_est = (p**k - obs) / p**(k-1)
# Pool these estimates by residue class (p == 2, p % 4 == 1, p % 4 == 3) and take the majority
# vote (robust to the +-1..2 stamping noise, since hundreds of primes fall in the training
# range) to recover the three integer discounts EXACTLY.  Emit a MODE PP law over (p, k); the
# checker then reconstructs f on any withheld tag by multiplying this law over its OWN
# prime-power factorization -- the decomposition the law respects, not the one the data arrived
# in.
import sys
from collections import Counter, defaultdict


def factorize(n):
    m = n
    d = 2
    facs = []
    while d * d <= m:
        if m % d == 0:
            k = 0
            while m % d == 0:
                m //= d
                k += 1
            facs.append((d, k))
        d += 1 if d == 2 else 2
    if m > 1:
        facs.append((m, 1))
    return facs


def main():
    toks = sys.stdin.read().split()
    ntr = int(toks[0])
    idx = 1
    pts = []
    for _ in range(ntr):
        n = int(toks[idx]); obs = int(toks[idx + 1])
        idx += 2
        pts.append((n, obs))

    buckets = defaultdict(list)
    for n, obs in pts:
        if n == 1:
            continue
        facs = factorize(n)
        if len(facs) != 1:
            continue          # composite with >1 distinct prime -- confounded, skip
        p, k = facs[0]
        r = "2" if p == 2 else ("1" if p % 4 == 1 else "3")
        a_est = (p ** k - obs) / (p ** (k - 1))
        buckets[r].append(round(a_est))

    def vote(r):
        vals = buckets.get(r)
        if not vals:
            return 0
        return Counter(vals).most_common(1)[0][0]

    a2, a1, a3 = vote("2"), vote("1"), vote("3")

    print("MODE PP")
    print("p**k - (%d if p==2 else (%d if p%%4==1 else %d))*p**(k-1)" % (a2, a1, a3))


if __name__ == "__main__":
    main()
