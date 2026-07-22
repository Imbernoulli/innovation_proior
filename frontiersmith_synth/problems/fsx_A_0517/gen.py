# gen.py -- hidden-factor-evaluation-dag (Format D)
# `python3 gen.py <testId>` prints ONE instance (difficulty ladder testId=1..10).
#
# Secret structure: n = 2*F variables are partitioned into F disjoint blocks
# {2m, 2m+1}.  Each block m hosts one hidden homogeneous linear factor
#     f_m = a_m * x_{2m} + b_m * x_{2m+1}   (a_m, b_m nonzero mod p).
# Each target polynomial P_i is the product of THREE distinct hidden factors
# (three distinct blocks).  Because the blocks are disjoint, P_i is a
# multilinear degree-3 polynomial with exactly 8 monomials; the factors are
# SHARED across the k targets.  gen.py prints only the fully expanded
# coefficient list -- the factorization is hidden.
import sys, random

P = 2147483647  # 2^31 - 1, prime

def build(testId):
    t = max(1, int(testId))
    rng = random.Random(1000003 * t + 777)
    F = 6 + 2 * t          # number of hidden factors / blocks
    k = 6 + 3 * t          # number of target polynomials
    n = 2 * F              # variables
    # hidden factors: block m uses vars (2m, 2m+1) with nonzero coeffs
    fac = []
    for m in range(F):
        a = rng.randrange(1, P)
        b = rng.randrange(1, P)
        fac.append(((2 * m, a), (2 * m + 1, b)))
    polys = []
    used = set()
    from math import comb
    cap = comb(F, 3)
    for _ in range(k):
        while True:
            S = tuple(sorted(rng.sample(range(F), 3)))
            if S not in used or len(used) >= cap:
                break
        used.add(S)
        # expand product of the three linear factors over F_p
        poly = {(): 1}  # monomial (sorted var tuple) -> coeff
        for fi in S:
            nxt = {}
            for mono, c in poly.items():
                for (v, cf) in fac[fi]:
                    key = tuple(sorted(mono + (v,)))
                    nxt[key] = (nxt.get(key, 0) + c * cf) % P
            poly = nxt
        poly = {mono: c for mono, c in poly.items() if c % P != 0}
        polys.append(poly)
    return P, n, k, polys

def main():
    testId = sys.argv[1] if len(sys.argv) > 1 else "1"
    p, n, k, polys = build(testId)
    out = []
    out.append("%d %d %d" % (p, n, k))
    for poly in polys:
        items = sorted(poly.items())
        out.append(str(len(items)))
        for mono, c in items:
            exps = [0] * n
            for v in mono:
                exps[v] += 1
            out.append(str(c % p) + " " + " ".join(map(str, exps)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
