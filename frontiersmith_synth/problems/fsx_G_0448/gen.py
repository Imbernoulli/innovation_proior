#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance of the BILINEAR MULTIPLICATION-SPLIT
problem (format D, family flop-integer-mult-splits, theme: bignum library).

Setting.  A big-number library needs a fixed "limb-combination kernel": given two
short limb vectors x = (x_0..x_{p-1}) and y = (y_0..y_{q-1}), it must evaluate r
prescribed BILINEAR forms

        b_k(x,y) = sum_{i,j} T[k][i][j] * x_i * y_j        (k = 0 .. r-1)

simultaneously.  The single-digit (scalar) MULTIPLICATIONS x_a*y_b are the costly
operations; additions and multiplications by fixed small constants are free (they
are cheap shifts/adds on limbs).  The task is to compute all r forms using as FEW
scalar products of the shape  (linear form in x) * (linear form in y)  as possible
-- exactly the Karatsuba / Toom "few multiplications" objective, generalized to an
arbitrary fixed coefficient tensor T.

The tensor T is a FIXED, PUBLISHED integer 3-tensor with an overcomplete internal
structure: its true bilinear rank (the minimum number of scalar products) is NOT
known in closed form, so there is no reachable optimal formula -- only better and
better splits.

Difficulty ladder (testId 1..N): the limb widths p,q,r and the planted term count
grow, enlarging the tensor and widening the gap between naive and clever splits.

STDOUT (the instance):
    line 1:  p q r
    then r blocks, each of p lines of q integers  =  T[k][i][0..q-1]
The internal planted factorization is NEVER printed; only the assembled tensor T.
"""
import sys
import random


def gen_instance(testId):
    rng = random.Random(1000 + testId * 7919)
    p = 4 + (testId % 4)          # 4..7
    q = 4 + ((testId + 2) % 4)    # 4..7
    r = 3 + (testId % 3)          # 3..5
    R0 = p + q + testId           # planted (overcomplete) term count
    ndict = 2 + (testId % 2)      # 2..3 distinct x-patterns -> low x-mode structure

    udict = []
    for _ in range(ndict):
        u = [rng.choice([-1, 0, 0, 1]) for _ in range(p)]
        if all(x == 0 for x in u):
            u[rng.randrange(p)] = 1
        udict.append(u)

    T = [[[0] * q for _ in range(p)] for _ in range(r)]
    for _ in range(R0):
        u = udict[rng.randrange(ndict)]
        v = [rng.choice([-1, 0, 0, 1]) for _ in range(q)]
        if all(x == 0 for x in v):
            v[rng.randrange(q)] = 1
        w = [rng.choice([-1, 0, 1]) for _ in range(r)]
        if all(x == 0 for x in w):
            w[rng.randrange(r)] = 1
        for k in range(r):
            if w[k] == 0:
                continue
            for i in range(p):
                if u[i] == 0:
                    continue
                for j in range(q):
                    if v[j] == 0:
                        continue
                    T[k][i][j] += w[k] * u[i] * v[j]
    return p, q, r, T


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p, q, r, T = gen_instance(testId)
    out = [f"{p} {q} {r}"]
    for k in range(r):
        for i in range(p):
            out.append(" ".join(str(T[k][i][j]) for j in range(q)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
