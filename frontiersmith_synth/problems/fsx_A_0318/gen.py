#!/usr/bin/env python3
"""gen.py <testId>  -> one instance to stdout.

Forest-fire watchtower ALERT tensor. Difficulty ladder testId 1..10.
We plant a low-rank structure along ONE (test-dependent) mode of a 3D integer
tensor, so that a good decomposition exists but the true tensor rank stays
genuinely open (planted slice-rank 2 => cross-slice combinations may do better;
tensor rank is NP-hard, no proven optimum).

Format (stdin the solver reads):
    line 1:  a b c
    next a*b lines:  for i in 0..a-1, j in 0..b-1  ->  c integers = T[i][j][0..c-1]
"""
import sys, random


def build(testId):
    rng = random.Random(700 + testId)
    n = 4 + (testId - 1) // 3          # 4,4,4,5,5,5,6,6,6,7
    a = b = c = n
    pm = (testId - 1) % 3 + 1          # planted low-rank mode: 1,2,3 cycling
    rho = 2                            # planted slice rank (>=2 keeps rank open)
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    vals = [-3, -2, -1, 1, 2, 3]

    def rm(rows, cols):
        return [[rng.choice(vals) for _ in range(cols)] for _ in range(rows)]

    if pm == 3:
        for k in range(c):
            P = rm(a, rho); Q = rm(rho, b)
            for i in range(a):
                for j in range(b):
                    T[i][j][k] = sum(P[i][t] * Q[t][j] for t in range(rho))
    elif pm == 1:
        for i in range(a):
            P = rm(b, rho); Q = rm(rho, c)
            for j in range(b):
                for k in range(c):
                    T[i][j][k] = sum(P[j][t] * Q[t][k] for t in range(rho))
    else:  # pm == 2
        for j in range(b):
            P = rm(a, rho); Q = rm(rho, c)
            for i in range(a):
                for k in range(c):
                    T[i][j][k] = sum(P[i][t] * Q[t][k] for t in range(rho))
    return a, b, c, T


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if testId < 1:
        testId = 1
    if testId > 10:
        testId = 10
    a, b, c, T = build(testId)
    out = ["%d %d %d" % (a, b, c)]
    for i in range(a):
        for j in range(b):
            out.append(" ".join(str(T[i][j][k]) for k in range(c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
