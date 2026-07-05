#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance (a 3-D "synergy tensor") to stdout.

E-SPORTS ARENA skin (family: tensor-decomposition-rank, format D, AlphaEvolve-inspired):
The tournament scheduler must evaluate a 3-way synergy table T[i][j][k] = the bonus that
hero i, item j and map k contribute together. The engine evaluates it as a sum of
rank-1 "trilinear gadgets" (a_r . x)(b_r . y)(c_r . z); each gadget costs ONE scalar
multiply. We ship a synergy tensor whose FRONTAL slices (one per map) are each planted
as a LOW-RANK matrix, but the number of planted gadgets exceeds every dimension
(over-complete rank) so no polynomial diagonalization recovers the true optimum.

Deterministic: everything seeded by testId only. Integer entries.

STDOUT format:
    I J K
    then K frontal slices; slice k is I lines of J integers  (value = T[i][j][k]).
"""
import sys, random

# (I, J, K, s): K = smallest dim (fewest frontal slices), s = planted rank of each slice.
#   planted total rank R = s*K > max(I,J)  -> over-complete (Jennrich cannot recover)
#   s < min(I,J)                            -> slice factoring beats every fiber decomposition
PARAMS = {
    1:  (5, 5, 3, 2),
    2:  (5, 6, 4, 2),
    3:  (6, 6, 4, 2),
    4:  (6, 6, 3, 3),
    5:  (7, 6, 4, 2),
    6:  (7, 7, 4, 2),
    7:  (6, 7, 4, 3),
    8:  (7, 7, 4, 3),
    9:  (8, 7, 5, 2),
    10: (8, 8, 4, 3),
}


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in PARAMS:
        # clamp into range so the ladder is always defined
        t = ((t - 1) % len(PARAMS)) + 1
    I, J, K, s = PARAMS[t]
    rng = random.Random(90210 + 1000 * t)

    # T[i][j][k]
    T = [[[0] * K for _ in range(J)] for _ in range(I)]

    def nonzero_vec(n):
        while True:
            v = [rng.randint(-4, 4) for _ in range(n)]
            if any(x != 0 for x in v):
                return v

    # Each frontal slice k is the sum of s rank-1 integer outer products a_t (x) b_t.
    # -> slice matrix rank <= s (generically exactly s). Total planted gadgets = s*K.
    for k in range(K):
        for _t in range(s):
            a = nonzero_vec(I)
            b = nonzero_vec(J)
            for i in range(I):
                if a[i] == 0:
                    continue
                for j in range(J):
                    T[i][j][k] += a[i] * b[j]

    out = [f"{I} {J} {K}"]
    for k in range(K):
        for i in range(I):
            out.append(" ".join(str(T[i][j][k]) for j in range(J)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
