# TIER: greedy
"""Greedy tier: a short, low-restart simulated-annealing search.

Because the arithmetic-progression plateau (R = 1) is wide and the more-sums-than-
differences 'islands' (R > 1) are isolated, a pure hill-climb from the AP never moves
(no single swap improves R). This tier therefore uses a light SA (2 restarts, warm
temperature) that crosses one valley into a nearby weak MSTD layout, typically reaching
R ~ 1.01-1.05. It underperforms the strong tier's deeper multi-restart search."""
import sys
import math
import random


def ratio(A):
    P = set()
    D = set()
    for a in A:
        for b in A:
            P.add(a + b)
            D.add(a - b)
    return len(P) / len(D)


def anneal(n, M, seed, restarts, L, Tm):
    rng = random.Random(seed)
    U = list(range(M + 1))
    d = max(1, M // (n - 1))
    best = [i * d for i in range(n)]          # AP fallback (always feasible)
    br = ratio(best)
    for rs in range(restarts):
        cur = set(rng.sample(U, n))
        cr = ratio(sorted(cur))
        for it in range(L):
            S = set(cur)
            out = rng.choice(list(S))
            S.discard(out)
            cands = [v for v in U if v not in S]
            v = rng.choice(cands)
            S.add(v)
            nr = ratio(sorted(S))
            T = Tm * (1 - it / L) + 0.001
            if nr >= cr or rng.random() < math.exp((nr - cr) / T):
                cur = S
                cr = nr
                if nr > br:
                    br = nr
                    best = sorted(S)
    return best


def main():
    toks = sys.stdin.read().split()
    n, M, seed = int(toks[0]), int(toks[1]), int(toks[2])
    A = anneal(n, M, seed, restarts=2, L=5000, Tm=0.5)
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
