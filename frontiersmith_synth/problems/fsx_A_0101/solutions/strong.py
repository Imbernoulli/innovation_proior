# TIER: strong
# Multi-start randomised-order greedy (deterministic seeds) plus the lex order
# as one candidate; keep the largest resonance-free deployment found.  Random
# insertion orders escape the {0,1}^n plateau that lex greedy is stuck on.
import sys, random


def third(a, b, n):
    return tuple((-(a[i] + b[i])) % 3 for i in range(n))


def to_vec(idx, n):
    v = []
    for _ in range(n):
        v.append(idx % 3); idx //= 3
    return tuple(v)


def greedy(order, blocked, n):
    S = []; Sset = set(); forbidden = set(blocked)
    for v in order:
        if v in forbidden or v in Sset:
            continue
        for u in S:
            forbidden.add(third(u, v, n))
        S.append(v); Sset.add(v)
    return S


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); k = int(next(it))
    blocked = set()
    for _ in range(k):
        blocked.add(tuple(int(next(it)) for _ in range(n)))
    av = [to_vec(i, n) for i in range(3 ** n)]

    best = greedy(av, blocked, n)  # lex candidate
    seeds = 8 if n <= 6 else 4
    for s in range(seeds):
        rnd = random.Random(4242 + s)
        o = av[:]
        rnd.shuffle(o)
        S = greedy(o, blocked, n)
        if len(S) > len(best):
            best = S
    out = [str(len(best))] + [" ".join(map(str, v)) for v in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
