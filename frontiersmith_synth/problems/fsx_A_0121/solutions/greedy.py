# TIER: greedy
# Randomized-order greedy construction (seeded, best-of-K).  Repeatedly shuffles the
# profile universe and greedily adds a profile whenever it introduces no resonance;
# keeps the largest catalogue found.  A generic "design an ordering" heuristic.
import sys, itertools, random


def build(n, order):
    S = set()
    lst = []
    for v in order:
        ok = True
        for a in lst:
            c = tuple((-(a[k] + v[k])) % 3 for k in range(n))
            if c in S:
                ok = False
                break
        if ok:
            S.add(v)
            lst.append(v)
    return lst


def main():
    n = int(sys.stdin.read().split()[0])
    V = [tuple(p) for p in itertools.product(range(3), repeat=n)]
    iters = {4: 100, 5: 60, 6: 30, 7: 12, 8: 6}.get(n, 8)
    rng = random.Random(20240624 + n)
    best = []
    for _ in range(iters):
        order = V[:]
        rng.shuffle(order)
        s = build(n, order)
        if len(s) > len(best):
            best = s
    out = [str(len(best))]
    for r in best:
        out.append(" ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
