# TIER: greedy
# Randomized multi-start: build several interference-free backbones from RANDOM
# route orderings (order is weight-blind) and keep the one of highest total
# throughput.  Uses the weights only to SELECT among random caps, not to steer
# construction -- so it beats the blind baseline but stays well short of a
# weight-steered heuristic.
import sys, random


def coords_of(i, n):
    c = [0] * n
    for k in range(n - 1, -1, -1):
        c[k] = i % 3
        i //= 3
    return c


def build(order, coords, n):
    forbidden = set()
    chosen = []
    chosen_set = set()
    for i in order:
        if i in forbidden or i in chosen_set:
            continue
        ci = coords[i]
        for q in chosen:
            cq = coords[q]
            r = 0
            for k in range(n):
                r = r * 3 + ((-(ci[k] + cq[k])) % 3)
            forbidden.add(r)
        chosen.append(i)
        chosen_set.add(i)
    return chosen


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    N = 3 ** n
    w = list(map(int, toks[1:1 + N]))
    coords = [coords_of(i, n) for i in range(N)]

    R = 24 if n <= 6 else 12
    best = None
    best_w = -1
    for s in range(R):
        rnd = random.Random(s * 31 + 5)
        order = list(range(N))
        rnd.shuffle(order)
        ch = build(order, coords, n)
        tw = sum(w[i] for i in ch)
        if tw > best_w:
            best_w = tw
            best = ch

    out = [str(len(best))]
    for i in best:
        out.append("".join(str(x) for x in coords[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
