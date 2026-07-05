# TIER: strong
# Weight-STEERED priority construction (the FunSearch template): process routes
# in descending throughput order and greedily deploy each one that does not
# create an interfering triple.  Then run a handful of weight-biased perturbed
# restarts and keep the best backbone found.  Far better than weight-blind
# selection, yet still only a heuristic -- the true max-weight cap is unknown,
# leaving headroom above this score.
import sys, random


def coords_of(i, n):
    c = [0] * n
    for k in range(n - 1, -1, -1):
        c[k] = i % 3
        i //= 3
    return c


def build(order, coords, n, w):
    forbidden = set()
    chosen = []
    chosen_set = set()
    tot = 0
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
        tot += w[i]
    return chosen, tot


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    N = 3 ** n
    w = list(map(int, toks[1:1 + N]))
    coords = [coords_of(i, n) for i in range(N)]
    wmax = max(w) if w else 1

    # primary: strict descending-weight priority
    order = sorted(range(N), key=lambda i: -w[i])
    best, best_w = build(order, coords, n, w)

    # a few weight-biased perturbed restarts
    R = 30 if n <= 6 else 12
    for s in range(R):
        rnd = random.Random(s * 97 + 3)
        jitter = 0.25 * wmax
        order = sorted(range(N), key=lambda i: -(w[i] + rnd.uniform(-jitter, jitter)))
        ch, tw = build(order, coords, n, w)
        if tw > best_w:
            best_w = tw
            best = ch

    out = [str(len(best))]
    for i in best:
        out.append("".join(str(x) for x in coords[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
