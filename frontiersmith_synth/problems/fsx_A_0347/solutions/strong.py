# TIER: strong
# Two-phase construction:
#   (1) greedy max-gain: build A one offset at a time, each time adding the offset
#       that creates the most NEW pairwise sums (a Mian-Chowla / B_2-flavoured rule);
#   (2) seeded hill-climb polish: swap one offset in for one out whenever it does not
#       shrink the sumset, escaping the greedy's local structure.
# Beats the random-restart heuristic and the AP baseline, but stays well short of the
# (unreachable) perfect-Sidon ceiling -> genuine headroom.
import sys
import random


def ss(A):
    s = set()
    for x in A:
        for y in A:
            s.add(x + y)
    return len(s)


def greedy_maxgain(k, M):
    A = []
    cur = set()
    inA = set()
    cand = list(range(M + 1))
    while len(A) < k:
        best = None
        bg = -1
        for x in cand:
            if x in inA:
                continue
            news = {x + a for a in A}
            news.add(2 * x)
            g = len(news - cur)
            if g > bg:
                bg = g
                best = x
        A.append(best)
        inA.add(best)
        for a in A:
            cur.add(best + a)
        cur.add(2 * best)
    return A


def main():
    tok = sys.stdin.read().split()
    k, M = int(tok[0]), int(tok[1])
    A = set(greedy_maxgain(k, M))
    cur = ss(A)
    rng = random.Random(777 + 17 * k)
    universe = list(range(M + 1))
    iters = 1800
    for _ in range(iters):
        out_el = rng.choice(list(A))
        add_el = rng.choice(universe)
        if add_el in A:
            continue
        B = set(A)
        B.discard(out_el)
        B.add(add_el)
        v = ss(B)
        if v >= cur:
            A = B
            cur = v
    A = sorted(A)
    out = [str(len(A))] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
