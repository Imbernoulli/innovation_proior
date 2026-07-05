# TIER: trivial
# Weight-BLIND greedy in canonical base-3 order: deploy routes left-to-right,
# skipping any route that would complete an interfering triple.  This reproduces
# the checker's baseline (it just returns the 0/1 sub-cube), so it scores ~0.1.
import sys


def coords_of(i, n):
    c = [0] * n
    for k in range(n - 1, -1, -1):
        c[k] = i % 3
        i //= 3
    return c


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    N = 3 ** n
    coords = [coords_of(i, n) for i in range(N)]
    forbidden = set()
    chosen = []
    chosen_set = set()
    for i in range(N):
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
    out = [str(len(chosen))]
    for i in chosen:
        out.append("".join(str(x) for x in coords[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
