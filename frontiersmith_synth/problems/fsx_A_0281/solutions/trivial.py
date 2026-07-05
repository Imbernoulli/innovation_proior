# TIER: trivial
# Baseline: put relays along the single unblocked row with the most free cells.
# A single grid row is always corner-free.  Reproduces the checker's baseline B.
import sys


def main():
    t = sys.stdin.read().split()
    i = 0
    n = int(t[i]); i += 1
    k = int(t[i]); i += 1
    blocked = set()
    for _ in range(k):
        blocked.add((int(t[i]), int(t[i + 1]))); i += 2

    best = []
    for y in range(n):
        row = [(x, y) for x in range(n) if (x, y) not in blocked]
        if len(row) > len(best):
            best = row

    out = [str(len(best))]
    for (x, y) in best:
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
