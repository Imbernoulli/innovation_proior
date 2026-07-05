# TIER: trivial
# Reproduces the checker baseline: emit the largest single unobstructed line.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    b = int(next(it))
    blocked = set()
    for _ in range(b):
        r = int(next(it)); c = int(next(it))
        blocked.add((r, c))

    best = []
    for r in range(m):
        line = [(r, c) for c in range(m) if (r, c) not in blocked]
        if len(line) > len(best):
            best = line
    for c in range(m):
        line = [(r, c) for r in range(m) if (r, c) not in blocked]
        if len(line) > len(best):
            best = line

    out = [str(len(best))]
    for (r, c) in best:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
