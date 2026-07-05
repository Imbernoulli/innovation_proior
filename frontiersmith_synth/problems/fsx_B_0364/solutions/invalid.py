# TIER: invalid
# Emits splices in order but performs NO swaps -- distant channel pairs are not adjacent,
# so the checker's adjacency gate fails and the score is 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); e = int(next(it)); q = int(next(it))
    for _ in range(e):
        next(it); next(it)
    for _ in range(n):
        next(it)
    out = []
    for k in range(1, q + 1):
        out.append("G %d" % k)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
