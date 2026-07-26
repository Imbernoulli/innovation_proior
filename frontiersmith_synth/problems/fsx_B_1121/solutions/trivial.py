# TIER: trivial
"""Identity labeling: assign symbol i to slot i, exactly the order the slots
were printed in the input. No graph reasoning of any kind -- doesn't even
look at where the edges point."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    for _ in range(n):
        for _ in range(m):
            next(it)  # discard the targets, we never use them
    out = []
    for _v in range(n):
        out.append(" ".join(str(i) for i in range(m)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
