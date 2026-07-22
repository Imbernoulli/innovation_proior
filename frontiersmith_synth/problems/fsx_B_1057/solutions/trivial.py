# TIER: trivial
"""Always bypass. Never puts anything on the shelf, so every visit is a genuine
miss under this policy's own play -- reproduces the checker's own baseline B."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); C = int(next(it))
    for _ in range(K):
        next(it)
    seq = [int(next(it)) for _ in range(N)]
    out = []
    for _ in seq:
        out.append("B")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
