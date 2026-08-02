# TIER: trivial
"""One private, all-constant, single-use template per line -- explains
nothing, exactly reproduces the checker's own baseline B = N*W (Ratio ~0.1)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it))
    lines = [[next(it) for _ in range(W)] for _ in range(N)]

    out = [str(N)]
    for row in lines:
        out.append(" ".join(row))
    out.append(" ".join(str(i + 1) for i in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
