# TIER: invalid
# Repeats item 1 for every slot instead of a permutation -> must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    out = []
    for _ in range(n):
        out.append("1 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
