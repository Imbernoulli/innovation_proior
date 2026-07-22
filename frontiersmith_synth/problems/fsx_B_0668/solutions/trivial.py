# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    # we don't need the rest of the instance for the trivial schedule
    out = []
    for i in range(1, n + 1):
        out.append(f"{i} 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
