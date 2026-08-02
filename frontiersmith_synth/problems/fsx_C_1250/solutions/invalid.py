# TIER: invalid
# Emits a syntactically well-formed but out-of-range configuration (truncation depth
# beyond TMAX on every position) -- must be rejected by the checker's range check.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it)); G = int(next(it)); S = int(next(it))
    TMAX = int(next(it)); COMP_EXTRA = int(next(it))

    out = []
    for _ in range(K):
        out.append("%d 0" % (TMAX + 3))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
