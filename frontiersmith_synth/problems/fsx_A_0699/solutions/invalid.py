# TIER: invalid
# Looks plausible (same ring partition + colour alternation as trivial) but uses
# an out-of-range colour id (p+1), which must be rejected by feasibility checks.
import sys


def main():
    data = sys.stdin.read().split("\n")
    R, A, k, p = [int(x) for x in data[0].split()]
    out = []
    for r in range(R):
        color = p + 1  # always out of [1,p]
        for a in range(A):
            out.append("%d %d" % (r, color))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
