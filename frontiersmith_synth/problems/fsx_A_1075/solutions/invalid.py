# TIER: invalid
"""Deliberately infeasible: declares only the m leaves and then points every query's
root at leaf 0, so almost every query's declared range ([0,1)) will not match its
required [L,R) -- the checker must reject this with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    m = int(data[p]); p += 1
    p += m + 1  # skip dims
    Q = int(data[p]); p += 1
    # queries themselves are irrelevant to this broken output

    out = [str(m)]
    out.extend("L %d" % i for i in range(m))
    out.append(" ".join("0" for _ in range(Q)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
