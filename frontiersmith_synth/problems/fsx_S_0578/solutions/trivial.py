# TIER: trivial
# Triplicate every comparator of a reference sorting network.  Deleting any single
# copy still leaves two -> trivially 1-fault-tolerant.  Size 3S = the checker's
# calibration baseline (ratio ~ 0.1): maximum redundancy, zero insight.
import sys


def batcher(n):
    net = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a = i + j; b = i + j + k
                        if a < n and b < n:
                            net.append((a, b))
            k //= 2
        p *= 2
    return net


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = batcher(n)
    out = []
    for (a, b) in P:
        out.append((a, b)); out.append((a, b)); out.append((a, b))
    lines = [str(len(out))] + ["%d %d" % (a, b) for (a, b) in out]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
