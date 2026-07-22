# TIER: greedy
# The obvious fault-tolerance recipe: DUPLICATE every comparator (place each one
# twice, back to back).  Deleting one copy leaves the other in the same spot -> the
# network is unchanged, so it is 1-fault-tolerant at exactly 2S comparators.  This
# is the documented TRAP: valid, but it pays one full redundant comparator per
# comparator and never shares fault coverage.
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
        out.append((a, b)); out.append((a, b))
    lines = [str(len(out))] + ["%d %d" % (a, b) for (a, b) in out]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
