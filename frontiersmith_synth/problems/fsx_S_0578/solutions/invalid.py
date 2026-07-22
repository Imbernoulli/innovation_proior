# TIER: invalid
# Emits a PLAIN (minimal) sorting network with no redundancy: it sorts with zero
# faults, but deleting almost any comparator breaks the sort, so it fails the
# single-comparator deletion sweep -> the checker scores it 0.
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
    lines = [str(len(P))] + ["%d %d" % (a, b) for (a, b) in P]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
