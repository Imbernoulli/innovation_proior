import sys

# gen.py <testId>  -- prints ONE fault-tolerant-sorting-network instance to stdout.
# The instance is a single integer n (number of wires) plus S, the size of the
# judge's reference sorting network (Batcher odd-even mergesort). Difficulty grows
# with testId (n = testId + 2, so n runs 3..12 over the 10-case ladder).
#
# There is NO planted optimum: the minimum size of a 1-fault-tolerant sorting
# network is unknown in general. The obvious "duplicate every comparator" recipe
# is always valid at 2S comparators, but a strong solution amortizes redundancy
# with a shared mop-up stage that repairs many single-comparator failures at once.


def batcher(n):
    """Batcher's odd-even mergesort comparator network for n wires (any n>=2)."""
    net = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a = i + j
                        b = i + j + k
                        if a < n and b < n:
                            net.append((a, b))
            k //= 2
        p *= 2
    return net


def main():
    tid = int(sys.argv[1])
    n = tid + 2                      # testId 1..10 -> n = 3..12
    S = len(batcher(n))
    sys.stdout.write("%d %d\n" % (n, S))


if __name__ == "__main__":
    main()
