# TIER: invalid
# Emits a well-formed tree whose rods are NOT in torque balance (all arms = 1),
# so wL*aL == wR*aR fails at internal nodes -> the feasibility gate must score 0.
import sys


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); ws = [int(x) for x in t[2:2 + N]]
    # left-chain caterpillar, every arm forced to 1 (violates balance in general)
    lines = []
    M = 2 * N - 1
    # node ids: leaves 0..N-1 in order, internals N..2N-2 chaining them
    # build serialization: post-order-ish. We'll just emit leaves then internals.
    # internal i (>=N) combines (prev subtree, leaf) -- but to keep it simple and
    # guaranteed non-integer-balance, use arms (1,1) everywhere.
    # Structure: node N = I(leaf0, leaf1); node N+k = I(node N+k-1, leaf k+1)
    out = [str(M)]
    for w in ws:
        out.append("L %d" % w)
    # first internal combines leaf0,leaf1
    out.append("I 0 1 1 1")
    for k in range(2, N):
        parent_prev = N + (k - 2)
        out.append("I %d %d 1 1" % (parent_prev, k))
    sys.stdout.write("\n".join(out) + "\n")


main()
