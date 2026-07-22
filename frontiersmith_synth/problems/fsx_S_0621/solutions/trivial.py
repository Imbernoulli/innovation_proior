# TIER: trivial
# Static plan: multiply the UNION of every block that is nonzero in ANY pattern,
# perform NO tests.  This is exactly the checker's baseline construction, so it
# scores ~0.1.  It pays full price on every pattern regardless of its sparsity.
import sys

def main():
    data = sys.stdin.read().split('\n')
    B, P, M = map(int, data[0].split())
    union = set()
    for p in range(P):
        row = data[1 + p]
        for i, ch in enumerate(row):
            if ch == '1':
                union.add(i)
    U = sorted(union)
    out = []
    n = len(U)
    # M-chain of the whole union, then HALT (index n).
    for i, blk in enumerate(U):
        out.append("M %d %d" % (blk, i + 1))
    out.append("H")
    sys.stdout.write("%d\n" % len(out))
    sys.stdout.write("\n".join(out) + "\n")

main()
