# TIER: greedy
# The obvious "more ventilation is better" recipe: spend the whole budget on
# the B vents with the largest conductance. On trap cases this fills the hall
# with mid-height leaks and kills the stack effect it depends on.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    W = int(next(it))
    B = int(next(it))
    _k0 = next(it)
    for _ in range(3 * W):
        next(it)
    D = int(next(it))
    for _ in range(4 * D):
        next(it)
    M = int(next(it))
    cands = []
    for _ in range(M):
        _c = int(next(it))
        _z = int(next(it))
        g = int(next(it))
        cands.append(g)
    order = sorted(range(M), key=lambda j: (-cands[j], j))
    take = sorted(order[:B])
    sys.stdout.write("%d\n%s\n" % (len(take), " ".join(str(j + 1) for j in take)))


main()
