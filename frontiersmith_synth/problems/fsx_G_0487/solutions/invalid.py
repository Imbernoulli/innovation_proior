# TIER: invalid
# Emits an infeasible schedule: a solid k x k block, which is packed with conflict corners
# (e.g. (0,0),(1,0),(0,1) with d=1). Must score exactly 0.
import sys


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); nb = int(next(it))
    for _ in range(nb):
        next(it); next(it)
    return N


def main():
    N = read_instance()
    k = min(N, 5)
    cells = [(r, c) for r in range(k) for c in range(k)]
    out = [str(len(cells))]
    for (r, c) in cells:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


main()
