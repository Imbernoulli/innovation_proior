# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    W = int(next(it)); H = int(next(it))
    A = int(next(it))
    anchors = [int(next(it)) for _ in range(A)]
    L = int(next(it))
    loads = []
    for _ in range(L):
        r = int(next(it)); c = int(next(it)); f = int(next(it))
        loads.append((r, c, f))
    M = int(next(it)); K = int(next(it))

    # Deliberately infeasible: grow a straight column under the first mound
    # but stop one cell short of its row, so that mound is never covered by
    # material at all (must score Ratio: 0.0).
    r_l, c_l, f_l = loads[0]
    a = anchors[0]
    cells = [(r, a) for r in range(1, max(1, r_l - 1))]
    print(len(cells))
    out = [f"{r} {c}" for (r, c) in cells]
    if out:
        print("\n".join(out))

main()
