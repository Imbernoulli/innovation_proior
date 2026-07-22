# TIER: strong
# INSIGHT: the targets are unlabeled, so spend the free permutation to keep
# every drone ON ITS OWN STRUT RADIUS -- match each drone to the same-(x,y)
# target and let it move purely VERTICALLY.  Struts have distinct radii, so
# drones never swap columns and the crossing that fuels downwash disappears.
# The vertical moves are longer than the tempting radial swap, but the
# truncated cones stay empty, which the penalty rewards far more.
# Then DESCEND LAYER BY LAYER, highest target first (one wave per height
# band), so while a tall mover is far above a neighbouring strut they are
# not sharing the same tick.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); L = int(next(it)); S = int(next(it))
    W = int(next(it)); K = int(next(it))
    for _ in range(6):
        next(it)
    F0 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]
    F1 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]

    # match each drone to the target sharing its (x,y) column (same strut radius)
    from collections import defaultdict
    buckets = defaultdict(list)
    for j, (x, y, z) in enumerate(F1):
        buckets[(x, y)].append(j)

    assign = [-1] * N
    for i, (x, y, z) in enumerate(F0):
        cand = buckets.get((x, y))
        if cand:
            assign[i] = cand.pop()          # unique column -> unique target
        else:
            assign[i] = -1

    # repair any drone whose column had no target (should not happen here):
    # give it any still-free target.
    used = set(a for a in assign if a >= 0)
    free = [j for j in range(N) if j not in used]
    fi = 0
    for i in range(N):
        if assign[i] < 0:
            assign[i] = free[fi]; fi += 1

    # layered descent: wave by TARGET height, highest first -> wave 0
    tz = sorted(set(F1[assign[i]][2] for i in range(N)), reverse=True)
    rank = {z: min(r, W - 1) for r, z in enumerate(tz)}
    waves = [rank[F1[assign[i]][2]] for i in range(N)]

    out = ["%d %d" % (assign[i], waves[i]) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")

main()
