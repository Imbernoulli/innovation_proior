# TIER: greedy
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    next(it); next(it)  # ALPHA, LAMBDA -- unused by this strategy
    pts = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    for _ in range(N * N):
        next(it)  # bonus matrix -- unused: the obvious textbook approach

    # Classic single-linkage: always merge the two currently-closest clusters
    # by centroid distance. Oblivious to the bonus matrix (partition quality)
    # and to the horizon-weighted schedule (it merges cheap pairs whenever
    # they appear, front-loading the low-weight slots with whatever happens
    # to be nearest right now, not with what SHOULD be done early).
    info = {}
    for i in range(1, N + 1):
        x, y = pts[i - 1]
        info[i] = (1, float(x), float(y))
    active = list(range(1, N + 1))
    merges = []
    next_id = N + 1
    while len(active) > K:
        best = None
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                cntA, sxA, syA = info[a]
                cntB, sxB, syB = info[b]
                cxA, cyA = sxA / cntA, syA / cntA
                cxB, cyB = sxB / cntB, syB / cntB
                d2 = (cxA - cxB) ** 2 + (cyA - cyB) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, a, b)
        _, a, b = best
        merges.append((a, b))
        cntA, sxA, syA = info[a]
        cntB, sxB, syB = info[b]
        new_id = next_id
        next_id += 1
        info[new_id] = (cntA + cntB, sxA + sxB, syA + syB)
        active.remove(a); active.remove(b); active.append(new_id)

    out = [str(len(merges))]
    for a, b in merges:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
