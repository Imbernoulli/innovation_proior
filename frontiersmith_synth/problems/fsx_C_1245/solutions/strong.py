# TIER: strong
"""The insight: minimizing bits per block MAXIMIZES how often the width
changes from block to block, and every change is charged C by the decoder's
branch predictor. So instead of chasing the tightest width block-by-block,
restrict the whole stream to a SMALL, REUSED palette of widths (quantize),
sized directly from the data's own magnitude distribution (1-D k-means over
each element's minimal bit-length), and only pick block boundaries that keep
each block within one palette width, extended as far as possible. Spending a
few extra bits per element inside a block (using a slightly-too-wide palette
width) buys a branch-free decode. We try a handful of palette sizes
(k = 1..5) built this way and keep whichever total cost is lowest -- this is
a constrained search over a small family of REGULAR width sequences, not an
unrestricted per-block-minimal search."""
import sys
from collections import deque


def kmeans1d(vals, k, iters=25):
    vals = sorted(vals)
    if k <= 1:
        return [max(vals)]
    if len(vals) <= k:
        return sorted(set(vals))
    centers = [vals[int(i * (len(vals) - 1) / (k - 1))] for i in range(k)]
    for _ in range(iters):
        buckets = [[] for _ in range(k)]
        for v in vals:
            bi = min(range(k), key=lambda ci: abs(v - centers[ci]))
            buckets[bi].append(v)
        newc = [ (sum(b) / len(b)) if b else centers[i] for i, b in enumerate(buckets) ]
        if newc == centers:
            break
        centers = newc
    buckets = [[] for _ in range(k)]
    for v in vals:
        bi = min(range(k), key=lambda ci: abs(v - centers[ci]))
        buckets[bi].append(v)
    # representative width per cluster = its MAX member, so the cluster stays feasible
    reps = sorted(set(max(b) for b in buckets if b))
    return reps


def reach_array(A, w):
    """reach[j] = farthest exclusive end i s.t. max(A[j:i]) - min(A[j:i]) <= 2**w - 1
    (== 0 required when w == 0). O(N) amortized two-pointer with monotonic deques."""
    N = len(A)
    cap = (1 << w) - 1 if w > 0 else 0
    reach = [0] * N
    dqmax, dqmin = deque(), deque()
    right = 0
    for left in range(N):
        if right < left:
            right = left
        while right < N:
            v = A[right]
            while dqmax and A[dqmax[-1]] <= v: dqmax.pop()
            dqmax.append(right)
            while dqmin and A[dqmin[-1]] >= v: dqmin.pop()
            dqmin.append(right)
            if A[dqmax[0]] - A[dqmin[0]] > cap:
                if dqmax[-1] == right: dqmax.pop()
                if dqmin[-1] == right: dqmin.pop()
                break
            right += 1
        reach[left] = right
        if dqmax and dqmax[0] == left: dqmax.popleft()
        if dqmin and dqmin[0] == left: dqmin.popleft()
    return reach


def dp_for_palette(N, H, C, A, palette):
    """Cheapest chain of 'commit to width w at position j, extend maximally'
    moves -- O(N * |palette|)."""
    reaches = {w: reach_array(A, w) for w in palette}
    f = [float("inf")] * (N + 1)
    fw = [None] * (N + 1)
    par = [None] * (N + 1)
    f[0] = 0.0
    for j in range(N):
        if f[j] == float("inf"):
            continue
        for w in palette:
            i = reaches[w][j]
            if i <= j:
                continue
            trans = C if (fw[j] is not None and fw[j] != w) else 0
            cand = f[j] + H + (i - j) * w + trans
            if cand < f[i] - 1e-9:
                f[i] = cand
                fw[i] = w
                par[i] = j
    if f[N] == float("inf"):
        return None, float("inf")
    blocks = []
    i = N
    while i > 0:
        j = par[i]
        blocks.append((i - j, fw[i]))
        i = j
    blocks.reverse()
    return blocks, f[N]


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it)); C = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    bit_lengths = [a.bit_length() for a in A]

    best = None
    for k in (1, 2, 3, 4, 5):
        palette = kmeans1d(bit_lengths, k)
        blocks, cst = dp_for_palette(N, H, C, A, palette)
        if blocks is None:
            continue
        if best is None or cst < best[1]:
            best = (blocks, cst)

    blocks, _ = best
    print(len(blocks))
    out = []
    for ln, w in blocks:
        out.append(f"{ln} {w}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
