# TIER: strong
"""The insight: total cost is governed by the trace's JUMP-EPOCH structure,
not by raw touch count -- a burst of L nearby touches costs the same
whether L is 3 or 3000 to *detect*, but whether it is worth paying a
one-time linear relocation to serve it cheaply afterwards depends on L.
This reframes "where do I put my F bookmarks?" as CHANGEPOINT DETECTION
plus a per-segment rebuild-vs-hop decision:

  1. Segment the trace by finding real jump boundaries: compute the median
     absolute step between consecutive touches (the typical *within-epoch*
     jitter) and cut a new segment whenever a step exceeds a robust
     multiple of that median. This adapts to each instance instead of
     assuming fixed-size or evenly-spaced bursts.
  2. Walk the segments in order. For each one, compare (a) servicing it
     with whatever bookmarks are ALREADY placed (possibly a nearby earlier
     hub -- reuse is free, no relocation needed) against (b) relocating one
     currently-idle-or-least-useful bookmark to this segment's centroid and
     paying the linear relocation cost. Take whichever is cheaper. This
     naturally starves tiny/scattered segments of relocations (not worth
     the linear cost) while funding the segments whose LENGTH repays it,
     and lets revisited hubs reuse a bookmark left there earlier for free.

This is a decomposition (segment, then a local cost-benefit exchange), not
a fixed schedule -- it exploits both the log-vs-linear cost asymmetry and
the trace's actual (non-uniform) epoch structure.
"""
import sys


def hop_cost(a, b):
    return abs(a - b).bit_length() + 1


def reloc_cost(a, b):
    return abs(a - b) // 4 + 2


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def segment_trace(pts):
    M = len(pts)
    if M <= 1:
        return [(0, M)]
    steps = [abs(pts[i] - pts[i - 1]) for i in range(1, M)]
    steps.sort()
    med = steps[len(steps) // 2] if steps else 0
    thresh = max(50, med * 6)
    segs = []
    start = 0
    for i in range(1, M):
        if abs(pts[i] - pts[i - 1]) > thresh:
            segs.append((start, i))
            start = i
    segs.append((start, M))
    return segs


def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0]); M = int(data[1]); F = int(data[2])
    pts = [int(x) for x in data[3:3 + M]]

    C0 = (N + 1) // 2
    finger_pos = [C0] * (F + 1)
    placed = [False] * (F + 1)
    events = []

    segs = segment_trace(pts)
    for (s, e) in segs:
        seg_pts = pts[s:e]
        centroid = median(seg_pts)

        cur_best = []
        cur_best_sum = 0
        for p in seg_pts:
            best = hop_cost(C0, p)
            for j in range(1, F + 1):
                if placed[j]:
                    c = hop_cost(finger_pos[j], p)
                    if c < best:
                        best = c
            cur_best.append(best)
            cur_best_sum += best

        best_total = cur_best_sum
        best_choice = None
        for i in range(1, F + 1):
            rc = reloc_cost(finger_pos[i], centroid)
            if rc >= best_total:
                continue  # relocation alone already too expensive
            svc = 0
            for idx, p in enumerate(seg_pts):
                c = hop_cost(centroid, p)
                svc += c if c < cur_best[idx] else cur_best[idx]
            tot = rc + svc
            if tot < best_total:
                best_total = tot
                best_choice = i

        if best_choice is not None:
            events.append((s + 1, best_choice, centroid))
            finger_pos[best_choice] = centroid
            placed[best_choice] = True

    out = [str(len(events))]
    for (t, i, q) in events:
        out.append(f"{t} {i} {q}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
