# TIER: greedy
# Obvious recipe: grow ONE connected blob of cut cells (single-pass BFS from a
# fixed generic seed) until it uses up roughly the marks needed to hit the
# 1/3 area target once mirrored by the fold group.  No reasoning about the
# fold axes at all -- every image of this one blob is congruent to every
# other image, so however many pieces the unfolded sheet ends up with, they
# are all copies of the SAME shape.
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    N, t, V = int(data[0]), int(data[1]), int(data[2])

    target_area = N * N / 3.0
    M_budget = min(V, max(1, round(target_area / t)))

    r0 = max(2, min(N - 3, N // 5))
    c0 = max(2, min(N - 3, N - 3 - N // 8))
    bm = 2  # stays off the sheet boundary, but never reasons about the fold axes

    visited = {(r0, c0)}
    order = [(r0, c0)]
    dq = deque([(r0, c0)])
    while dq and len(order) < M_budget:
        r, c = dq.popleft()
        for nr, nc in ((r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)):
            if len(order) >= M_budget:
                break
            if bm <= nr < N - bm and bm <= nc < N - bm and (nr, nc) not in visited:
                visited.add((nr, nc))
                order.append((nr, nc))
                dq.append((nr, nc))

    marks = order[:M_budget]
    out = [str(len(marks))]
    for r, c in marks:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
