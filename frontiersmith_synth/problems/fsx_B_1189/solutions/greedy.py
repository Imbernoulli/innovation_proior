# TIER: greedy
"""The obvious first attempt: classic two-microphone multilateration.
Take mic0 and mic1's reading lists, sort each ascending, keep the smallest
W entries of each (the "extra" entries, if any, are just assumed to be
mismeasurement and dropped), and assume the k-th smallest reading at mic0
is the SAME wall as the k-th smallest reading at mic1 (same sorted order
everywhere). For each paired (ra, rb) circle-circle intersection has (up
to) two solutions; always take the same fixed-handedness one, with no
cross-check against any other microphone. This ignores the
echo-labeling-combinatorics mechanism entirely -- it ONLY ever looks at
two microphones, so a rank-order flip elsewhere in the room, or a decoy
number in the smallest-W window, silently derails it."""
import sys
import math


def circle_intersect(Ma, Mb, ra, rb):
    ax, ay = Ma; bx, by = Mb
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    if d < 1e-9:
        d = 1e-9
    a = (ra * ra - rb * rb + d * d) / (2 * d)
    h2 = ra * ra - a * a
    h = math.sqrt(h2) if h2 > 0 else 0.0
    px, py = ax + a * dx / d, ay + a * dy / d
    perp_x, perp_y = -dy / d, dx / d
    # fixed handedness convention: always the "+h*perp" branch, never checked
    return (px + h * perp_x, py + h * perp_y)


def line_through_image(S, I):
    ix, iy = I[0] - S[0], I[1] - S[1]
    n = math.hypot(ix, iy)
    if n < 1e-6:
        ix, iy, n = 1.0, 0.0, 1.0
    mx, my = (S[0] + I[0]) / 2.0, (S[1] + I[1]) / 2.0
    perp_x, perp_y = -iy / n, ix / n
    return (mx + perp_x, my + perp_y, mx - perp_x, my - perp_y)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); K = int(next(it)); _tid = int(next(it))
    S = (float(next(it)), float(next(it)))
    mics = []
    for _ in range(K):
        mx = float(next(it)); my = float(next(it))
        L = int(next(it))
        readings = [float(next(it)) for _ in range(L)]
        mics.append(((mx, my), readings))

    (Ma, ra_list), (Mb, rb_list) = mics[0], mics[1]
    ra_sorted = sorted(ra_list)[:W]
    rb_sorted = sorted(rb_list)[:W]

    out = [str(W)]
    for k in range(W):
        I = circle_intersect(Ma, Mb, ra_sorted[k], rb_sorted[k])
        x1, y1, x2, y2 = line_through_image(S, I)
        out.append("%.9f %.9f %.9f %.9f" % (x1, y1, x2, y2))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
