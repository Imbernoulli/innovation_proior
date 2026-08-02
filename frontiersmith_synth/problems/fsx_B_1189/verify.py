#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Echo Room
problem (fsx_B_1189, format C). Prints "... Ratio: <float in [0,1]>".

The participant's artifact is W wall LINES (each two points). We reflect
the known source S across each submitted line to get a candidate image
point, forward-simulate first-order echo times from those candidate image
points at TWO held-out microphone positions (never shown to the solver --
re-derived here from the same seeded construction as gen.py), and score by
how well the sorted predicted times match the sorted true times there.
A room that only fits the given microphones (e.g. a mirrored/rank-swapped
mislabeling) but was not built from geometrically consistent wall lines
will predict badly at the held-out microphones even if it looked perfect
on every microphone it was allowed to see.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roomgeo

SCALE = 6.0          # MAE (in coordinate units) that maps to zero quality
MIN_LINE_LEN = 1e-3  # the two points defining a wall line must differ by this much
MIN_WALL_DIST = 0.05  # source must be at least this far from a submitted wall line
MAX_WALL_DIST = 60.0
MIN_SEP = 0.05        # distinctness floor between two candidate image points
COORD_BOUND = 1e4
BASE_R0 = 1.0
BASE_ANGLE_OFFSET = 0.15


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_floats(tokens, need):
    vals = []
    for t in tokens[:need]:
        try:
            v = float(t)
        except ValueError:
            return None
        if not math.isfinite(v):
            return None
        vals.append(v)
    if len(vals) != need:
        return None
    return vals


def baseline_image_points(S, W):
    pts = []
    for k in range(W):
        theta = 2 * math.pi * k / W + BASE_ANGLE_OFFSET
        pts.append((S[0] + BASE_R0 * math.cos(theta), S[1] + BASE_R0 * math.sin(theta)))
    return pts


def mae_at_held(image_pts, held_mics, true_image_pts):
    total = 0.0
    for h in held_mics:
        pred = sorted(dist(p, h) for p in image_pts)
        true = sorted(dist(p, h) for p in true_image_pts)
        total += sum(abs(a - b) for a, b in zip(pred, true)) / len(true)
    return total / len(held_mics)


def quality(image_pts, held_mics, true_image_pts):
    mae = mae_at_held(image_pts, held_mics, true_image_pts)
    return max(0.0, 1.0 - mae / SCALE)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        itoks = open(in_path).read().split()
    except Exception:
        fail("cannot read input")
    try:
        it = iter(itoks)
        W = int(next(it)); K = int(next(it)); test_id = int(next(it))
        S = (float(next(it)), float(next(it)))
        for _ in range(K):
            next(it); next(it)
            L = int(next(it))
            for _ in range(L):
                next(it)
    except Exception:
        fail("bad instance (should never happen)")

    # Re-derive ground truth (including the two held-out microphones this
    # .in file never printed) from the SAME deterministic construction
    # gen.py used. testId carries no geometric information by itself --
    # cross-check it reproduces this exact input before trusting it.
    if not (1 <= test_id <= 10):
        fail("bad testId in instance")
    truth = roomgeo.build(test_id)
    if truth["W"] != W or truth["K"] != K:
        fail("instance/testId mismatch (should never happen)")
    chk = ["%d %d %d" % (truth["W"], truth["K"], test_id),
           "%.9f %.9f" % (truth["S"][0], truth["S"][1])]
    for i in range(K):
        mx, my = truth["given_mics"][i]
        toks = ["%.9f" % mx, "%.9f" % my, str(len(truth["obs"][i]))]
        toks += ["%.9f" % t for t in truth["obs"][i]]
        chk.append(" ".join(toks))
    if "\n".join(chk) + "\n" != open(in_path).read():
        fail("instance does not match the deterministic generator (tampered .in?)")

    true_image_pts = truth["image_pts"]
    held_mics = truth["held_mics"]

    # ---- parse participant output ----
    try:
        otoks = open(out_path).read().split()
    except Exception:
        fail("no output")
    if not otoks:
        fail("empty output")
    try:
        M = int(otoks[0])
    except Exception:
        fail("bad wall count token")
    if M != W:
        fail("wall count %d != required %d" % (M, W))
    need = 1 + 4 * W
    if len(otoks) < need:
        fail("truncated output: need %d tokens, got %d" % (need, len(otoks)))
    if len(otoks) > need:
        fail("trailing garbage after expected %d tokens" % need)

    rest = parse_floats(otoks[1:], 4 * W)
    if rest is None:
        fail("non-finite or unparsable coordinate")
    for v in rest:
        if abs(v) > COORD_BOUND:
            fail("coordinate out of bounds: %g" % v)

    image_pts = []
    for k in range(W):
        x1, y1, x2, y2 = rest[4 * k:4 * k + 4]
        P1, P2 = (x1, y1), (x2, y2)
        if dist(P1, P2) < MIN_LINE_LEN:
            fail("wall %d: the two points are too close to define a line" % k)
        dx, dy = P2[0] - P1[0], P2[1] - P1[1]
        norm = math.hypot(dx, dy)
        nx, ny = -dy / norm, dx / norm  # unit normal
        sd = (S[0] - P1[0]) * nx + (S[1] - P1[1]) * ny  # signed distance S to line
        if abs(sd) < MIN_WALL_DIST:
            fail("wall %d passes too close to the source (d=%.6f)" % (k, abs(sd)))
        if abs(sd) > MAX_WALL_DIST:
            fail("wall %d absurdly far from the source (d=%.6f)" % (k, abs(sd)))
        # reflect S across the line: I = S - 2*sd*n
        I = (S[0] - 2 * sd * nx, S[1] - 2 * sd * ny)
        image_pts.append(I)

    for a in range(W):
        for b in range(a + 1, W):
            if dist(image_pts[a], image_pts[b]) < MIN_SEP:
                fail("walls %d and %d are duplicates (image points coincide)" % (a, b))

    F = quality(image_pts, held_mics, true_image_pts)
    B = quality(baseline_image_points(S, W), held_mics, true_image_pts)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print("quality=%.6f baseline=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
