#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for a photoresist-mask
optical-proximity-correction (OPC) instance.

Fixed, known optical model (same on every test case -- stated in statement.md):
  - separable "partial coherence" blur kernel, radius 2, weights A=[1,2,3,2,1]
    (kernel(dx,dy) = A[dx+2]*A[dy+2], sum over the 5x5 window = 81).
  - intensity I(x,y) = sum over the mask's 5x5 window of mask_bit * kernel(dx,dy)
    (cells outside the grid contribute 0 -- unexposed).
  - three FIXED dose thresholds T = (33, 41, 49); a pixel PRINTS at dose T iff
    I(x,y) >= T (positive-tone resist).

Objective (process-window pattern fidelity):
  For a candidate mask, let IoU_T = |printed_T ^ target| / |printed_T v target|
  (Jaccard overlap) at each dose T.  F = 0.4*mean(IoU_T) + 0.6*min(IoU_T) -- the
  0.6 weight on the worst dose encodes DOSE LATITUDE: a mask that only works at
  one dose is punished even if its average looks fine.

Baseline B = F computed for the "obvious" mask := the target pattern itself
(drawing exactly the shape you want). B is always > 0 for our instances (every
target feature reaches at least one dose threshold under identity printing).
  Ratio = min(1000, 100*F/max(1e-9,B)) / 1000
"""
import sys

A = [1, 2, 3, 2, 1]  # kernel weights, radius 2 (index 0..4 <-> offset -2..2)
DOSES = (33, 41, 49)
MEAN_WEIGHT = 0.4  # F = MEAN_WEIGHT*mean(IoU) + (1-MEAN_WEIGHT)*min(IoU)


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty input")
    try:
        n = int(toks[0])
    except ValueError:
        fail("bad N in input")
    if len(toks) < 1 + n:
        fail("input truncated")
    grid = []
    for i in range(n):
        row = toks[1 + i]
        if len(row) != n or any(c not in "01" for c in row):
            fail("malformed input row (harness bug, should not happen)")
        grid.append([int(c) for c in row])
    return n, grid


def read_output(path, n):
    try:
        with open(path) as f:
            content = f.read()
    except Exception:
        fail("cannot read output")
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    # allow (and ignore) purely-whitespace trailing lines beyond N, but require
    # AT LEAST N well-formed lines and no non-blank junk after them.
    if len(lines) < n:
        fail("too few output lines: need %d, got %d" % (n, len(lines)))
    extra = lines[n:]
    if any(l.strip() != "" for l in extra):
        fail("extra non-blank output after row %d" % n)
    grid = []
    for i in range(n):
        row = lines[i]
        if len(row) != n:
            fail("row %d has length %d, expected %d" % (i, len(row), n))
        if any(c not in "01" for c in row):
            fail("row %d has a character outside {0,1}" % i)
        grid.append([int(c) for c in row])
    return grid


def intensity_grid(mask, n):
    """Exact-integer 5x5 separable-kernel convolution (out-of-grid = 0)."""
    inten = [[0] * n for _ in range(n)]
    for x in range(n):
        for dx in range(-2, 3):
            xx = x + dx
            if xx < 0 or xx >= n:
                continue
            wx = A[dx + 2]
            row = mask[xx]
            for y in range(n):
                s = 0
                lo = max(0, y - 2)
                hi = min(n - 1, y + 2)
                for yy in range(lo, hi + 1):
                    s += row[yy] * A[yy - y + 2]
                inten[x][y] += s * wx
    return inten


def printed_at(inten, n, t):
    return [[1 if inten[x][y] >= t else 0 for y in range(n)] for x in range(n)]


def iou(p, target, n):
    inter = 0
    union = 0
    for x in range(n):
        prow = p[x]
        trow = target[x]
        for y in range(n):
            pv = prow[y]
            tv = trow[y]
            if pv and tv:
                inter += 1
            if pv or tv:
                union += 1
    return inter / union if union > 0 else 1.0


def fidelity(mask, target, n):
    inten = intensity_grid(mask, n)
    ious = [iou(printed_at(inten, n, t), target, n) for t in DOSES]
    return MEAN_WEIGHT * (sum(ious) / len(ious)) + (1.0 - MEAN_WEIGHT) * min(ious)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, target = read_input(in_path)
    mask = read_output(out_path, n)

    F = fidelity(mask, target, n)
    B = fidelity(target, target, n)  # the "obvious" baseline: mask == target

    if F != F or F in (float("inf"), float("-inf")):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
