#!/usr/bin/env python3
"""Deterministic scorer for ale-15 "Continuous Facility Layout".

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--energy]

Reads the instance and a candidate solution, validates feasibility, and prints
a single number on stdout:

  * default            -> the ALE-Bench continuous SCORE (higher is better,
                          0.0 if the solution is infeasible).
  * with --energy flag -> the raw objective ENERGY (the thing to MINIMIZE). On
                          an infeasible solution the script exits non-zero so
                          callers must use the default score for the floor.

Instance format (stdin of the solver):
    line 1:  N W H
    next N lines:  w h     (width, height of rectangle i, 1-based)

Solution format (the solver's stdout):
    N lines, line i:  x y   (integer bottom-left corner of rectangle i)

Feasibility requirements (any violation => SCORE 0):
    * exactly N coordinate pairs are present
    * every x, y parses as an integer
    * the rectangle stays fully inside the container:
          0 <= x  and  x + w <= W
          0 <= y  and  y + h <= H

Objective (to MINIMIZE):
    energy = OVERLAP_WEIGHT   * (total pairwise overlap area)
           + DISPERSION_WEIGHT * (sum over rects of squared distance of the
                                  rectangle centre from the mean centre)

    overlap area of rects i,j = max(0, overlap_x) * max(0, overlap_y) where
        overlap_x = min(xi+wi, xj+wj) - max(xi, xj)
        overlap_y = min(yi+hi, yj+hj) - max(yi, yj)
    summed over all unordered pairs i<j.

ALE score (to MAXIMIZE; this is what the judge reports):
    score = 0.0                                      if infeasible
    score = round(SCORE_SCALE / (1 + energy / N))    otherwise
  Lower energy -> higher score; an empty / out-of-container / malformed output
  floors the score to exactly 0.0.
"""
import sys

SCORE_SCALE = 1_000_000_000.0   # fixed normalisation constant (frozen)
OVERLAP_WEIGHT = 1.0            # weight on total pairwise overlap area
DISPERSION_WEIGHT = 1.0e-4     # weight on the dispersion (compactness) term


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    W = int(next(it))
    H = int(next(it))
    rects = []
    for _ in range(N):
        w = int(next(it))
        h = int(next(it))
        rects.append((w, h))
    return N, W, H, rects


def parse_solution(path, N, W, H, rects):
    """Return list of (x,y) integer positions, or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != 2 * N:
        return None
    pos = []
    for i in range(N):
        try:
            x = int(toks[2 * i])
            y = int(toks[2 * i + 1])
        except (ValueError, IndexError):
            return None
        w, h = rects[i]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return None  # rectangle leaves the container -> infeasible
        pos.append((x, y))
    return pos


def compute_energy(N, rects, pos):
    # Dispersion term: sum of squared distances of centres from the mean centre.
    cx = [pos[i][0] + rects[i][0] * 0.5 for i in range(N)]
    cy = [pos[i][1] + rects[i][1] * 0.5 for i in range(N)]
    mx = sum(cx) / N
    my = sum(cy) / N
    dispersion = 0.0
    for i in range(N):
        ddx = cx[i] - mx
        ddy = cy[i] - my
        dispersion += ddx * ddx + ddy * ddy

    # Overlap term: total pairwise overlap area (O(N^2) reference computation).
    overlap = 0.0
    for i in range(N):
        xi, yi = pos[i]
        wi, hi = rects[i]
        xi2 = xi + wi
        yi2 = yi + hi
        for j in range(i + 1, N):
            xj, yj = pos[j]
            wj, hj = rects[j]
            ox = min(xi2, xj + wj) - max(xi, xj)
            if ox <= 0:
                continue
            oy = min(yi2, yj + hj) - max(yi, yj)
            if oy <= 0:
                continue
            overlap += ox * oy

    return OVERLAP_WEIGHT * overlap + DISPERSION_WEIGHT * dispersion


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_energy = "--energy" in sys.argv[1:]
    if len(args) < 2:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--energy]\n")
        sys.exit(2)
    inst_path, sol_path = args[0], args[1]
    N, W, H, rects = read_instance(inst_path)
    pos = parse_solution(sol_path, N, W, H, rects)
    if pos is None:
        if want_energy:
            sys.stderr.write("infeasible\n")
            sys.exit(1)
        print(0.0)
        return
    energy = compute_energy(N, rects, pos)
    if want_energy:
        print(repr(energy))
        return
    score = round(SCORE_SCALE / (1.0 + energy / N))
    print(score)


if __name__ == "__main__":
    main()
