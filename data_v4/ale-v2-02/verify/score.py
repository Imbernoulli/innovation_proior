#!/usr/bin/env python3
"""Deterministic local scorer for ale-v2-02 (Rectangle Strip Packing).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE

Reads the instance (W N, then N pairs w_i h_i) and the solution (N lines,
each "x_i y_i", the bottom-left corner of rectangle i in input order).

Feasibility (any violation -> score 0):
  * exactly N coordinate pairs, all integers;
  * 0 <= x_i  and  x_i + w_i <= W              (inside the strip, no rotation);
  * 0 <= y_i                                    (strip floor at y = 0);
  * no two rectangles overlap (interiors disjoint; touching edges allowed).

If feasible, let H = max_i (y_i + h_i) be the used height (H = 0 if N = 0).
A lower bound on any feasible height is
    LB = max( ceil(sum_i w_i*h_i / W),  max_i h_i ).
The score is the ratio of this lower bound to the achieved height, scaled:
    score = 100.0 * LB / H            (H > 0),
    score = 100.0                     (H == 0, i.e. N == 0).
Higher is better; score is in (0, 100]. Because no feasible packing can beat
the area/height lower bound, the score never exceeds 100. An infeasible or
unparsable solution scores exactly 0.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    W = int(next(it))
    N = int(next(it))
    rects = []
    for _ in range(N):
        w = int(next(it))
        h = int(next(it))
        rects.append((w, h))
    return W, N, rects


def read_solution(path, N):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) != 2 * N:
        return None
    coords = []
    for i in range(N):
        try:
            x = int(toks[2 * i])
            y = int(toks[2 * i + 1])
        except (ValueError, IndexError):
            return None
        coords.append((x, y))
    return coords


def overlaps(a, b):
    # a, b = (x, y, w, h); interiors overlap?
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    return True


def score(instance_file, solution_file):
    W, N, rects = read_instance(instance_file)
    coords = read_solution(solution_file, N)
    if coords is None:
        return 0.0
    placed = []
    total_area = 0
    max_h = 0
    H = 0
    for i, (w, h) in enumerate(rects):
        x, y = coords[i]
        if x < 0 or y < 0:
            return 0.0
        if x + w > W:
            return 0.0
        placed.append((x, y, w, h))
        total_area += w * h
        max_h = max(max_h, h)
        H = max(H, y + h)

    # Overlap check. Sort by x to prune; N is small (<=200) so O(N^2) is fine.
    n = len(placed)
    for i in range(n):
        ax, ay, aw, ah = placed[i]
        for j in range(i + 1, n):
            if overlaps(placed[i], placed[j]):
                return 0.0

    if N == 0 or H == 0:
        return 100.0
    LB = max(math.ceil(total_area / W), max_h)
    return 100.0 * LB / H


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(1)
    s = score(sys.argv[1], sys.argv[2])
    print(f"{s:.6f}")


if __name__ == "__main__":
    main()
