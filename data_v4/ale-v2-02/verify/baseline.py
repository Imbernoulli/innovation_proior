#!/usr/bin/env python3
"""Trivial baseline packer for ale-v2-02: next-fit shelf packing in input order.

Reads the instance on stdin, writes a feasible packing on stdout (same format
as the solver). Rectangles are placed left-to-right on a "shelf"; when the next
rectangle does not fit in the remaining width of the current shelf, a new shelf
is opened above (at the current shelf's max height). This is the standard naive
strip-packing baseline; the SA solver must beat its height (i.e. score higher).
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    W = int(next(it)); N = int(next(it))
    rects = [(int(next(it)), int(next(it))) for _ in range(N)]
    out = []
    shelf_x = 0          # current x cursor on the shelf
    shelf_y = 0          # bottom y of the current shelf
    shelf_h = 0          # height of the current shelf so far
    for (w, h) in rects:
        if shelf_x + w > W:
            # open a new shelf above
            shelf_y += shelf_h
            shelf_x = 0
            shelf_h = 0
        out.append(f"{shelf_x} {shelf_y}")
        shelf_x += w
        shelf_h = max(shelf_h, h)
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
