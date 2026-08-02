#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE photoresist-mask instance to stdout.

Format:
    N
    <N lines, each an N-character string over {0,1}: the TARGET pattern>

Deterministic: fully determined by testId (small -> large/adversarial ladder).
Cases 2,3,4,7,8,10 are TRAP cases (isolated sub-resolution features, tight-pitch
dense features, and mixed-scale combinations that a uniform bias cannot fix at once).
"""
import random
import sys


def mkgrid(n):
    return [[0] * n for _ in range(n)]


def rect(g, x0, y0, x1, y1):
    n = len(g)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            if 0 <= x < n and 0 <= y < n:
                g[x][y] = 1


def case1():
    n = 13
    t = mkgrid(n)
    rect(t, 4, 4, 8, 8)  # single large solid block: sanity case
    return n, t


def case2():
    n = 13
    t = mkgrid(n)
    rect(t, 5, 5, 7, 7)  # one isolated 3x3 square: sub-resolution shrink trap
    return n, t


def case3():
    n = 17
    t = mkgrid(n)
    for (x0, y0) in [(2, 2), (2, 12), (12, 2), (12, 12), (7, 7)]:
        rect(t, x0, y0, x0 + 2, y0 + 2)  # five scattered isolated 3x3 squares
    return n, t


def case4():
    n = 17
    t = mkgrid(n)
    for y in range(1, n, 2):
        rect(t, 2, y, n - 3, y)  # tight-pitch parallel lines: bridging trap
    return n, t


def case5():
    n = 17
    t = mkgrid(n)
    rect(t, 7, 3, 9, 13)
    rect(t, 3, 7, 13, 9)  # plus/cross: convex + concave corners
    return n, t


def case6():
    n = 17
    t = mkgrid(n)
    rect(t, 3, 3, 6, 13)
    rect(t, 3, 10, 13, 13)  # L-shape: corner rounding trap
    return n, t


def case7():
    n = 19
    t = mkgrid(n)
    for y in range(2, 10, 2):
        rect(t, 2, y, 9, y)  # dense comb teeth
    rect(t, 13, 13, 15, 15)  # + one isolated 3x3 square, different scale
    return n, t


def case8():
    n = 21
    t = mkgrid(n)
    for y in range(2, 11, 2):
        rect(t, 2, y, 10, y)  # dense comb teeth
    rect(t, 14, 3, 18, 7)  # a solid block
    for (x, y) in [(14, 14), (18, 18), (16, 16)]:
        rect(t, x, y, x + 2, y + 2)  # isolated 3x3 squares
    return n, t


def case9():
    n = 23
    rnd = random.Random(9)
    t = mkgrid(n)
    for _ in range(6):
        x0 = rnd.randint(2, n - 6)
        y0 = rnd.randint(2, n - 6)
        w = rnd.randint(1, 4)
        h = rnd.randint(1, 4)
        rect(t, x0, y0, min(x0 + w, n - 3), min(y0 + h, n - 3))
    return n, t


def case10():
    n = 25
    t = mkgrid(n)
    for y in range(2, 12, 2):
        rect(t, 2, y, 12, y)  # dense comb teeth
    rect(t, 16, 3, 21, 9)  # solid block
    rect(t, 16, 12, 18, 21)
    rect(t, 16, 18, 22, 21)  # L-shape
    for (x, y) in [(3, 15), (3, 19), (3, 22), (7, 15)]:
        rect(t, x, y, x + 2, y + 2)  # isolated 3x3 squares
    return n, t


CASES = [case1, case2, case3, case4, case5, case6, case7, case8, case9, case10]


def main():
    test_id = int(sys.argv[1])
    fn = CASES[(test_id - 1) % len(CASES)]
    n, t = fn()
    out = [str(n)]
    for row in t:
        out.append("".join(str(c) for c in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
