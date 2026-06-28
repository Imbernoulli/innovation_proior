#!/usr/bin/env python3
import math
import random
import subprocess
import sys


SOL = "/tmp/fcs_ge_04_sol"


def twice_area(poly):
    s = 0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s)


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, p):
    return (
        cross(a, b, p) == 0
        and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def segments_intersect(a, b, c, d):
    def sign(v):
        return (v > 0) - (v < 0)

    c1 = sign(cross(a, b, c))
    c2 = sign(cross(a, b, d))
    c3 = sign(cross(c, d, a))
    c4 = sign(cross(c, d, b))
    if c1 == 0 and on_segment(a, b, c):
        return True
    if c2 == 0 and on_segment(a, b, d):
        return True
    if c3 == 0 and on_segment(c, d, a):
        return True
    if c4 == 0 and on_segment(c, d, b):
        return True
    return c1 * c2 < 0 and c3 * c4 < 0


def is_simple(poly):
    if len(poly) < 3 or len(set(poly)) != len(poly) or twice_area(poly) == 0:
        return False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if a == b:
            return False
        for j in range(i + 1, n):
            if i == j:
                continue
            if (i + 1) % n == j or i == (j + 1) % n:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                return False
    return True


def strict_inside(poly, p):
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if on_segment(a, b, p):
            return False

    wn = 0
    x, y = p
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if a[1] <= y:
            if b[1] > y and cross(a, b, (x, y)) > 0:
                wn += 1
        else:
            if b[1] <= y and cross(a, b, (x, y)) < 0:
                wn -= 1
    return wn != 0


def brute_count(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    total = 0
    for x in range(min(xs), max(xs) + 1):
        for y in range(min(ys), max(ys) + 1):
            if strict_inside(poly, (x, y)):
                total += 1
    return total


def pick_count(poly):
    b = 0
    s = 0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
        b += math.gcd(abs(x2 - x1), abs(y2 - y1))
    return (abs(s) - b) // 2 + 1


def run_sol(poly):
    data = str(len(poly)) + "\n" + "\n".join(f"{x} {y}" for x, y in poly) + "\n"
    got = subprocess.check_output([SOL], input=data.encode()).decode().strip()
    return int(got)


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def random_radial_polygon(rng, lo=-7, hi=7):
    for _ in range(1000):
        m = rng.randint(3, 11)
        pts = set()
        while len(pts) < m:
            pts.add((rng.randint(lo, hi), rng.randint(lo, hi)))
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        poly = sorted(pts, key=lambda p: (math.atan2(p[1] - cy, p[0] - cx), p[0], p[1]))
        if is_simple(poly):
            return poly
    raise RuntimeError("failed to generate radial polygon")


def random_convex_polygon(rng):
    for _ in range(1000):
        pts = [(rng.randint(-8, 8), rng.randint(-8, 8)) for _ in range(rng.randint(5, 25))]
        hull = convex_hull(pts)
        if len(hull) >= 3 and is_simple(hull):
            return hull
    raise RuntimeError("failed to generate convex polygon")


def with_collinear_insertions(poly):
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        out.append(a)
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        g = math.gcd(abs(dx), abs(dy))
        if g > 1:
            out.append((a[0] + dx // g, a[1] + dy // g))
    return out if is_simple(out) else poly


def adversarial_cases():
    return [
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (2, 0), (0, 2)],
        [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)],
        [(0, 0), (10, 0), (10, 1), (0, 1)],
        [(0, 0), (6, 0), (3, 1)],
        [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)],
        [(-3, -2), (5, -2), (5, 3), (-3, 3)],
        [(0, 0), (3, 0), (6, 0), (6, 3), (3, 6), (0, 3)],
        [(0, 0), (8, 0), (8, 8), (6, 8), (6, 2), (2, 2), (2, 8), (0, 8)],
        [(0, 0), (5, -1), (7, 2), (4, 5), (1, 4), (-2, 2)],
        [(0, 0), (2, 1), (4, 0), (5, 3), (3, 5), (0, 4), (-1, 2)],
        [(10**9 - 3, 10**9 - 5), (10**9, 10**9 - 5), (10**9, 10**9), (10**9 - 3, 10**9)],
        [(-10**9, -10**9), (10**9, -10**9), (10**9, 10**9), (-10**9, 10**9)],
        [(-10**9, 0), (0, -10**9), (10**9, 0), (0, 10**9)],
    ]


def check(poly, expected, label):
    got = run_sol(poly)
    if got != expected:
        print(f"Mismatch on {label}", file=sys.stderr)
        print(f"poly={poly}", file=sys.stderr)
        print(f"expected={expected} got={got}", file=sys.stderr)
        sys.exit(1)


def main():
    rng = random.Random(20260628)
    brute_cases = []
    for poly in adversarial_cases()[:11]:
        brute_cases.append(poly)

    while len(brute_cases) < 360:
        if rng.random() < 0.45:
            poly = random_convex_polygon(rng)
            if rng.random() < 0.4:
                poly = with_collinear_insertions(poly)
        else:
            poly = random_radial_polygon(rng)
        xs = [x for x, _ in poly]
        ys = [y for _, y in poly]
        if is_simple(poly) and (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) <= 400:
            brute_cases.append(poly)

    for idx, poly in enumerate(brute_cases):
        expected = brute_count(poly)
        check(poly, expected, f"brute-{idx}")

    for idx, poly in enumerate(adversarial_cases()[11:]):
        check(poly, pick_count(poly), f"large-pick-{idx}")

    print(f"PASS {len(brute_cases)} brute-force cases + {len(adversarial_cases()) - 11} large exact cases")


if __name__ == "__main__":
    main()
