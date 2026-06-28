import sys
from fractions import Fraction

# Independent oracle for half-plane feasibility.
# Each half-plane: a*x + b*y <= c (integers).
# The intersection (clipped to a large box) is a convex region. If non-empty,
# it has at least one vertex = intersection of two boundary lines (where the two
# include the four box lines). So: enumerate all pairwise boundary-line
# intersections (exact rationals), and test each against every constraint.
# Output YES iff some candidate satisfies all constraints (with <= as closed).

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        m = int(next(it))
    except StopIteration:
        return
    planes = []
    for _ in range(m):
        a = int(next(it)); b = int(next(it)); c = int(next(it))
        planes.append((a, b, c))

    # Add the same bounding box the solution uses, so unbounded-but-feasible
    # regions also expose a vertex inside the box.
    B = 4000000000000
    planes_box = planes + [(1, 0, B), (-1, 0, B), (0, 1, B), (0, -1, B)]

    n = len(planes_box)

    def feasible_point(x, y):
        # x,y are Fractions; closed half-planes (<=)
        for (a, b, c) in planes_box:
            if a * x + b * y > c:
                return False
        return True

    # Enumerate all vertices: intersections of pairs of boundary lines.
    for i in range(n):
        ai, bi, ci = planes_box[i]
        for j in range(i + 1, n):
            aj, bj, cj = planes_box[j]
            D = ai * bj - aj * bi
            if D == 0:
                continue  # parallel, no unique vertex
            # Cramer
            x = Fraction(ci * bj - cj * bi, D)
            y = Fraction(ai * cj - aj * ci, D)
            if feasible_point(x, y):
                print("YES")
                return
    print("NO")

main()
