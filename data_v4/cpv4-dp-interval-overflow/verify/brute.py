import sys

# Independent brute force.
#
# The DP (sol.cpp) relies on the optimal-substructure recurrence over arcs.
# To verify it WITHOUT borrowing that assumption, we instead ENUMERATE every
# distinct triangulation of the convex polygon explicitly, compute the exact
# total cost of each (sum over its actual triangles of v[a]*v[b]*v[c]), and
# take the minimum over the full list. This is exponential, so it is only run
# on tiny n (the generator keeps n <= 7), where it is trivially fast and is, by
# construction, obviously correct: it literally tries all triangulations.

def all_triangulations(poly):
    """Yield every triangulation of the convex polygon given by the list of
    vertex indices `poly` (in boundary order). Each triangulation is yielded as
    a list of triangles, where each triangle is a 3-tuple of vertex indices.

    Method: fix the edge (poly[0], poly[-1]). In any triangulation that edge
    belongs to exactly one triangle, whose apex is some interior vertex poly[k]
    (0 < k < len-1). That apex splits the polygon into the left sub-polygon
    poly[0..k] and the right sub-polygon poly[k..end]; recurse on each.
    """
    m = len(poly)
    if m < 3:
        yield []
        return
    a, b = poly[0], poly[-1]
    for k in range(1, m - 1):
        c = poly[k]
        for left in all_triangulations(poly[0:k + 1]):
            for right in all_triangulations(poly[k:m]):
                yield left + right + [(a, c, b)]


def solve(n, v):
    if n < 3:
        return 0
    best = None
    poly = list(range(n))
    for tri in all_triangulations(poly):
        total = 0
        for (a, b, c) in tri:
            total += v[a] * v[b] * v[c]
        if best is None or total < best:
            best = total
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    v = [int(data[idx + t]) for t in range(n)]
    print(solve(n, v))


if __name__ == "__main__":
    main()
