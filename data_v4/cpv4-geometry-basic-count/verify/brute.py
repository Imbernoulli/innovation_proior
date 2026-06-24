import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    raw = []
    for _ in range(n):
        x = int(data[idx]); idx += 1
        y = int(data[idx]); idx += 1
        raw.append((x, y))

    # Dedup identical points: coincident points are one location.
    pts = sorted(set(raw))
    m = len(pts)

    # Brute force: enumerate every unordered triple of DISTINCT points, test
    # whether it forms a right triangle with the right angle at SOME vertex and
    # both legs parallel to the axes. Count each qualifying triple once.
    def axis_right_triangle(a, b, c):
        # a,b,c distinct points. Try each as the right-angle apex.
        for apex, p, q in ((a, b, c), (b, a, c), (c, a, b)):
            # legs apex->p and apex->q must be axis-parallel and perpendicular:
            # one shares x with apex (vertical leg), the other shares y (horizontal leg).
            ax, ay = apex
            # p vertical, q horizontal
            if p[0] == ax and p[1] != ay and q[1] == ay and q[0] != ax:
                return True
            # p horizontal, q vertical
            if p[1] == ay and p[0] != ax and q[0] == ax and q[1] != ay:
                return True
        return False

    cnt = 0
    for a, b, c in combinations(pts, 3):
        if axis_right_triangle(a, b, c):
            cnt += 1

    print(cnt)

if __name__ == "__main__":
    main()
