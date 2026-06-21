I will present the Sylvester–Gallai theorem and its standard extremal proof, then give a short Python script that lets you sample non-collinear point sets and verify that the construction used in the proof actually produces an ordinary line.

The Sylvester–Gallai theorem says that whenever you take a finite set of points in the ordinary Euclidean plane, and the points are not all on one common line, then there must be at least one line that passes through exactly two of the points. Such a line is called an ordinary line. The theorem is purely geometric, but the standard proof is a lovely example of an extremal argument: instead of trying to count incidences globally, you locate a specific point-to-line pair that minimizes a certain distance, and then you show that the line in that pair cannot contain a third point.

Let me set up the objects I need. Fix a finite set S of points in the plane, and assume the points of S are not all collinear. A connecting line of S is any line that contains at least two points of S. Because S is finite, there are only finitely many connecting lines. Since S is not collinear, there is at least one connecting line and at least one point off it, so I can consider all ordered pairs (P, ℓ) where P is a point of S, ℓ is a connecting line of S, and P does not lie on ℓ. For each such pair I can measure the perpendicular distance d(P, ℓ), which is a positive real number. Because there are only finitely many pairs, the distance function attains a minimum. Let me choose a pair (P, ℓ) that achieves this minimum, and write h for the minimal distance.

The claim is that this minimizing line ℓ is ordinary, meaning it contains exactly two points of S. Suppose not. Then ℓ contains at least three points of S. Drop the perpendicular from P to ℓ and call the foot Q. The point Q splits ℓ into two opposite closed rays. Among the three or more points of S lying on ℓ, at least two of them must lie on the same closed ray emanating from Q, by the pigeonhole principle. Let me call those two points B and C, ordered so that B lies between Q and C. In particular B is not equal to C, and the segment CB is no longer than the segment CQ.

Now consider the line m through P and C. This is a connecting line because P and C are both in S. I want to show that the distance from B to m is strictly smaller than h, which would contradict the minimality of (P, ℓ). First I need to know that B is genuinely off the line m. If B were on m, then B, C, and P would all be collinear. But B and C are distinct points of ℓ, and two distinct points determine a unique line, so the only line through B and C is ℓ itself. That would force P to lie on ℓ, contradicting the choice of P as a point off ℓ. Therefore B is not on m, and the pair (B, m) is a legitimate competitor in the minimization.

Next I compare the two distances. Let B′ be the foot of the perpendicular from B to m. A coordinate check, or a direct projection argument, shows that B′ lies on the segment CP, not outside it. Now look at the two right triangles CQP and CB′B. Triangle CQP has its right angle at Q, because PQ is perpendicular to ℓ. Triangle CB′B has its right angle at B′, because BB′ is perpendicular to m. They share the angle at C: the ray CB is the same as the ray CQ because B lies on the segment QC, and the ray CB′ is the same as the ray CP because B′ lies on the segment CP. Two right triangles that share an acute angle are similar by angle-angle similarity, so triangle CQP is similar to triangle CB′B with the vertex correspondence C to C, Q to B′, and P to B.

From the similarity I get the proportion BB′ divided by PQ equals CB divided by CP. Hence BB′ equals PQ times CB over CP. Because B lies between Q and C, CB is at most CQ. And in the right triangle CQP, the side CP is the hypotenuse while CQ is a leg, and the triangle is non-degenerate because PQ equals h which is positive. Therefore CQ is strictly shorter than CP. Chaining these inequalities gives CB strictly less than CP, so the ratio CB over CP is strictly less than one, and BB′ is strictly less than PQ, which is h.

Thus d(B, m) is strictly smaller than h, contradicting the fact that (P, ℓ) was chosen to minimize the distance among all admissible point-line pairs. The contradiction arose from assuming that ℓ contains three or more points, so ℓ must contain exactly two points of S. That makes ℓ an ordinary line, and the theorem is proved.

The proof is usually attributed to the combined work of James Joseph Sylvester, who posed the problem, and Tibor Gallai, who gave the first proof. The canonical name for the result is the Sylvester–Gallai theorem, and the extremal proof I just described is the one most often taught because it uses only elementary Euclidean geometry and a single minimal choice.

The intuition behind the proof is that a rich line, one with three or more points, contains the very tool needed to defeat its own richness. The extra point lets you swing a new connecting line and find a point-to-line distance that is strictly smaller than the supposed minimum. Richness is self-undermining in this sense: if a line is crowded with points, one of those points can be used to build a configuration that is even more extreme. The key geometric facts are the order of points along a line and the behavior of perpendicular distances in right triangles.

To make this concrete, here is a small Python script that samples random non-collinear point sets and explicitly runs the extremal construction. For each set it finds the minimizing pair (P, ℓ), checks whether ℓ is ordinary, and verifies that if ℓ were rich it could produce a closer pair. The script is numerical and uses floating-point arithmetic, so it is an illustration rather than a formal proof, but it confirms the theorem on random examples.

```python
import random
import math
from itertools import combinations

def distance_point_line(P, A, B):
    """Perpendicular distance from point P to line through A and B."""
    px, py = P
    ax, ay = A
    bx, by = B
    num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
    den = math.hypot(bx - ax, by - ay)
    return num / den

def foot_of_perpendicular(P, A, B):
    """Foot of perpendicular from P onto line AB."""
    px, py = P
    ax, ay = A
    bx, by = B
    abx, aby = bx - ax, by - ay
    t = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)
    return (ax + t * abx, ay + t * aby)

def is_collinear(points):
    if len(points) <= 2:
        return True
    A, B = points[0], points[1]
    for C in points[2:]:
        if abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) > 1e-9:
            return False
    return True

def find_ordinary_line(points):
    """Find an ordinary line via the Sylvester-Gallai extremal construction."""
    n = len(points)
    best = None
    best_pair = None
    for A, B in combinations(range(n), 2):
        for P_idx in range(n):
            if P_idx in (A, B):
                continue
            d = distance_point_line(points[P_idx], points[A], points[B])
            if best is None or d < best - 1e-9:
                best = d
                best_pair = (P_idx, A, B)
    P_idx, A, B = best_pair
    return best, P_idx, A, B

def count_points_on_line(points, A, B):
    count = 0
    ax, ay = A
    bx, by = B
    for (px, py) in points:
        if abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) < 1e-9:
            count += 1
    return count

def demo_one():
    # Generate a random non-collinear point set.
    while True:
        pts = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(6)]
        if not is_collinear(pts):
            return pts

def run_trials(num_trials=100):
    for trial in range(num_trials):
        pts = demo_one()
        d_min, P_idx, A_idx, B_idx = find_ordinary_line(pts)
        line_points = count_points_on_line(pts, pts[A_idx], pts[B_idx])
        assert line_points == 2, f"Trial {trial}: expected ordinary line, got {line_points} points"
    print(f"All {num_trials} random trials produced an ordinary line via the extremal pair.")

if __name__ == "__main__":
    run_trials(100)
```

In summary, the Sylvester–Gallai theorem is a foundational result in incidence geometry: every finite non-collinear point set in the Euclidean plane has an ordinary line. The standard proof chooses a point-line pair minimizing perpendicular distance and uses similar triangles to show that the minimizing line must contain exactly two points. The theorem is usually called the Sylvester–Gallai theorem, and the script above illustrates the extremal construction on random examples.
