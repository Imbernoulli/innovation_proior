I would call this the windmill construction via oriented side-count conservation. The problem gives a finite set S of at least two points in the plane with no three collinear, and asks us to start a windmill so that every point of S becomes a pivot infinitely many times. A windmill means a line through one point of S that rotates clockwise until it hits another point of S, which then becomes the new pivot, after which rotation continues about that new point, and so on forever. My strategy is to endow the line with an arrow and track the number of points of S strictly on the left side of that oriented line. That count turns out to be an invariant of the entire process, and if I initialize it at the most balanced value that parity permits, then the line is forced to sweep through every point of S again and again.

Let me explain why the left count stays constant. When the line is pivoting about a fixed point P and has not yet reached any other point of S, no point of S crosses the line, because the first moment another point touches the line is exactly when the pivot changes. So the left count is unchanged during a leg. The only interesting event is a pivot swap, when the rotating line about P meets Q and Q becomes the new pivot. I place Q at the origin and P at (a,0) with a positive, and I look at the instant when the line is horizontal. Just before contact, when the pivot is still P, the line is slightly counterclockwise from horizontal, and a short signed-area calculation shows that Q is on the left. Just after contact, when the pivot is Q, the line has moved slightly clockwise, and the same calculation shows that P is on the left. If the arrow points the other way, both signed areas flip sign, so the statement becomes that Q is on the right before and P is on the right after. In either case, the side that loses the new pivot Q gains the old pivot P, so the total number of points strictly on the left is unchanged. Therefore, once I choose a starting orientation, the left count L is preserved for all time away from the swap instants.

The next step is to find a starting value of L that I can maintain and that will force the line to visit every point. Through any point T in S I can find an oriented line with a balanced split, and I will set L equal to that balanced split. If |S| is odd, say |S| = 2n+1, then through every T there is a line with n points on each side. To see this, start with any oriented line through T avoiding other points of S and rotate it 180 degrees. The left count begins at some n+r and ends at n-r, because reversing the arrow swaps left and right. Between the finitely many moments when the line also passes through another point of S, the count changes by plus or minus one, since no three points are collinear. Thus some open interval of directions has left count exactly n, and I can choose a direction inside that interval that also avoids every direction parallel to a line through two points of S. Then the line through T contains only T and has n points on each side. If |S| is even, say |S| = 2n, the points other than T number 2n-1, so an even split is impossible, but the same rotation argument gives an oriented line through T with n-1 points on the left and n points on the right. Therefore, no matter the parity, every point T has an associated balanced oriented line l_T through T.

Why does a balanced line force a visit? For a fixed oriented direction and a fixed left count k, there is at most one parallel position of an oriented line that contains at least one point of S and has k points on the left. This is because if I parameterize the parallel lines by u dot x = t, where u is the left normal to the direction, then the left count is the number of points of S with u dot X strictly greater than t. As t increases, that count strictly decreases whenever t passes the value u dot X for some X in S. So two different parallel positions containing points of S cannot have the same left count. Now the windmill always contains a pivot on its line and always has left count k. When its oriented direction matches that of l_T, the only line in that direction with left count k through a point of S is l_T itself. I also chose l_T to avoid directions parallel to any two-point line, so this matching moment is not a swap instant. Hence at that moment the windmill line equals l_T and therefore passes through T. Because the windmill line always contains exactly its current pivot away from swaps, the current pivot must be T.

It remains to argue that every direction occurs infinitely often. After the first hit, each leg of the windmill starts on a line through two points of S and ends on another such line. The clockwise angles between successive pivot lines belong to a finite set, so they have a positive minimum. Since the process has infinitely many legs, the accumulated clockwise rotation is unbounded. The line rotates continuously during each leg, so every completed half-turn of accumulated motion covers every undirected direction, and every completed full turn covers every oriented direction. In the odd case the balanced split n/n is symmetric under reversing the arrow, so it suffices to become parallel to l_T, which happens in every half-turn. In the even case the split is asymmetric, so the windmill must attain the exact oriented direction of l_T, which happens in every full turn. Either way, every point of S is forced to be a pivot infinitely many times.

Here is a small Python verification that checks the two structural facts the proof relies on: a balanced oriented line exists through every point, and the side-count is preserved across every pivot swap. It uses the signed-area test to decide left membership, and it perturbs the contact direction infinitesimally to separate the moments just before and just after a swap.

```python
import random
import math

def det(u, v):
    return u[0]*v[1] - u[1]*v[0]

def left_count(points, pivot, direction):
    """Number of points of S strictly to the left of the oriented line."""
    return sum(1 for p in points
               if p != pivot
               and det(direction, (p[0]-pivot[0], p[1]-pivot[1])) > 0)

def rotate(v, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (c*v[0] - s*v[1], s*v[0] + c*v[1])

def swap_invariant(points, P, Q):
    """Check that left count is preserved when the pivot jumps from P to Q.
    The contact direction is the unit vector from Q toward P."""
    v = (P[0]-Q[0], P[1]-Q[1])
    norm = math.hypot(v[0], v[1])
    direction = (v[0]/norm, v[1]/norm)
    d_before = rotate(direction, 1e-8)   # pivot P, slightly before contact
    d_after = rotate(direction, -1e-8)   # pivot Q, slightly after contact
    return left_count(points, P, d_before) == left_count(points, Q, d_after)

def find_balanced_direction(points, T, target):
    """Find an oriented direction through T whose left count equals target."""
    for k in range(10000):
        theta = k * 2 * math.pi / 10000
        d = (math.cos(theta), math.sin(theta))
        if left_count(points, T, d) == target:
            return d
    return None

def check_parity(n, seed_value):
    random.seed(seed_value)
    points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(n)]
    points = sorted(set(points))
    target = (n - 1) // 2 if n % 2 == 1 else (n - 2) // 2
    for T in points:
        assert find_balanced_direction(points, T, target) is not None
    for i in range(n):
        for j in range(i + 1, n):
            assert swap_invariant(points, points[i], points[j])
    return target

if __name__ == "__main__":
    for n in [6, 7, 8, 9]:
        target = check_parity(n, seed_value=n)
        print(f"|S|={n}: target left count {target} works for every point and every swap.")
```

The script first defines the signed-area orientation test that tells whether a point lies to the left of an oriented line. For each tested cardinality it generates a random non-collinear set, computes the parity-appropriate balanced count, and searches for an oriented direction through each point that realizes that count. It then checks the swap invariant for every unordered pair: with the contact direction chosen as the unit vector from the new pivot toward the old pivot, a tiny counterclockwise perturbation before the swap and a tiny clockwise perturbation after the swap give the same left count. Passing these checks on both even and odd sample sizes confirms that the side-count engine in the proof is numerically sound.
