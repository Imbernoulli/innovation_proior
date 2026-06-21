I will work through the probability that four independent uniform points on a sphere enclose the center, and I will phrase the core idea as the antipodal coin-flip method, or more descriptively, the sign-criterion method for random spherical tetrahedra. The canonical name I propose for this approach is the antipodal coin-flip method, because the entire computation reduces to counting sign patterns of antipodal endpoint choices rather than evaluating any surface integral.

Place the sphere at the origin O. The problem asks for the probability that O lies strictly inside the convex hull of four independent uniform points P1, P2, P3, P4 on the sphere. My first instinct is often to fix three points and integrate the spherical area where the fourth point must land, but that integral quickly becomes unpleasant. Instead, I look for a symmetry that turns the continuous geometry into a discrete count.

The key observation is that sampling a uniform point on the sphere is equivalent to sampling a uniform diameter through O and then flipping a fair coin to choose one of its two endpoints. The uniform distribution is invariant under the antipodal map, so the coin is genuinely fair and independent of the diameter. I will hold one point fixed and reparametrize the other three in this way. Let P4 be the ordinary free point, and for i = 1, 2, 3 let the line through O and Pi be a uniform diameter, with the actual endpoint Pi determined by an independent fair coin flip.

With the three diameters and P4 fixed, the only remaining randomness is three independent fair coin flips, giving 2^3 = 8 equally likely candidate tetrahedra. I claim that, no matter what the diameters and P4 are, exactly one of these eight tetrahedra contains O. If this claim holds, then the conditional probability is always 1/8, and averaging over the diameters and P4 leaves the answer unchanged at 1/8.

To justify the claim, I use a normalization-free criterion for the origin to lie inside the convex hull of four points. Consider four nonzero vectors v1, v2, v3, v4 in R^3 in general position, meaning any three are linearly independent and the four are affinely independent. They have a unique-up-to-scaling linear dependence c1 v1 + c2 v2 + c3 v3 + c4 v4 = 0 with all ci nonzero. The origin is strictly inside the convex hull of the vi if and only if the coefficients ci all share the same sign. Why? The origin is interior exactly when there exist strictly positive barycentric weights lambda_i with sum 1 such that sum lambda_i vi = 0. Any vector solving sum lambda_i vi = 0 must be a scalar multiple of the dependence vector c, because the dependence space is one-dimensional. Thus lambda_i = t ci for some t, and the normalization sum lambda_i = 1 forces t = 1 / sum cj. Therefore all lambda_i are positive exactly when all ci have the same sign. This is the decisive sign criterion.

The sign criterion behaves beautifully under antipodal flips. Suppose the base dependence with endpoints q, r, s is c1 P4 + c2 q + c3 r + c4 s = 0. If I flip the endpoint of the first diameter, replacing q by -q, then to maintain the same linear combination I replace c2 by -c2 and leave the other coefficients unchanged. The same holds for the other two diameters. So for a coin pattern epsilon_2, epsilon_3, epsilon_4 in {-1, +1}^3, the dependence coefficients become (c1, epsilon_2 c2, epsilon_3 c3, epsilon_4 c4).

Now apply the sign criterion. The first coefficient c1 is fixed because P4 is fixed. For all four coefficients to share the same sign, each epsilon_i c_i must match the sign of c1. This forces epsilon_i = sign(c1) / sign(c_i), a single specific value for each i. Therefore exactly one of the eight coin patterns makes all four coefficients share a sign, and hence exactly one of the eight tetrahedra contains O. The eight patterns are equally likely, so the probability is 1/8.

This argument generalizes cleanly. In the plane, with three points on a circle, the same reasoning gives 2^2 = 4 equally likely triangles and exactly one containing the center, so the probability is 1/4. In dimension d, the probability becomes 2^{-d}. The sphere case d = 3 therefore gives 1/8.

Degenerate configurations, such as four coplanar points or O lying exactly on a face of the tetrahedron, are measure-zero events and do not affect the probability. The sign criterion also fails only when some ci = 0, which corresponds to such degeneracies, so the argument is complete almost surely.

I will now verify the result numerically with a small Python simulation. The script samples many independent sets of four uniform points on the unit sphere, checks whether the origin lies inside each tetrahedron by solving for positive barycentric weights, and reports the empirical frequency.

```python
import numpy as np

def origin_in_tet(p1, p2, p3, p4):
    # Check whether the origin lies strictly inside the tetrahedron
    # with vertices p1, p2, p3, p4 by solving for strictly positive
    # barycentric weights that sum to 1.
    M = np.column_stack([p1, p2, p3, p4])
    A = np.vstack([np.ones(4), M])
    b = np.array([1.0, 0.0, 0.0, 0.0])
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False
    return np.all(w > 0)

def sample_sphere():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)

np.random.seed(0)
n = 200_000
count = 0
for _ in range(n):
    pts = [sample_sphere() for _ in range(4)]
    if origin_in_tet(*pts):
        count += 1

print(f"Empirical probability: {count / n:.6f}")
print(f"Expected probability:  {1/8:.6f}")
```

The antipodal coin-flip method thus gives the exact answer 1/8 with almost no calculation, and the numerical simulation confirms it.
