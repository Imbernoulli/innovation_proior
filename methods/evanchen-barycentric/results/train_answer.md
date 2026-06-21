I am looking at an olympiad-style triangle geometry problem: in an acute scalene triangle ABC, with M, N, P the midpoints of the sides, the perpendicular bisectors of AB and AC meet the median ray AM at D and E, and the lines BD and CE meet at F inside the triangle. The goal is to prove that A, N, F, P are concyclic. My first instinct is to try synthetic angle chasing, but that quickly runs into trouble. F is built from the intersection of two cevians that themselves come from perpendicular bisectors crossing a median, so the angles at F depend on a web of asymmetric relations between the two sides AB and AC. Because the triangle is scalene, there is no symmetry to collapse the two halves of the figure, and a clean synthetic chain is hard to find. Coordinate computation seems more reliable, but the usual frames all have a weakness here. Cartesian coordinates make the circumcircle and perpendicular bisectors messy and unstructured. Complex numbers are elegant for circles but painful for lines, and this problem is line-heavy. Vector methods lack a usable equation of a line. Each of these systems also imposes an external axis on the triangle, so the three vertices sit on unequal footing.

The right move is to make the triangle itself the coordinate frame. The method is called barycentric coordinates, also known as areal coordinates. Instead of placing the triangle in an external xy-plane, I write every point P as a weighted average of the three vertices: P = xA + yB + zC with x + y + z = 1. The weights are unique, and they have a concrete geometric meaning: x is the ratio of the signed area [PBC] to [ABC], and similarly for y and z. In this system the vertices become the standard basis, A = (1, 0, 0), B = (0, 1, 0), C = (0, 0, 1). The sides of the triangle are the coordinate planes: BC is x = 0, CA is y = 0, and AB is z = 0. A midpoint like P of AB is simply (1/2, 1/2, 0), and any point dividing a side in a ratio can be read off directly from its coordinates. This makes ratios, collinearity, and line intersections naturally linear.

To handle perpendicularity and circles, I need a metric inside the barycentric frame. I move the vector origin to the circumcenter O, which is allowed because the affine constraint x + y + z = 1 makes the coordinates translation-invariant. With O as origin, |A| = |B| = |C| = R, and the dot products become A·B = R² - c²/2 and its cyclic versions, where a = BC, b = CA, c = AB. For a displacement vector between two normalized points, the coordinate triple sums to zero. When I expand the dot product of two such displacements, the R² terms factor as R² times the product of the two coordinate sums, which is zero, so the radius disappears entirely. What remains is the barycentric perpendicularity criterion, often called EFFT: for displacements (x₁, y₁, z₁) and (x₂, y₂, z₂), perpendicularity means a²(y₁z₂ + y₂z₁) + b²(z₁x₂ + z₂x₁) + c²(x₁y₂ + x₂y₁) = 0. The squared length of a displacement (x, y, z) is -a²yz - b²zx - c²xy. The circumcircle, the circle through A, B, C, has the beautifully simple homogeneous equation a²yz + b²zx + c²xy = 0, and a general circle adds a linear correction.

Now I can solve the problem mechanically. Let A = (1, 0, 0), B = (0, 1, 0), C = (0, 0, 1). Then M = (0, 1/2, 1/2), N = (1/2, 0, 1/2), P = (1/2, 1/2, 0). The median ray AM consists of points with y = z, so I write D = (1 - 2t, t, t). Since D lies on the perpendicular bisector of AB, the segment DP must be perpendicular to AB. Applying EFFT to DP = (2t - 1/2, 1/2 - t, -t) and AB = (-1, 1, 0) yields t = c² / (3c² + b² - a²). Similarly E = (1 - 2k, k, k) with k = b² / (3b² + c² - a²). The lines BD and CE intersect at F. Because BD passes through B, the ratio z/x is constant along it; because CE passes through C, y/x is constant. Using the values of j and k and writing S = b² + c² - a², one finds F = (p, q, r) with q/p = b²/S and r/p = c²/S. Finally, a homothety centered at A with ratio 2 sends N to C and P to B, so it sends the circle through A, N, P to the circumcircle. Thus A, N, F, P are concyclic if and only if F' = 2F - A lies on the circumcircle a²yz + b²zx + c²xy = 0. Substituting the coordinates of F' gives 4a²b²c² - 2a²b²c² - 2a²b²c² = 0, an identity, so the concyclicity holds for every valid triangle.

```python
import sympy as sp

# Barycentric coordinates for a triangle with side lengths a, b, c.
# Vertices: A=(1,0,0), B=(0,1,0), C=(0,0,1).

# Symbolic side lengths (squared sides to keep everything polynomial)
a2, b2, c2 = sp.symbols('a2 b2 c2', positive=True, real=True)
S = b2 + c2 - a2  # shorthand from the proof

# Vertices
A = sp.Matrix([1, 0, 0])
B = sp.Matrix([0, 1, 0])
C = sp.Matrix([0, 0, 1])

# Midpoints
M = (B + C) / 2
N = (A + C) / 2
P = (A + B) / 2

# EFFT perpendicularity test for displacements u, v.
def efft(u, v, a2, b2, c2):
    return a2 * (u[1] * v[2] + u[2] * v[1]) + \
           b2 * (u[2] * v[0] + u[0] * v[2]) + \
           c2 * (u[0] * v[1] + u[1] * v[0])

# Find D on AM: D = (1 - 2t, t, t), with DP perpendicular to AB.
t = sp.symbols('t', real=True)
D = sp.Matrix([1 - 2*t, t, t])
eq_D = efft(P - D, B - A, a2, b2, c2)
t_val = sp.solve(eq_D, t)[0]
D = sp.simplify(D.subs(t, t_val))

# Find E on AM: E = (1 - 2k, k, k), with EN perpendicular to AC.
k = sp.symbols('k', real=True)
E = sp.Matrix([1 - 2*k, k, k])
eq_E = efft(N - E, C - A, a2, b2, c2)
k_val = sp.solve(eq_E, k)[0]
E = sp.simplify(E.subs(k, k_val))

# Intersect BD and CE. Since BD passes through B, z/x is constant.
# Since CE passes through C, y/x is constant.
p = sp.symbols('p', real=True)
ratio_r = D[2] / D[0]   # z/x along BD
ratio_q = E[1] / E[0]   # y/x along CE
q_expr = sp.simplify(ratio_q * p)
r_expr = sp.simplify(ratio_r * p)
# normalization p + q + r = 1
p_val = sp.solve(p + q_expr + r_expr - 1, p)[0]
F = sp.Matrix([
    sp.simplify(p_val),
    sp.simplify(q_expr.subs(p, p_val)),
    sp.simplify(r_expr.subs(p, p_val))
])

# Homothety centered at A with ratio 2: F' = 2F - A.
Fp = 2 * F - A

# Circumcircle equation: a^2*y*z + b^2*z*x + c^2*x*y = 0
circ = a2 * Fp[1] * Fp[2] + b2 * Fp[2] * Fp[0] + c2 * Fp[0] * Fp[1]
print("Circumcircle value at F':", sp.simplify(circ))
# Expected output: 0
```
