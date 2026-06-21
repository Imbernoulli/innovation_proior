The problem is to place $N$ identical points on the unit sphere $S^{n-1}$ so as to minimize the total pairwise repulsion $\sum_{x \neq y} f(|x-y|^2)$, where $f$ is a repulsive radial potential written as a function of squared distance — Coulomb $f(r) = r^{-1/2}$ (i.e. $1/|x-y|$) in $\mathbb{R}^3$, the harmonic law $r^{-(n/2-1)}$ in $\mathbb{R}^n$, any Riesz power $1/r^s$, a Gaussian $e^{-cr}$, and so on. The honest difficulty is not finding good configurations but certifying them. The obvious approach is direct minimization: drop $N$ points at random, compute the sphere-tangential gradient of the energy, slide everything downhill. That works as a heuristic — for $N=12$ on $S^2$ it lands on the icosahedron — but the moment one wants to *trust* it, it collapses. The energy landscape has a number of local minima growing roughly exponentially in $N$, so a downhill walk can never know it found the global minimum rather than one of a swarm of traps; worse, for $N=8$ and $N=20$ the true minimizer is not the Platonic solid one would guess (the cube and dodecahedron lose), so even the intuition that the symmetric configuration wins is unreliable. Steepest descent is a way to *guess*, never to *prove*, and it must be redone for every $N$ and every $f$. The earlier rigorous results — Yudin's spherical-harmonic bounds for the simplex and cross polytope, their extensions to the $E_8$ and Leech minimal vectors and the icosahedron — each treat essentially one harmonic potential and one configuration with an auxiliary polynomial chosen by hand, and the linear-programming code bounds of Delsarte–Goethals–Seidel and Kabatiansky–Levenshtein certify *cardinality* for a packing threshold, not the *energy* of a smooth potential. What is missing is a single uniform recipe that covers an arbitrary completely monotonic $f$ and an arbitrary special configuration at once, and explains why the same exceptional objects keep winning across many energies.

I propose a dual positive-definite certificate, a linear-programming energy bound, together with an explicit construction of the optimal certificate that proves *universal optimality*: that certain sharp configurations simultaneously minimize energy for every completely monotonic potential. The starting move is to stop searching configurations and instead bound them all at once. Since the energy depends only on inner products through $|x-y|^2 = 2 - 2\langle x,y\rangle$, pass to the variable $t = \langle x,y\rangle \in [-1,1]$ and set $a(t) = f(2-2t)$. Now choose a polynomial $h$ with two properties. The first is domination, $h(t) \leq a(t)$ for all $t \in [-1,1)$, which lets me replace the awkward potential by a polynomial and lose nothing as a lower bound. The second is that $h$ be positive-definite on the sphere: expanded as $h(t) = \sum_i \alpha_i\, C_i^{n/2-1}(t)$ in the Gegenbauer (ultraspherical) polynomials, all coefficients $\alpha_i \geq 0$. The $C_i^{n/2-1}$ are the reproducing kernels of the degree-$i$ spherical harmonics — under the rotation group $O(n)$, $L^2(S^{n-1})$ decomposes into harmonics $V_0, V_1, V_2, \ldots$, each $V_i$ has a kernel that depends only on $\langle x,y\rangle$ because $O(n)$ acts distance-transitively, and orthogonality forces that kernel to be $C_i^{\lambda}$ with $\lambda = n/2-1$, normalized by $C_0 = 1$, $C_1 = 2\lambda t$. Their decisive feature is that written as $\langle \mathrm{ev}_{i,x}, \mathrm{ev}_{i,y}\rangle$ the double sum factors into a norm squared,
$$\sum_{x,y} C_i(\langle x,y\rangle) = \Big|\sum_x \mathrm{ev}_{i,x}\Big|^2 \geq 0,$$
which cannot be negative; Schoenberg's theorem says these are *all* the continuous distance-only positive-definite kernels. The reason the Gegenbauer basis is the right one, and not the obvious monomial basis, is precisely that here every coefficient's contribution to the all-pairs sum has a fixed, predictable sign for every configuration. Adding back the diagonal terms $h(1)$ and using $C_0 = 1$ so the $i=0$ term is exactly $\alpha_0 N^2$, every $i>0$ term with $\alpha_i \geq 0$ only helps, so for *any* $N$ points,
$$\text{energy} \;\geq\; \sum_{x \neq y} h(\langle x,y\rangle) \;=\; \sum_{x,y} h(\langle x,y\rangle) - N h(1) \;\geq\; N^2 \alpha_0 - N h(1).$$
This is a lower bound depending only on $h$, not on the configuration. Maximizing $N^2 \alpha_0 - N h(1)$ — a linear functional of the coefficients — subject to the pointwise domination (a linear inequality at each $t$) and $\alpha_i \geq 0$ is an infinite-dimensional linear program; sampling the domination constraint on a fine grid of $t$ gives a finite, solvable LP. This is the generalization of the code LP bounds from a packing threshold to a smooth potential.

The numerical LP is close but not tight in general — for $N=20$ on $S^2$ with Coulomb a degree-6 $h$ gives roughly $301.2$ against a true minimum near $301.76$ — so the real question is for *which* configurations the gap closes exactly. Tracing the inequalities back to equalities, the bound is sharp for a configuration $S$ precisely when $h = a$ at every inner product occurring between distinct points of $S$, and $\sum_{x,y} C_i(\langle x,y\rangle) = 0$ for every $i>0$ with $\alpha_i > 0$. The second condition is exactly that the degree-$i$ harmonics sum to zero over $S$, which by Delsarte–Goethals–Seidel holds for $1 \leq i \leq M$ when $S$ is a spherical $M$-design. So I want $S$ to have few distinct inner products $t_1 < \cdots < t_m$ (few interpolation conditions) and high design strength. These pull against each other, and they balance at what I call a *sharp* configuration: $m$ distinct inner products and a spherical $(2m-1)$-design. That degree is within one of the absolute ceiling — if $S$ had $m$ distances and were a $(2m+1)$-design, the nonnegative, not-identically-zero polynomial $(1-\langle x,y\rangle)\prod_i(\langle x,y\rangle - t_i)^2$ of degree $2m+1$ would vanish at every point of $S$ yet integrate positively over the sphere, a contradiction. The regular simplices, cross polytopes, the icosahedron ($m=3$; a 5-design), the $E_8$ minimal vectors ($m=4$; a 7-design) and the Leech minimal vectors ($m=6$; an 11-design) are all sharp.

For a sharp $S$ and a completely monotonic $f$, the magic auxiliary function is the Hermite interpolant of $a$ to order two at the $m$ inner products $t_1,\ldots,t_m$ — matching $h(t_i)=a(t_i)$ and $h'(t_i)=a'(t_i)$ — a polynomial of degree $2m-1$, exactly what the design strength supports. Plain interpolation at the $m$ points would only give first-order contact, and $h$ would cross $a$ and violate domination; second-order contact makes $h$ touch $a$ tangentially with no sign change. That domination holds is not wishful: the Hermite remainder formula gives
$$a(t) - h(t) = \frac{a^{(2m)}(\xi)}{(2m)!}\prod_{i=1}^{m}(t - t_i)^2$$
for some $\xi$, where the product is a square and $a^{(2m)}(\xi) \geq 0$ because $f$ completely monotonic ($(-1)^k f^{(k)} \geq 0$) is equivalent, after the chain rule through $a(t)=f(2-2t)$, to $a$ being *absolutely monotonic* — all derivatives $a^{(k)} \geq 0$. So $h \leq a$ on $[-1,1)$, with equality at each $t_i$, for free. This is exactly why the strong hypothesis is needed: ordinary convexity gives only the simplex (one distance, a tangent line); all derivatives nonnegative are what make the high-order remainder argument run. And since the construction uses nothing about $f$ beyond absolute monotonicity of $a$, the very same configuration minimizes energy for Coulomb, every Riesz power, the Gaussian — all at once.

Tightness then needs only that $S$ is a $(2m-1)$-design, which equals $\deg(h)$, killing the $i>0$ terms — provided $h$ really is positive-definite, the subtle demand. Writing $F(t)=\prod_{i=1}^m (t-t_i)$ so that $h = H(a, F^2)$, I need $F^2$ to be *conductive*: that $H(a,\cdot)$ of every absolutely monotonic $a$ comes out positive-definite. The propagation rests on the Hermite product identity
$$H(a, g_1 g_2) = H(a, g_1) + g_1\, H\!\big(Q(a, g_1), g_2\big), \qquad Q(a,g) = \frac{a - H(a,g)}{g},$$
where $Q$ is the smooth quotient after interpolation. Three facts make the induction close. First, $Q(a,g)$ is again absolutely monotonic when the roots of $g$ lie in the interval (which they do, being inner products): $Q(a,g)(t)=a^{(\deg g)}(\xi)/(\deg g)! \geq 0$ for the value, and composing the interpolation identities, higher derivatives of $Q$ reduce to high derivatives of $a$, hence $\geq 0$. Second, products of positive-definite functions are positive-definite — the triple Gegenbauer integral $\int C_i C_j C_k$ is a sum of squared harmonic integrals, so positive-definite functions form a multiplicatively closed cone (the Schur product theorem). Third, the linear factors are conductive because interpolating $a$ against $t-r$ gives the constant $a(r)\geq 0$, and $F$ itself is strictly positive-definite: its low Gegenbauer coefficients are, by the design property with $\deg F + i \leq 2m-1$, finite sums over $S$ that for $y \in S$ telescope to the single positive term $F(1)C_i(1)$ since $F$ vanishes at the other inner products. Expressing $F = p_m + \alpha\,p_{m-1}$ in the $(1-t)\,d\mu$-orthogonal Jacobi family makes the partial products $\prod_{i=1}^j (t-t_i)$ positive-definite for $j<m$ as well; multiplying conductivity up the identity yields $F$ conductive, hence $F^2 = F\cdot F$ conductive, hence $h$ positive-definite. That completes the certificate: $h$ has degree $2m-1$, dominates $a$ with double contact at the $m$ inner products, and is positive-definite, so every $N$-point configuration has energy at least $N^2\alpha_0 - N h(1)$ and the sharp $S$ meets it with equality — $S$ is a global minimizer, and a *universally optimal* one, minimizing every completely monotonic energy simultaneously. On compact subintervals of $(0,4]$ it suffices to verify the nonnegative basis potentials $f(r)=(4-r)^k$ and pass to uniform limits, since inverse powers and Gaussians have positive expansions in those; and universal optimality forces optimal-code (maximal-minimal-angle) behavior too, because in the large-$s$ limit of $1/r^s$ the energy is dominated by the minimal distance.

One configuration resists the clean argument: the vertices of the 600-cell ($n=4$, $N=120$). They have eight distinct inner products — $-1$, $0$, $\pm 1/2$, $(\pm 1 \pm \sqrt5)/4$ — and form an 11-design, not the 15-design the eight-root second-order Hermite polynomial of degree 15 would need, so sharpness is not automatic and there is no conductivity proof. The fix uses the extra symmetry: although the degree-12 harmonics are an obstruction, the harmonics of degrees 13 through 19 sum to zero over the vertices, so I take a polynomial of degree at most 17 matching $a$ at all eight inner products, matching $a'$ at the seven inner products other than $-1$, and imposing $\alpha_{11}=\alpha_{12}=\alpha_{13}=0$. The $\alpha_{12}$ condition removes the degree the design and symmetry do not kill; $\alpha_{11}$ and $\alpha_{13}$ are part of the interpolation system that makes the later inequalities work; and the remaining domination and nonnegative-coefficient checks are done by exact arithmetic in $\mathbb{Q}(\sqrt5)$.

The code below is the numerical side — the finite LP probe corresponding to the dual bound. It imposes the domination $h(t)\leq a(t)$ only on a grid, so it is a way to search for auxiliary polynomials and compare against candidate energies; it is not, by itself, the continuous certificate, which requires the exact Hermite construction above.

```python
import numpy as np
from scipy.special import gegenbauer
from scipy.optimize import linprog

def lp_energy_lower_bound(n, N, f_of_squared_dist, degree=12, grid=600):
    """Grid LP value for sum_{x!=y} f(|x-y|^2) over N points on S^{n-1}.

    h(t) = sum_i alpha_i C_i^{n/2-1}(t), alpha_i >= 0, h(t) <= f(2-2t);
    if that domination is verified on the full interval, then
    energy >= N^2 alpha_0 - N h(1). This code enforces domination on a grid."""
    lam = n / 2.0 - 1.0
    C1 = np.array([gegenbauer(i, lam)(1.0) for i in range(degree + 1)])
    obj = np.array([N * N if i == 0 else 0.0 for i in range(degree + 1)]) - N * C1
    c = -obj
    ts = np.linspace(-1 + 1e-3, 1 - 1e-4, grid)
    A_ub = np.array([[gegenbauer(i, lam)(t) for i in range(degree + 1)] for t in ts])
    b_ub = np.array([f_of_squared_dist(2 - 2 * t) for t in ts])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (degree + 1),
                  method="highs")
    return -res.fun, res.x

def coulomb(R):  # f(R) = R^{-1/2}, R = squared distance -> 1/dist
    return R ** -0.5

def icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    raw = []
    for a in (-1, 1):
        for b in (-phi, phi):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    P = np.unique(np.array(raw, float), axis=0)
    return P / np.linalg.norm(P[0])

def energy(P, f):
    E = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                E += f(np.sum((P[i] - P[j]) ** 2))
    return E

if __name__ == "__main__":
    bound, alpha = lp_energy_lower_bound(3, 12, coulomb)
    P = icosahedron()
    print("LP grid value       :", bound)
    print("icosahedron energy  :", energy(P, coulomb))
    print("gap (energy-bound)  :", energy(P, coulomb) - bound)
```
