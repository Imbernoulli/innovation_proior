The kissing number $\tau_n$ asks how many non-overlapping unit spheres can simultaneously touch a central unit sphere in $\mathbb{R}^n$. Two outer spheres that both touch the center have their centers at distance exactly $2$ from the central center and at distance $\geq 2$ from each other, so projecting the contact directions onto the unit sphere reduces the whole question to a statement about angles: place unit vectors $x_1,\dots,x_N$ on $S^{n-1}$ so that any two are at least $60$ degrees apart, i.e. $\langle x_i, x_j\rangle \leq \tfrac12$ for all $i\neq j$, and maximize $N$. Thus $\tau_n = A(n,\tfrac12)$, the maximum size of a spherical code with angle parameter $s=\tfrac12$. The problem splits into two halves of completely different character. A lower bound $\tau_n \geq N$ is constructive — exhibit $N$ vectors obeying the constraint and you are done — and the minimal vectors of the great lattices already supply these: the icosahedron's $12$ vertices give $\tau_3 \geq 12$, the $240$ shortest vectors of $E_8$ give $\tau_8 \geq 240$, the $196560$ shortest vectors of the Leech lattice give $\tau_{24} \geq 196560$. The murderous half is the upper bound: to certify $\tau_n \leq M$ you must rule out a *continuum* of possible configurations of $M+1$ vectors with a single finite argument. That asymmetry is exactly why Newton and Gregory could dispute $\tau_3$ for centuries — the picture of $12$ is a drawing, the proof that $13$ is impossible took until 1953.

What I have for the upper-bound half is, honestly, almost nothing uniform. Schütte and van der Waerden's $\tau_3 \leq 12$ is a hand-to-hand fight on the $2$-sphere: take $13$ points, build the graph of near-$60$-degree pairs, triangulate, and bound the total spherical area against $4\pi$. It is correct but welded to $n=3$; in $n=4$ the spherical triangles become tetrahedra and the bookkeeping explodes. Coxeter's bound through the Schläfli function, completed by Böröczky, is genuinely general — $A(n,s)\leq 2F_{n-1}(\alpha)/F_n(\alpha)$ — but at $s=\tfrac12$ it is numerically loose; it only becomes sharp when the angle is small ($s\to 1$, where it nails the $600$-cell, $A(4,\cos\tfrac{\pi}{5})=120$). Fejes Tóth bounds the minimum distance for a fixed cardinality, which is the wrong direction: I want cardinality for a fixed angle. None of these can certify $\tau_8 \leq 240$ tightly enough to *meet* the $E_8$ construction. What I want is a single dimension-independent machine that eats the constraint $\langle x_i, x_j\rangle \leq \tfrac12$ and emits a number.

I propose the Delsarte linear-programming bound. The object that encodes a configuration is its Gram matrix $M = [\langle x_i, x_j\rangle]$, an $N\times N$ matrix with four properties: ones on the diagonal (unit vectors), off-diagonal entries $\leq \tfrac12$ (the constraint), positive semidefinite (it is a Gram matrix), and rank at most $n$ (the vectors live in $\mathbb{R}^n$ — the only place the dimension enters, so it had better be the lever). The one general move I can make is to apply a polynomial $f$ to the entries of $M$ and sum every entry, then bound that sum two different ways and trap $N$ between them. Split the diagonal off:
$$S = \sum_{i,j} f(\langle x_i, x_j\rangle) = N f(1) + \sum_{i\neq j} f(\langle x_i, x_j\rangle).$$
If $f(t)\leq 0$ for every $t$ in the window $[-1,\tfrac12]$ where the off-diagonal inner products live, every off-diagonal term is nonpositive and $S \leq N f(1)$ — an *upper* bound linear in $N$. For the lower bound I need a piece of $S$ that scales like $N^2$, and that has to come from the constant function: if $f$ contains a positive multiple of $1$, summing it over all $i,j$ gives that multiple times $N^2$. Quadratic $\leq$ linear forces $N$ to be bounded.

The crux is finding building-block functions $g$ of the inner product whose entry-sum $\sum_{i,j} g(\langle x_i,x_j\rangle)$ is guaranteed nonnegative for *every* point set, so that I can expand $f$ in them with nonnegative coefficients and discard everything but the constant term. A function is positive-definite on $S^{n-1}$ exactly when the matrix $[g(\langle x_i,x_j\rangle)]$ is PSD for all configurations, and a PSD matrix has nonnegative entry-sum (sandwich it between the all-ones vector: $\mathbf{1}^\top G\,\mathbf{1}\geq 0$). The geometry of the sphere hands me a basis of such functions: the **Gegenbauer (ultraspherical) polynomials** $P_k^{(n)}$, defined by $P_0^{(n)}=1$, $P_1^{(n)}=t$, the three-term recurrence
$$(k+n-2)\,P_{k+1}^{(n)}(t) = (2k+n-2)\,t\,P_k^{(n)}(t) - k\,P_{k-1}^{(n)}(t),$$
normalized so $P_k^{(n)}(1)=1$ and orthogonal on $[-1,1]$ under the weight $(1-t^2)^{(n-3)/2}$ — the measure obtained by pushing the uniform measure on $S^{n-1}$ down to the inner-product coordinate. (For $n=3$ these are Legendre polynomials, for $n=4$ Chebyshev of the second kind; the dimension sits right in the recurrence coefficients, so the machine is dimension-aware.) They are not pulled from a hat: they are the zonal spherical harmonics, and the **addition theorem** explains why their entry-sum is nonnegative. If $\{S_{k,l}\}_{l=1}^m$ is an orthonormal basis for the degree-$k$ harmonics, then
$$P_k^{(n)}(\langle x,y\rangle) = \frac{\omega_n}{m}\sum_{l=1}^m S_{k,l}(x)\,S_{k,l}(y),$$
which is literally the inner product of the feature vector $v_k(x)=(S_{k,1}(x),\dots,S_{k,m}(x))$ with $v_k(y)$ — a kernel. Summing over the configuration,
$$\sum_{i,j} P_k^{(n)}(\langle x_i, x_j\rangle) = \frac{\omega_n}{m}\sum_{l=1}^m\Big(\sum_i S_{k,l}(x_i)\Big)^2 \geq 0,$$
a sum of squares, nonnegative for any configuration with no constraint at all. And $P_0^{(n)}=1$ is the constant whose sum is exactly $N^2$ — my quadratic term.

Now the machine assembles. Write $f(t) = \sum_{k=0}^d c_k P_k^{(n)}(t)$ and impose two admissibility conditions: **(A1)** $f(t)\leq 0$ for all $t\in[-1,\tfrac12]$, and **(A2)** $c_0>0$ with $c_k\geq 0$ for $k\geq 1$. The lower count keeps only the constant term, $S = \sum_k c_k \sum_{i,j}P_k^{(n)} \geq c_0 N^2$ (every $k\geq 1$ term is a nonnegative coefficient times a nonnegative sum-of-squares); the upper count gives $S \leq N f(1)$. Chaining $c_0 N^2 \leq N f(1)$,
$$\tau_n = A(n,\tfrac12) \leq \frac{f(1)}{c_0}.$$
Any admissible $f$ gives a rigorous bound; I want the best one, minimizing $f(1)/c_0$, and for fixed degree $d$ this is a **linear program** over the coefficients $c_k$ (normalize $c_0=1$ and minimize $f(1)=\sum_k c_k$, since $P_k^{(n)}(1)=1$). A grid-sampled version of the sign condition on $[-1,s]$ is a fast way to *search* for a candidate, though the final certificate still needs the continuous sign condition proved by exact factor/sign reasoning or reliable interval arithmetic.

The beautiful part is that tightness *reads the optimal polynomial straight off the candidate configuration*, so I need not search blindly. For the bound to be exact both inequalities must be equalities. The upper one is tight iff $f$ vanishes at every inner product that actually occurs between distinct vectors of the optimal configuration; the lower one is tight iff $\sum_{i,j}P_k^{(n)}=0$ for every $k\geq 1$ with $c_k>0$, which in the relevant cases is exactly the spherical-design condition (harmonic moments vanish through the certificate degree). The roots of the optimal $f$ are therefore dictated by the geometry. There is one subtlety on multiplicities: a root in the *interior* of the window must have even multiplicity so that $f$ only touches zero from below and stays $\leq 0$ (a simple interior root would let $f$ change sign), while the window endpoints $-1$ and $\tfrac12$ may take simple roots since $f$ only needs to be $\leq 0$ on one side there. For $n=8$ the $E_8$ minimal vectors, normalized, have only the inner products $\{-1,-\tfrac12,0,\tfrac12\}$ — so simple roots at the ends, double roots at the interior values:
$$f_8(t) = (t+1)\,(t+\tfrac12)^2\, t^2\,(t-\tfrac12),$$
degree $6$. It is $\leq 0$ on $[-1,\tfrac12]$ ($(t+1)\geq 0$, two squares $\geq 0$, $(t-\tfrac12)\leq 0$); expanding in the dimension-$8$ Gegenbauer basis the coefficients all come out nonnegative with $c_0 = 3/320$, and $f_8(1)/c_0 = (9/4)/(3/320) = 240$. Meeting the $E_8$ construction, $\tau_8 = 240$, with a one-line polynomial certificate where the old proofs had none. For $n=24$ the Leech vectors carry one extra pair of inner products, $\{-1,-\tfrac12,-\tfrac14,0,\tfrac14,\tfrac12\}$, giving
$$f_{24}(t) = (t+1)\,(t+\tfrac12)^2\,(t+\tfrac14)^2\, t^2\,(t-\tfrac14)^2\,(t-\tfrac12),$$
degree $10$, $\leq 0$ on the window by the same sign accounting; its dimension-$24$ coefficients are nonnegative with (this normalization) $f_{24}(1)=2025/1024$, $c_0 = 15/1490944$, so $f_{24}(1)/c_0 = 196560$, meeting Leech: $\tau_{24}=196560$. The exactness is no accident — these configurations are so symmetric that only a handful of inner products occur and they are tight designs, which is precisely what snaps the LP certificate shut.

The same shape has a closed form via Jacobi-polynomial zeros, Levenshtein's universal polynomials. With $(\alpha,\beta)=(a+\tfrac{n-3}{2}, b+\tfrac{n-3}{2})$, $a,b\in\{0,1\}$, largest zeros $t_k^{a,b}$ partitioning $[-1,1)$ into intervals $I_{2k-1}=[t_{k-1}^{1,1}, t_k^{1,0}]$ and $I_{2k}=[t_k^{1,0}, t_k^{1,1}]$, the polynomial $f_m^{(n,s)}(t) = (t-s)(T_{k-1}^{1,0})^2$ for $m=2k-1$ or $(t+1)(t-s)(T_{k-1}^{1,1})^2$ for $m=2k$ — the same window-endpoint factors times a perfect square supplying the even interior roots — satisfies (A1),(A2) for $s\in I_m$ and yields $A(n,s)\leq L_m(n,s)$, e.g.
$$L_{2k-1}(n,s) = \binom{k+n-3}{k-1}\Big[\frac{2k+n-3}{n-1} - \frac{P_{k-1}^{(n)}(s)-P_k^{(n)}(s)}{(1-s)P_k^{(n)}(s)}\Big].$$
At $s=\tfrac12$ this recovers the exact cases, $L_6(8,\tfrac12)=240$ and $L_{10}(24,\tfrac12)=196560$, and within the degree ranges where the universal theorem applies (up to degree $m$, and up to $m+2$ in the strengthened form) it is the best possible pure-LP bound.

Pure LP nonetheless stalls in low dimensions. At $n=3$, $L_5(3,\tfrac12)\approx 13.285$ and higher-degree improvements only reach about $13.18$ — below $14$ but not below $13$; at $n=4$ no pure-LP polynomial beats $25$ (Arestov–Babenko). The obstruction is a flexible, non-tight optimum: in $n=3$ the $12$-point configurations form a positive-dimensional family, so a tight certificate would have to vanish on a whole *interval* of inner products, and a polynomial zero on a subinterval is identically zero. The fix, Musin's relaxation, recovers the off-diagonal terms I was discarding near $t=-1$. Relax (A1): allow $f>0$ near the antipode, requiring only $f\leq 0$ on $[t_0,\tfrac12]$ (with $-1\leq t_0<-\tfrac12$) and $f$ decreasing on $[-1,t_0]$. Now off-diagonal terms with inner product in $[t_0,\tfrac12]$ are still nonpositive and dropped, but for each fixed point the neighbors with inner product $\leq t_0$ lie in the opposite spherical cap, still obey the pairwise constraint, and so at most $\mu$ of them fit. Defining $h_m = \max\big[f(1) + \sum_{j=1}^m f(\langle e_1, y_j\rangle)\big]$ over $m$ cap points with $\langle e_1, y_j\rangle\leq t_0$ and pairwise $\langle y_i,y_j\rangle\leq\tfrac12$, the same two-way count gives
$$\tau_n \leq \frac{\max\{h_0, h_1, \dots, h_\mu\}}{c_0}.$$
The $h_m$ are honestly nonconvex sub-optimizations over cap configurations, so this is no longer a clean LP, but it is what closes the gap: with $t_0=-0.5907$, $\mu=4$, and a degree-$9$ polynomial the $n=3$ bound drops below $13$, and integrality plus the icosahedron's $12$ gives $\tau_3=12$ without the 1953 enumeration; with $t_0=-0.608$, $\mu=6$, and a degree-$9$ polynomial the $n=4$ bound matches the $24$-cell, settling $\tau_4=24$.

The other jaw — the lower-bound construction — is where intermediate dimensions are actually decided, because there the certificate sits far above any known configuration. The principled way to build configurations borrows the structure of error-correcting codes. From a binary $(n,M,d)$ code $C$, Construction A makes a lattice (integer points reducing mod $2$ to a codeword of $C$) and Construction B refines it (even weight, coordinate sum divisible by $4$); the contact count is read off the code's weight distribution. Construction A gives $2^d A_d(x)$ contacts if $d<4$, $2n+16A_4(x)$ if $d=4$, and $2n$ if $d>4$; Construction B gives $2^{d-1}A_d(x)$ if $d<8$, $2n(n-1)+128A_8(x)$ if $d=8$, and $2n(n-1)$ if $d>8$ — and in $n=24$ Construction B reproduces the even Leech lattice, closing the loop with the upper-bound story. Cross-sections, laminations, and code concatenation propagate records below $n=24$. In $n=11$ the laminated lattice $\Lambda_{11}$ gives $\tau_{11}\geq 582$ against an upper bound near $870$ — an open intermediate case where the record advances by better constructions, not by the certificate. The two jaws close on the same number in dimensions $8$ and $24$; everywhere in between, the gap is the open problem.

```python
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.optimize import linprog

def gegenbauer_n(n, kmax):
    """Dimension-n ultraspherical polynomials, normalized G_k(1)=1, by recurrence."""
    G = [np.array([1.0])]
    if kmax >= 1:
        G.append(np.array([0.0, 1.0]))
    for k in range(1, kmax):
        tGk = np.concatenate([[0.0], G[k]])
        gkm1 = np.zeros(len(tGk)); gkm1[:len(G[k-1])] = G[k-1]
        G.append(((2*k + n - 2)*tGk - k*gkm1) / (k + n - 2))
    return G[:kmax+1]

def gegenbauer_coeffs(fpoly, n):
    """Expand f (power-basis coeffs) in the Gegenbauer basis: returns c_0..c_d."""
    f = np.array(fpoly, dtype=float); d = len(f) - 1
    G = gegenbauer_n(n, d); c = np.zeros(d+1)
    for k in range(d, -1, -1):
        c[k] = f[k] / G[k][k]
        f[:k+1] -= c[k] * G[k][:k+1]
    return c

def polynomial_from_roots(roots_with_mult):
    poly = np.array([1.0])
    for r, m in roots_with_mult:
        for _ in range(m):
            poly = P.polymul(poly, [-r, 1.0])
    return poly

def sampled_lp_candidate(n, degree, s=0.5, grid=801):
    """Grid-relaxed LP search over f(t)=sum c_k G_k(t), normalized by c_0=1."""
    G = gegenbauer_n(n, degree)
    t = np.linspace(-1.0, s, grid)
    A = np.column_stack([P.polyval(t, g) for g in G])
    objective = np.ones(degree + 1)
    bounds = [(1.0, 1.0)] + [(0.0, None)] * degree
    res = linprog(objective, A_ub=A, b_ub=np.zeros(grid), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.fun, res.x

def certificate_bound(fpoly, n, s=0.5, grid=2001):
    """Numerically screen the Delsarte conditions and return f(1)/c_0."""
    poly = np.array(fpoly, dtype=float)
    c = gegenbauer_coeffs(poly, n)
    assert c[0] > 0 and np.all(c[1:] >= -1e-9), "Gegenbauer coefficients must be nonnegative"
    t = np.linspace(-1.0, s, grid)
    assert np.max(P.polyval(t, poly)) <= 1e-8, "f must be nonpositive on [-1, s]"
    return P.polyval(1.0, poly) / c[0]

print(certificate_bound(polynomial_from_roots(
    [(-1,1),(-0.5,2),(0.0,2),(0.5,1)]), 8))                              # 240.0
print(certificate_bound(polynomial_from_roots(
    [(-1,1),(-0.5,2),(-0.25,2),(0.0,2),(0.25,2),(0.5,1)]), 24))           # 196560.0
```
