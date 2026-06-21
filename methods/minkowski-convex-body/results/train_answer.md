The question I want to settle is when a region is *forced* to contain a lattice point other than the origin, judged from its size alone. Fix a full lattice $L \subset \mathbb{R}^n$ — all integer combinations $z_1 v_1 + \cdots + z_n v_n$ of $n$ independent vectors, the integer lattice $\mathbb{Z}^n$ being the basic case — and place a region $C$ with the origin among its points. Almost every existence problem in number theory wears this disguise: can a system of linear forms be made simultaneously small at integers? how well does a rational approximate a real? which primes are sums of two squares? Each reduces to *here is a region built from the data of the problem; produce an integer point inside it.* The trouble is that the lattice points of a curved or skew region admit no formula and cannot be enumerated. The only robust handle an arbitrary shape offers is its **volume**, so the question sharpens to: can a volume threshold alone guarantee a nonzero lattice point, and what is the threshold?

The existing tools all stall before this. Hermite's bound — that a positive-definite quadratic form $Q$ takes a value below a determinant-only quantity at some nonzero integer point, $\min_{x \in \mathbb{Z}^n \setminus 0} Q(x) \le c_n (\det Q)^{1/n}$ — is exactly the statement that an ellipsoid $\{Q \le c\}$ of large enough volume swallows a nonzero integer point. But its proof grinds the Gram matrix through reduction theory, the constant $c_n$ it yields is not sharp, and the whole argument is welded to the quadratic shape; it gives no clean geometric criterion. Direct construction (continued fractions, explicit identities for sums of squares) solves one problem at a time and dies on high-dimensional or skew regions. Dirichlet's pigeonhole principle is elementary and shape-free, but as stated it runs on the unit interval or a product of intervals — it exploits the *product* structure of a cube and has no evident way to see a curved or skew convex region. And a raw volume comparison gives nothing on its own: a thin needle of huge area threads between the lattice lines and contains only the origin, so "large volume $\Rightarrow$ lattice point" is simply false. The gap is the bridge from a measure inequality to an integer-coordinate point, and the missing piece is the right restriction on the shape.

I propose what I will call the **convex body theorem**. The right restriction is exactly two soft geometric words: $C$ must be **convex** (segments between its points stay inside) and **centrally symmetric** ($x \in C \Rightarrow -x \in C$). Both are forced. Drop convexity and a thin star of huge area dodges the lattice; drop symmetry and an off-center convex body of large volume puts only the origin on the lattice. So I expect the proof to *use both*, each in an essential place. The claim is then: if $L$ has covolume $d(L) = |\det(v_1,\dots,v_n)|$ and $C$ is convex and symmetric about the origin, then
$$\mathrm{vol}(C) > 2^n\, d(L) \ \Longrightarrow\ C \text{ contains a point of } L \text{ other than } 0,$$
and if $C$ is in addition compact, the weaker $\mathrm{vol}(C) \ge 2^n d(L)$ already suffices.

The threshold and its sharpness come from the cleanest case. For $\mathbb{Z}^n$ the open cube $\{\max_i |x_i| < 1\}$ has edge $2$ and volume $2^n$, yet every nonzero integer vector has some coordinate of absolute value at least $1$ and so sits on the boundary or outside — *not strictly inside*. Shrink the cube a hair and it still catches nothing. So whatever the threshold is, for $\mathbb{Z}^n$ it cannot dip below $2^n$, and scaling by the lattice gives $2^n d(L)$ in general. This open cube is also the standing witness that for an *open* body the inequality must be strict, and that equality demands compactness — the nonzero lattice points of the closed cube of volume exactly $2^n$ sit precisely on its boundary.

What makes it work is to **halve the body and fold it onto the lattice**. The factor $2^n$ is $2$ per dimension, which is the cue: pass to $\tfrac12 C$, whose volume is $2^{-n}\mathrm{vol}(C)$. If $\mathrm{vol}(C) > 2^n d(L)$ then $\mathrm{vol}(\tfrac12 C) > d(L)$ — more volume than one fundamental cell, which is the smell of pigeonhole. This is the engine, and I isolate it as a **fold lemma** that uses *neither* convexity nor symmetry, only measure and the lattice: if $S$ is any measurable set with $\mathrm{vol}(S) > d(L)$, there are distinct $x, y \in S$ with $x - y \in L$. The translates $\{u + F : u \in L\}$ of the fundamental cell $F$ tile $\mathbb{R}^n$ disjointly; cut $S$ into the pieces $S_u = S \cap (u+F)$, which are disjoint with union $S$, so $\sum_u \mathrm{vol}(S_u) = \mathrm{vol}(S) > \mathrm{vol}(F) = d(L)$. Translate each piece *back into the home cell*: $S_u - u \subseteq F$, volume unchanged, total still exceeding $\mathrm{vol}(F)$. They live inside one cell yet sum to more than its volume, so they cannot be pairwise disjoint — there is a point $a \in (S_u - u) \cap (S_v - v)$ with $u \ne v$. Unfolding, $x := a+u$ and $y := a+v$ are distinct points of $S$ with $x - y = u - v \in L \setminus \{0\}$. The lattice vector is hiding entirely in the *collision*; $x$ and $y$ are generic, but their difference is forced onto the lattice.

That gives a nonzero lattice vector $x - y$ from two points of $\tfrac12 C$, but I wanted it *inside $C$*, and this is where the two hypotheses earn their place, each used exactly once. Since $x, y \in \tfrac12 C$ we have $2x \in C$ and $2y \in C$. By **central symmetry**, $2y \in C \Rightarrow -2y \in C$ — symmetry supplies the second endpoint. Then
$$x - y = \tfrac12(2x) + \tfrac12(-2y)$$
is the *midpoint* of $2x$ and $-2y$, and by **convexity** the midpoint of two points of $C$ lies in $C$. So $x - y \in C$ is a nonzero lattice point. The halving was precisely the marriage: scaling by $\tfrac12$ is what makes $2x, 2y$ land back in $C$ and the difference come out as a midpoint rather than something at scale $2$ that would overshoot. And the $2^n$ is no longer mysterious — it is the cost of shrinking by $2$ in each of $n$ directions so that $\tfrac12 C$ can overflow one cell, exactly what the fold requires.

The compact equality case follows by a limiting argument that pins closedness as the deciding hypothesis. Suppose $C$ is compact with $\mathrm{vol}(C) = 2^n d(L)$ and contains no nonzero lattice point. For each $m \ge 1$ the dilate $(1 + 1/m)C$ has volume $(1+1/m)^n \cdot 2^n d(L) > 2^n d(L)$, so the strict form gives a nonzero lattice point $x_m \in (1+1/m)C \subseteq 2C$. But $(2C) \cap L$ is finite (bounded set, discrete lattice), so some single nonzero $x \in L$ equals $x_m$ for infinitely many $m$, i.e. $(1+1/m)^{-1} x \in C$ for infinitely many $m$; as $m \to \infty$ the closedness of $C$ forces $x \in C$ — a contradiction.

There is a second, sharper route by **packing** that delivers the same $2^n$ quantitatively and hands me the equality case for free. Describe $C$ by its gauge $f(x) = \min\{\lambda \ge 0 : x \in \lambda C\}$, a norm carrying exactly the three properties I keep using — homogeneity $f(tx) = t f(x)$ from dilation, the triangle inequality $f(x+y) \le f(x)+f(y)$ which *is* convexity, and $f(-x) = f(x)$ which *is* central symmetry — so $\{f < r\} = rC$ and $\mathrm{vol}\{f < r\} = r^n \mathrm{vol}(C)$. Let $M = \min_{l \in L \setminus 0} f(l)$, attained because the lattice is discrete in $f$. The bodies $B_a = \{x : f(x-a) < M/2\}$ over $a \in L$ pack disjointly: if $b \in B_a \cap B_c$ with $a \ne c$, then $f(a-c) \le f(a-b) + f(b-c) < M$, contradicting $f(a-c) \ge M$. Packing the $(\Omega+1)^n$ such bodies around the lattice points with coordinates in $\{0, \pm1, \dots, \pm\Omega/2\}$ ($\Omega$ even) into a centered cube of edge $\Omega + O(1)$ and letting $\Omega \to \infty$ yields $(M/2)^n \mathrm{vol}(C) \le d(L)$, that is
$$M^n \cdot \mathrm{vol}(C) \le 2^n d(L), \qquad \min_{l \in L \setminus 0} f(l) \le 2\,(d(L)/\mathrm{vol}(C))^{1/n}.$$
So $\mathrm{vol}(C) \ge 2^n d(L) \Rightarrow M \le 1$ (a nonzero lattice point in the closed $C$) and $\mathrm{vol}(C) > 2^n d(L) \Rightarrow M < 1$ (one strictly inside), recovering both parts with a sharp distance bound.

The payoff is that Hermite returns as one shape among infinitely many. With $f = \sqrt{Q}$ the unit body is the ellipsoid $\{Q < 1\}$ and the bound reads $\min_{\mathbb{Z}^n \setminus 0} Q \le 4\,(\mathrm{vol}\{Q<1\})^{-2/n}$ — a determinant-only bound on positive-definite quadratic forms with a transparent geometric constant, no quadratic algebra anywhere. The Diophantine problems become uniform. For linear forms $L_i(x) = \sum_j b_{ij} x_j$ with $\det B \ne 0$ and $\prod_i A_i \ge |\det B|$, the region $P = \{|L_i| \le A_i\} = B^{-1}R$ has volume $|\det B|^{-1} \cdot 2^n \prod A_i \ge 2^n$ and is convex and symmetric, so the compact form yields a nonzero integer $x$ with $|L_i(x)| \le A_i$ for all $i$; applied to $X_i - \alpha_i X_{n+1}$ and $X_{n+1}$ (determinant $1$) with bounds $Q^{-1/n}$ and $Q$ it gives Dirichlet's $|\alpha_i - x_i/q| \le q^{-1-1/n}$, infinitely many when some $\alpha_i$ is irrational. Choosing the body tunes the constant: the cube gives a nonzero vector with $\|x\|_\infty \le d(L)^{1/n}$, hence the Hermite constant $\gamma_n \le n$, while a disk in the plane gives the sharper $\gamma_2 \le 4/\pi$. And for a prime $p \equiv 1 \pmod 4$, picking $r$ with $r^2 \equiv -1 \pmod p$ and applying the theorem to the ellipse $\{(pz_1 + rz_2)^2 + z_2^2 < 2p\}$ (area $> 4$) forces a nonzero $(z_1,z_2)$ with $0 < (pz_1+rz_2)^2 + z_2^2 < 2p$, where the congruence $(pz_1+rz_2)^2 + z_2^2 \equiv (r^2+1)z_2^2 \equiv 0 \pmod p$ pins the value to exactly $p$, so $p$ is a sum of two squares; the four-dimensional analogue gives Lagrange's four-square theorem. Each was a separate clever construction before; now each is "write the region, check its volume clears $2^n d(L)$, read off the integer point."

A small checker confirms the threshold on concrete bodies — the criterion is "compute $\mathrm{vol}(C)$, compare with $2^n d(L)$; if it clears, the body must contain a nonzero lattice point, found by the fold":

```python
import numpy as np
from itertools import product

def covolume(basis):                      # d(L) = |det(v_1,...,v_n)|
    return abs(np.linalg.det(basis))

def body_volume(C, box_radius, n, samples=400000):   # Monte-Carlo vol(C)
    pts = np.random.uniform(-box_radius, box_radius, size=(samples, n))
    return np.mean([C(p) for p in pts]) * (2 * box_radius) ** n

def guarantees_nonzero_lattice_point(C, basis, box_radius, n):
    # The criterion: vol(C) >= 2^n * d(L) (compact body) forces a nonzero lattice point.
    return body_volume(C, box_radius, n) >= 2 ** n * covolume(basis)

def witness_nonzero_lattice_point(C, basis, coord_range):
    # When the criterion holds the theorem guarantees this search succeeds.
    n = basis.shape[1]
    for z in product(range(-coord_range, coord_range + 1), repeat=n):
        z = np.array(z)
        if np.any(z != 0) and C(basis @ z):
            return basis @ z
    return None

# Instance: a disk against Z^2 (threshold 2^2 * 1 = 4; the area-4 disk is the sharp
# compact case, with (+-1,0) on/inside it -- here we take area 4.2 > 4 to clear it).
I2   = np.eye(2)
disk = lambda x: x[0]**2 + x[1]**2 <= 4.2 / np.pi        # area = pi r^2 = 4.2 > 4
assert guarantees_nonzero_lattice_point(disk, I2, 2.0, 2)
assert witness_nonzero_lattice_point(disk, I2, 3) is not None   # e.g. (1,0)

# Sharpness: the open square of area (2-eps)^2 < 4 has no nonzero lattice point.
sq = lambda x: max(abs(x[0]), abs(x[1])) < 1 - 1e-9
assert witness_nonzero_lattice_point(sq, I2, 3) is None
```
