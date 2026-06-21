# Context: the sphere packing problem and the search for a sharp certificate

## Research question

How densely can equal non-overlapping balls fill Euclidean space $\mathbb{R}^d$? Make $X\subset\mathbb{R}^d$ a set of centers with pairwise distance $\geq 2$ (unit balls); the packing is $\mathcal{P}=\bigcup_{x\in X}B_d(x,1)$, and its density is

$$\Delta_{\mathcal{P}}=\limsup_{r\to\infty}\frac{\mathrm{Vol}(\mathcal{P}\cap B_d(0,r))}{\mathrm{Vol}(B_d(0,r))}.$$

The sphere packing constant is $\Delta_d=\sup_{\mathcal{P}}\Delta_{\mathcal{P}}$. The goal is not merely to exhibit a dense packing — good lattices have been known for over a century — but to **prove it optimal**: to produce an upper bound on $\Delta_d$ that exactly equals the density of a known packing.

Exact values are known only in dimensions $1$ ($\Delta_1=1$), $2$ (the hexagonal lattice, $\Delta_2=\pi/\sqrt{12}\approx0.9069$, proved by Thue and rigorously by Fejes Tóth), and $3$ (the Kepler conjecture, $\Delta_3=\pi/\sqrt{18}\approx0.7405$, proved by Hales by a massive computer-assisted exhaustion). Two dimensions stand out empirically: in $\mathbb{R}^8$ the $E_8$ root lattice and in $\mathbb{R}^{24}$ the Leech lattice are conjectured optimal, and they are extraordinarily symmetric and rigid.

## Background

**Lattices, dual lattices, and the figures of merit.** A lattice $\Lambda\subset\mathbb{R}^d$ has covolume $\mathrm{covol}(\Lambda)$ and dual $\Lambda^*=\{t:\ t\cdot x\in\mathbb{Z}\ \forall x\in\Lambda\}$ with $\mathrm{covol}(\Lambda^*)=1/\mathrm{covol}(\Lambda)$. A lattice is unimodular if $\mathrm{covol}=1$, self-dual (isodual) if $\Lambda^*=\Lambda$ up to isometry, even if $\|x\|^2\in2\mathbb{Z}$ for all $x$. The center density is $\delta=\Delta_{\mathcal{P}}/\mathrm{Vol}(B_d(0,1))$.

The $E_8$ lattice is
$$\Lambda_8=\Big\{(x_i)\in\mathbb{Z}^8\cup(\mathbb{Z}+\tfrac12)^8:\ \textstyle\sum_i x_i\equiv0\ (\mathrm{mod}\ 2)\Big\},$$
the unique even unimodular lattice of rank $8$; its minimal vector length is $\sqrt2$, and its vectors have lengths $\sqrt{2n}$, $n=0,1,2,\dots$. The Leech lattice is the unique even unimodular rank-$24$ lattice with no vectors of length $\sqrt2$; its minimal length is $2$, with vectors of length $\sqrt{2n}$, $n=2,3,\dots$, and each point has $196560$ nearest neighbors. Both are self-dual, and their vector lengths sit on the arithmetic progression $\sqrt{2n}$.

**Poisson summation** is the central analytic identity. For a nice radial $f$ and a lattice $\Lambda$,
$$\sum_{x\in\Lambda}f(x)=\frac{1}{\mathrm{covol}(\Lambda)}\sum_{t\in\Lambda^*}\widehat f(t),\qquad \widehat f(y)=\int_{\mathbb{R}^d}f(x)e^{-2\pi i x\cdot y}\,dx.$$
It is the only available bridge that turns "no two centers closer than the minimal distance" together with lattice periodicity into a *single scalar inequality* relating $f$ on $\Lambda$ to $\widehat f$ on $\Lambda^*$.

**The Gaussian is a fixed point of the Fourier transform, with a modular twist.** In $\mathbb{R}^d$,
$$\mathcal{F}\big(e^{\pi i\|x\|^2 z}\big)(y)=z^{-d/2}\,e^{\pi i\|y\|^2(-1/z)},\qquad z\in\mathbb{H}.$$
The Fourier transform sends the parameter $z$ to $-1/z$ and multiplies by $z^{-d/2}$.

**Modular and quasimodular forms.** $\mathrm{SL}_2(\mathbb{Z})$ acts on the upper half-plane $\mathbb{H}$ by $\gamma z=(az+b)/(cz+d)$; the slash operator is $(F|_k\gamma)(z)=(cz+d)^{-k}F(\gamma z)$. A modular form of weight $k$ for a congruence subgroup $\Gamma$ satisfies $F|_k\gamma=F$ for $\gamma\in\Gamma$ and is holomorphic (including at cusps); weakly holomorphic allows poles at cusps. The basic forms:
$$E_4=1+240\sum_{n\geq1}\sigma_3(n)q^n,\quad E_6=1-504\sum_{n\geq1}\sigma_5(n)q^n,\quad q=e^{2\pi iz},$$
$$\Delta=\frac{E_4^3-E_6^2}{1728}=q-24q^2+\cdots\ (\text{nonvanishing on }\mathbb{H}),\qquad j=\frac{1728\,E_4^3}{E_4^3-E_6^2}=q^{-1}+744+\cdots.$$
The weight-$2$ Eisenstein series $E_2=1-24\sum\sigma_1(n)q^n$ is **quasimodular**: it fails modularity by an anomaly,
$$z^{-2}E_2(-1/z)=E_2(z)-\frac{6i}{\pi}\,\frac1z.$$
The Jacobi thetanull functions $\theta_{00}=\sum_n e^{\pi i n^2 z}$, $\theta_{01}=\sum_n(-1)^n e^{\pi i n^2 z}$, $\theta_{10}=\sum_n e^{\pi i(n+1/2)^2 z}$ satisfy the Jacobi identity $\theta_{01}^4+\theta_{10}^4=\theta_{00}^4$ and have explicit $S,T$ transformation laws on their fourth powers; the theta series of a lattice, $\Theta_\Lambda(z)=\sum_{x\in\Lambda}e^{\pi i\|x\|^2 z}$, is modular of weight $d/2$.

**The diagnostic empirical finding.** Numerically optimized auxiliary functions (described below) reveal that in $\mathbb{R}^8$ the best attainable upper bound exceeds the $E_8$ density by a factor of only about $1.000001$, and in $\mathbb{R}^{24}$ exceeds the Leech density by about $1.000707$; later refinements push the $d=8$ gap below $1+10^{-28}$. In every other dimension $4\leq d\leq36$ the bound and the best packing stay visibly apart. This near-coincidence in exactly $8$ and $24$ — and nowhere else — is the phenomenon that suggests an *exactly sharp* certificate exists there.

## Baselines

**Delsarte's linear programming bound for codes (Delsarte 1972; Delsarte–Goethals–Seidel 1977).** For error-correcting codes on the Hamming scheme one bounds the maximum code size by choosing an auxiliary function that is a nonnegative combination of the Krawtchouk (zonal) polynomials and is nonpositive at the allowed distances; positivity in the "dual" (transform) domain plus nonpositivity in the "primal" domain yields a single linear functional bounding the code. It is a genuine linear program over the cone of admissible functions, and in exceptional, highly symmetric cases it is sharp.

**Kabatiansky–Levenshtein (1978).** They transported the spherical-code LP bound to Euclidean packings by relating packing density to codes on spheres of growing radius. This gives the best known asymptotic exponents.

**The Cohn–Kumar universal-optimality line (2007) and the theta/modular bound (Mallows–Odlyzko–Sloane-style arguments).** Theta-series/modular bounds (e.g. for the minimal norm of unimodular lattices) show modular forms can bound lattice invariants, and Cohn–Kumar's work on universally optimal point configurations shows that for some exceptional configurations LP-type bounds become *exactly* sharp.

**Direct geometric exhaustion (Hales, $d=3$).** Sharp in dimension $3$: a finite case analysis of local clusters that produces an exact certificate for the Kepler conjecture.

## Evaluation settings

The yardstick is the comparison, in a fixed dimension $d$, between an upper bound on $\Delta_d$ and the density of the best known packing (the lower bound). The natural test dimensions are $d=8$ (compare against $E_8$, density $\pi^4/384$) and $d=24$ (compare against Leech, density $\pi^{12}/12!$), with $d=1,2,3$ as sanity checks where the answer is known, and $4\leq d\leq36$ as the range where numerical bounds are tabulated. The relevant objects — the lattices $E_8$ and Leech, their theta series, their vector-length spectra, the modular forms $E_4,E_6,E_2,\theta_{ij},\Delta,j$, and the Poisson summation formula — all predate any certificate and are the fixed ground on which it is judged. The figure of merit is the ratio (upper bound)/(lower bound); a value of exactly $1$ is a proof of optimality.

## Code framework

Two computational tools exist beforehand. The first is a numerical optimizer for the auxiliary-function bound: parametrize a radial function, impose its sign conditions at sample radii, and solve the resulting finite linear program to see how close the bound gets to a target lattice. The second is a modular-forms toolkit (PARI/GP, SageMath, or `mpmath`) that can build $E_4,E_6,E_2,\Delta,j$, the thetanulls, and their $q$-expansions, and integrate them. The certificate, once found, will be filled into the stubs below.

```python
import numpy as np
from scipy.optimize import linprog
import mpmath as mp

# ---- Pre-existing: radial Fourier transform in R^d (Hankel transform) ----
# F(f)(s) = 2*pi*s^{-(d-2)/2} * \int_0^infty f(r) J_{(d-2)/2}(2*pi*r*s) r^{d/2} dr
def radial_ft(f, s, d, R=20.0, N=4000):
    r = np.linspace(1e-6, R, N)
    from scipy.special import jv
    nu = (d - 2) / 2
    integrand = f(r) * jv(nu, 2*np.pi*r*s) * r**(d/2)
    return 2*np.pi * s**(-nu) * np.trapz(integrand, r)

# ---- Pre-existing: the auxiliary-function LP, parametrized as Gaussian*poly ----
def auxiliary_function_bound(d, degree, radii):
    """Optimize a bound on the packing constant by choosing a radial f.
    Returns the best ratio f(0)/fhat(0) over admissible f.  This is the
    KNOWN numeric tool; it reveals d=8,24 are special but never reaches 1."""
    # TODO: build basis f_k(r) = L_k(2*pi*r^2) e^{-pi r^2} (Laguerre x Gaussian),
    #       assemble linear constraints f<=0 (r>=1) and fhat>=0, solve the LP.
    pass

# ---- The certificate to be discovered (one big empty slot) ----
def certificate(x_norm, d):
    """The radial function whose existence and closed form must be found."""
    # TODO: the construction.
    pass

def fourier_transform_of_certificate(x_norm, d):
    # TODO.
    pass

def verify_sign_conditions(g, ghat, d):
    # TODO.
    pass
```
