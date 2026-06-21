# High-Dimensional Walks and an FPRAS for Counting Bases of a Matroid

## Problem

Given a matroid $M=([n],\mathcal I)$ of rank $r$ through an independence oracle, estimate the
number of bases $|\mathcal B|$ to a $(1\pm\varepsilon)$ factor in time polynomial in
$n,r,1/\varepsilon,\log(1/\delta)$ (an FPRAS). Equivalently (self-reducibility +
Jerrum–Valiant–Vazirani), sample an almost-uniform random basis in polynomial time. The
natural candidate is the bases-exchange walk; the question is whether it mixes for *every*
matroid. The same question is the 1989 Mihail–Vazirani conjecture that the bases-exchange
graph has edge expansion $\ge 1$.

## Key idea

Encode a $d$-homogeneous distribution $\mu$ by its multiaffine generating polynomial
$g_\mu(x)=\sum_S\mu(S)\,x^S$ and build a weighted simplicial complex $X^{g_\mu}$ (maximal
faces $=\mathrm{supp}(\mu)$, $w(S)=\mu(S)$, lower faces by balanced extension). This sets up
a dictionary

| weighted complex | polynomial |
|---|---|
| dimension $d$ | degree $d$ |
| weight of $\emptyset$ | evaluation at $\mathbf 1$ |
| link of $\tau$ | derivative $\partial_\tau$ |
| local random walk on link's $1$-skeleton | (normalized) Hessian of the derivative |

Under it, **the non-lazy local walk of the link at $\tau$ equals the normalized Hessian
$\tilde\nabla^2 p_\tau=\frac{1}{d-k-1}\mathrm{diag}(\nabla p_\tau(\mathbf 1))^{-1}\nabla^2
p_\tau(\mathbf 1)$**, where $p_\tau=\prod_{i\in\tau}\partial_i\,g_\mu$. A polynomial is
log-concave iff its Hessian has at most one positive eigenvalue; *strong* log-concavity (the
same for every derivative) is therefore **exactly** the statement that every link walk has
its second eigenvalue $\le 0$, i.e. the complex is a $0$-local spectral expander — the
strongest case of the one-sided high-dimensional-expander condition (negative link
eigenvalues are irrelevant). One-sided local-to-global expansion then gives the top walk
spectral gap $\ge 1/d$. For matroids, $g_M$ is strongly log-concave, so $d=r$ gives a
polynomial mixing time, an FPRAS, and (via Cheeger) expansion $\ge 1$.

## Main theorem and proof

**Theorem.** Let $\mu:2^{[n]}\to\mathbb R_{\ge0}$ be a $d$-homogeneous strongly log-concave
distribution. The down-up walk $\mathcal M_\mu$ (from $\tau$: drop a uniform element, then
re-add proportional to $\mu$) has spectral gap $\ge 1/d$; for $\tau\in\mathrm{supp}(\mu)$ and
$0<\varepsilon<1$, $\ t_\tau(\varepsilon)\le d\log\frac{1}{\varepsilon\,\mu(\tau)}.$

**Walks.** On a pure $d$-complex with balanced $w$, the up-walk $P_k^{\wedge}$ on $X(k)$ and
down-walk $P_{k+1}^{\vee}$ on $X(k+1)$ are the two diagonal blocks of the square of the
bipartite walk between $X(k)$ and $X(k+1)$; both are stochastic, $w$-self-adjoint, PSD, and
share their nonzero spectrum. $\mathcal M_\mu=P_d^{\vee}$, so
$\lambda^*(\mathcal M_\mu)=\lambda^*(P_{d-1}^{\wedge})$.

**Linear-algebra facts.** (i) Congruence preserves "$\le 1$ positive eigenvalue" (write
$A=B+vv^\intercal$, $B\preccurlyeq0$; Cauchy interlacing). (ii) If $A$ has $\le1$ positive
eigenvalue and $B\succcurlyeq0$, so does $BA$ ($B=C^\intercal C$, $BA\sim CAC^\intercal$).
(iii) Symmetric entrywise-nonnegative $A$ with $\le1$ positive eigenvalue and row sums $w$
satisfies $A\preccurlyeq ww^\intercal/\sum_i w(i)$.

**$0$-local SE $\Rightarrow$ gap (Loewner + induction).** For each codim-one face $\eta$,
$0$-local spectral expansion plus (iii) gives $A_\eta\preccurlyeq w_\eta w_\eta^\intercal/w(\eta)$,
which sums over $\eta$ to
$$P_k^{\wedge}\preccurlyeq \tfrac{k}{k+1}P_k^{\vee}+\tfrac{1}{k+1}I.$$
By induction on $k$, using that $P_{k+1}^{\vee}$ shares the nonzero spectrum of
$P_k^{\wedge}$ and pushing eigenvalue counts through $t\mapsto\frac{k+1}{k+2}t+\frac{1}{k+2}$:
for $-1\le i\le k$, $P_k^{\wedge}$ has $\le|X(i)|$ eigenvalues above $1-\frac{i+1}{k+1}$. In
particular $\lambda_2(P_k^{\wedge})\le\frac{k}{k+1}$, so $\lambda^*(\mathcal M_\mu)\le1-1/d$.
Diaconis–Stroock gives the mixing-time bound.

**Strong log-concavity $\Rightarrow$ $0$-local SE.** With $w(\tau)=(d-k)!\,p_\tau(\mathbf 1)$
(Euler's identity, induction), the link walk $\tilde P^{\wedge}_{\tau,1}$ equals
$\tilde\nabla^2 p_\tau$ entrywise. Strong log-concavity makes $\nabla^2 p_\tau(\mathbf 1)$
have $\le1$ positive eigenvalue; by (ii) so does $\tilde\nabla^2 p_\tau=\tilde
P^{\wedge}_{\tau,1}$, i.e. $\lambda_2\le0$. $\square$

**Corollary (matroids).** $g_M=\sum_{B}\boldsymbol\lambda^B x^B$ is strongly log-concave (see
below), so the bases-exchange walk on a rank-$r$ matroid mixes in
$t_B(\varepsilon)\le r\log(n^r/\varepsilon)\le r^2\log(n/\varepsilon)$ steps, using only the
oracle. With JVV self-reducibility this gives an FPRAS for $|\mathcal B|$, for the number of
size-$k$ independent sets (truncate to rank $k$), the reliability polynomial $C_M(p)$, and —
for $0<q\le1,\ p\ge0$ — the random-cluster partition function $Z_M(p,q)$ and the Tutte
polynomial on $\{y\ge1,\ 0\le(x-1)(y-1)\le1\}$.

**Theorem (Mihail–Vazirani).** For every matroid, $h(G_M)\ge1$. *Proof.* The weighted graph
$H_M$ of $P_r^{\vee}$ has $\lambda_2\le1-1/r$, so Cheeger gives $\phi(H_M)\ge\frac{1}{2r}$.
Each off-diagonal weight is $P_r^{\vee}(\tau,\tau')=\frac{w(\tau')}{r\,w(\tau\cap\tau')}\le\frac{1}{2r}$
(balancedness: $w(\tau\cap\tau')\ge2$). Hence for $|S|\le|\mathcal B|/2$,
$\frac{1}{2r}\le\phi(S)\le\frac{1}{2r}\cdot\frac{|E(S,\bar S)|}{|S|}=\frac{h(S)}{2r}$,
so $h(S)\ge1$. $\square$

## Self-contained strong log-concavity (no Hodge theory needed)

**Criterion.** A $d$-homogeneous $p$ is strongly log-concave at $\mathbf 1$ if (1) every
derivative $\partial_{i_1}\cdots\partial_{i_k}p$ ($k\le d-2$) is indecomposable, and (2)
every $(d-2)$-th derivative quadratic is log-concave at $\mathbf 1$.

*Proof.* Induct on $\deg p$; assume each $p_i=\partial_i p$ strongly log-concave. For an
eigenpair $(\mu,\phi)$ of the normalized Hessian $\tilde\nabla^2 p$, split $\phi$ along
$\mathbf 1$ in each inner product $\langle\phi,\psi\rangle_{p_k}=(d-1)\sum_j\phi(j)\psi(j)\partial_j p_k(\mathbf 1)$;
since each $\tilde\nabla^2 p_k$ has its single positive eigenvalue in the $\mathbf 1$
direction and (Euler) $\frac{\langle\phi,\mathbf 1\rangle_{p_k}}{\langle\mathbf 1,\mathbf 1\rangle_{p_k}}=\mu\phi(k)$,
one gets $\mu\|\phi\|_p^2\le\mu^2\|\phi\|_p^2$, so $\mu\le0$ or $\mu=1$. Indecomposability
makes the graph of $\nabla^2 p$ connected, so $1$ is simple (Cheeger): exactly one positive
eigenvalue, i.e. log-concavity. $\square$

**Apply to $g_M$.** $\partial_S g_M=g_{M/S}$. For $|S|<r$, $M/S$ has rank $\ge2$ and the
exchange axiom makes $g_{M/S}$ indecomposable. For $|S|=r-2$, $g_{M/S}$ is quadratic with
$(\nabla^2 g_{M/S})_{ij}=\lambda_i\lambda_j$ if $\{i,j\}$ independent in $M/S$ else $0$; by
the matroid partition property (parallel classes $B_1,\dots,B_m$),
$\nabla^2 g_{M/S}=\boldsymbol\lambda_B\boldsymbol\lambda_B^\intercal-\sum_t\boldsymbol\lambda_{B_t}\boldsymbol\lambda_{B_t}^\intercal\preccurlyeq\boldsymbol\lambda_B\boldsymbol\lambda_B^\intercal$,
rank-one PSD $\Rightarrow$ $\le1$ positive eigenvalue. Hence $g_M$ is strongly log-concave.

**Random cluster ($0<q\le1$).** For $f_{M,k,q}=\sum_{|S|=k}q^{-\mathrm{rank}(S)}\boldsymbol\lambda^S x^S$,
the quadratic $A=q^{\mathrm{rank}(S)}\nabla^2\partial_S f$ satisfies $A\preccurlyeq vv^\intercal$
with $v_i=\lambda_i$ on loops, $q^{-1}\lambda_i$ on non-loops, since
$vv^\intercal-A=(q^{-2}-q^{-1})\sum_t\boldsymbol\lambda_{B_t}\boldsymbol\lambda_{B_t}^\intercal\succcurlyeq0$.

**Geometric scaling.** If $f=\sum_S c_S x^S$ is multiaffine homogeneous SLC then
$f_\alpha=\sum_S c_S^\alpha x^S$ is SLC for $0\le\alpha\le1$. Writing $\nabla^2\partial_T f=vv^\intercal-A$
($v>0$ by Perron–Frobenius), $c^\alpha=(v_iv_j-A_{ij})^\alpha$ Taylor-expands with all
$x^m$ ($m\ge1$) coefficients negative; Schur product theorem makes $\nabla^2\partial_T f_\alpha=$
(rank-one PSD) $-$ (PSD). This unifies $\det(L_S)^\alpha$ for $k$-DPPs, $\alpha\in[0,1]$.

## Algorithm (sampler + FPRAS)

```python
import numpy as np

class Matroid:
    def __init__(self, n, independent_fn):
        self.n = n
        self.independent = independent_fn          # frozenset -> bool

def sample_basis(M, B0, eps, rng):
    """Almost-uniform basis via the bases-exchange (down-up) walk.
       Strong log-concavity of g_M => spectral gap >= 1/r => this many steps suffice."""
    r = len(B0)
    T = int(np.ceil(r * (r * np.log(max(M.n, 2)) + np.log(1.0 / eps))))  # r*log(n^r/eps)
    B = set(B0)
    for _ in range(T):
        i = rng.choice(list(B)); B.remove(i)        # drop a uniform element
        add = [j for j in range(M.n)
               if j not in B and M.independent(frozenset(B | {j}))]
        B.add(rng.choice(add))                       # re-add a uniform completion
    return frozenset(B)

def count_bases(M, basis, eps, delta, rng):
    """FPRAS: telescoping marginals along contractions (self-reducible).
       |B(M)| = prod 1/p_e, each marginal p_e = Pr[e in a uniform basis] estimated
       by sampling; recurse on M/e; boost with median-of-means for the (1-delta) bound."""
    raise NotImplementedError   # standard JVV reduction; rapid mixing is what makes it run
```

Only the independence oracle is used; the contribution is the certificate that the walk mixes
in polynomial time, obtained from the polynomial$\leftrightarrow$complex dictionary, the
one-sided local-to-global spectral bound, and the elementary strong-log-concavity criterion.
