# Context

## Research question

Let $M = ([n], \mathcal I)$ be a matroid of rank $r$, presented only by an independence
oracle (a black box that, given $S \subseteq [n]$, answers whether $S \in \mathcal I$).
Write $\mathcal B$ for the set of bases (maximal independent sets, all of size $r$). We
want to estimate $|\mathcal B|$ — the number of bases — to within a $(1\pm\varepsilon)$
multiplicative factor, in time polynomial in $n$, $r$, $1/\varepsilon$, and
$\log(1/\delta)$ (a *fully polynomial-time randomized approximation scheme*, FPRAS).

This is a central counting problem. For graphic matroids the bases are the spanning trees
of a graph (countable exactly by Kirchhoff's matrix-tree theorem), but for a general
matroid given by an oracle the exact count is $\#P$-hard, and even with oracle access there
is a hard barrier: any *deterministic* algorithm making polynomially many independence
queries cannot beat a $2^{\Omega(r/\log n)}$ approximation factor. So the question is
sharply about *randomized* approximation.

The standard route to an FPRAS for a counting problem is via sampling. The problem is
*self-reducible*: deleting an element (restriction) and contracting an element ($M/i$) both
yield smaller matroids, so the count over $M$ decomposes into counts over minors. By the
sampling–counting equivalence for self-reducible problems, an FPRAS for $|\mathcal B|$ is
equivalent to a *fully polynomial almost-uniform sampler* (FPAUS) that draws a basis from a
distribution within total-variation $\varepsilon$ of uniform in polynomial time. So the
real question becomes: **is there a polynomial-time almost-uniform sampler for a random
basis of an arbitrary matroid?** The most natural candidate is a simple local Markov chain
on bases.

A closely related structural question, open since 1989, is whether the *bases-exchange
graph* of any matroid is a good expander — Mihail and Vazirani conjectured its edge
expansion is at least $1$. Rapid mixing of the natural chain and expansion of this graph
are two faces of the same coin.

## Background

**Matroids and the bases-exchange graph.** A matroid satisfies the hereditary property
($S \subseteq T \in \mathcal I \Rightarrow S \in \mathcal I$) and the exchange axiom
($S, T \in \mathcal I$, $|T| > |S| \Rightarrow \exists\, i \in T\setminus S$ with
$S \cup \{i\} \in \mathcal I$). All bases have size $r$. The *bases-exchange graph* $G_M$
has a vertex per basis and an edge between $B, B'$ when $|B \triangle B'| = 2$, i.e. $B'$ is
obtained from $B$ by swapping one element out and one in. The exchange axiom makes $G_M$
connected. It is exactly the $1$-skeleton of the *matroid base polytope*
$P_M = \mathrm{conv}\{\mathbf 1_B : B \in \mathcal B\}$ (every edge of $P_M$ is parallel to
$\mathbf 1_i - \mathbf 1_j$). The *contraction* $M/S$ has ground set $[n]\setminus S$ and
independent sets $\{T : T \cup S \in \mathcal I\}$; it is again a matroid, and its bases are
exactly the bases of $M$ containing $S$, shifted.

**The natural Markov chain (down–up / bases-exchange walk).** From a basis $B$: pick an
element $i \in B$ uniformly at random and remove it; then, among all $j$ with $B - i + j$ a
basis, pick one (uniformly, in the unweighted case) and move there. This chain is reversible
with uniform stationary distribution. Sampling a uniform basis reduces to running this chain
long enough — *if* it mixes in polynomially many steps.

**Mixing, conductance, and Cheeger.** For a reversible chain with transition matrix $P$ and
stationary $\pi$, the mixing time is governed by the spectral gap $1 - \lambda^*(P)$, where
$\lambda^*(P) = \max\{|\lambda_2|, |\lambda_n|\}$ is the second-largest eigenvalue in
absolute value: $t_\tau(\varepsilon) \le \frac{1}{1-\lambda^*(P)} \log\!\frac{1}{\varepsilon
\pi(\tau)}$ (Diaconis–Stroock 1991). For a weighted graph the *conductance* of a set $S$ is
$\phi(S) = \frac{w(E(S,\bar S))}{\mathrm{vol}(S)}$, and Cheeger's inequality
$\frac{d - \lambda_2}{2} \le \phi(G) \le \sqrt{2(d - \lambda_2)}$ relates the conductance to
the spectral gap of the (weighted) adjacency matrix. The *edge expansion* of $G_M$ is
$h(S) = |E(S,\bar S)|/|S|$ minimized over $|S| \le |\mathcal B|/2$.

**Generating polynomials of set distributions.** A distribution $\mu : 2^{[n]} \to
\mathbb R_{\ge 0}$ supported on size-$d$ sets is encoded by its *generating polynomial*
$g_\mu(x_1,\dots,x_n) = \sum_{S} \mu(S)\, x^S$, where $x^S = \prod_{i\in S} x_i$. This is a
$d$-homogeneous *multiaffine* polynomial (degree $\le 1$ in each variable). For a matroid,
the uniform measure on bases has generating polynomial $g_M(x) = \sum_{B \in \mathcal B}
x^B$, the *bases generating polynomial*; weighting element $i$ by $\lambda_i > 0$ gives
$g_M(x) = \sum_B \boldsymbol\lambda^B x^B$. Many counting quantities are evaluations or
coefficient sums of such polynomials.

**Log-concavity of $g_M$ (prior art).** A polynomial $p$ with nonnegative coefficients is
*log-concave* on $\mathbb R^n_{>0}$ if $\log p$ is concave there, equivalently
$\nabla^2 \log p \preccurlyeq 0$. Since
$\nabla^2 \log p = \frac{p\,\nabla^2 p - (\nabla p)(\nabla p)^\intercal}{p^2}$ and
$(\nabla p)(\nabla p)^\intercal$ is rank $1$, by Cauchy interlacing log-concavity of a
homogeneous nonnegative-coefficient $p$ is equivalent to $\nabla^2 p$ having **at most one
positive eigenvalue** at every $x > 0$. Building on the Hodge–Riemann relations of
Adiprasito–Huh–Katz for the Chow ring of a matroid, it is known (Anari–Oveis Gharan–Vinzant
2018) that $g_M$ itself is log-concave on the positive orthant for *every* matroid. That
work used this top-level log-concavity to give a *deterministic* $2^{O(r)}$-approximation to
$|\mathcal B|$ via an entropy/convex-optimization argument.

**Euler's identity for homogeneous polynomials.** For $d$-homogeneous $p$,
$d\, p(x) = \sum_k x_k\, \partial_k p(x)$, and consequently
$(d-1)\nabla p = (\nabla^2 p)\,x$ and
$(d-2)\nabla^2 p = \sum_k x_k\, \nabla^2(\partial_k p)$. These let one pass between a
polynomial, its derivatives, and Hessian sums when working at the all-ones point $\mathbf 1$.

**High-dimensional expanders and high-order walks (prior art).** A *simplicial complex* $X$
on $[n]$ is a downward-closed family of subsets; $X(k)$ are its $k$-element faces; the *link*
$X_\tau = \{\sigma \setminus \tau : \sigma \in X,\ \sigma \supseteq \tau\}$ localizes $X$
around a face $\tau$; $X$ is *pure of dimension $d$* if every maximal face has size $d$. A
*balanced* weight function satisfies $w(\tau) = \sum_{\sigma \in X(k+1),\ \sigma \supset
\tau} w(\sigma)$. One can define *high-order random walks*: an *up* walk $P_k^{\wedge}$ on
$X(k)$ (go up to a $(k{+}1)$-face, come back down) and a *down* walk $P_{k+1}^{\vee}$ on
$X(k+1)$; these are two-step walks on the bipartite incidence graph between $X(k)$ and
$X(k+1)$, are reversible w.r.t. $w$, and share their nonzero spectrum. A line of work showed
that such walks mix well when the *links* are good spectral expanders. Kaufman–Mass (2017)
introduced the high-order walk and gave a first mixing bound; Dinur–Kaufman (2017) improved
it to a spectral gap $\ge \frac{1}{k+2} - O((k+1)\lambda)$ *provided every link's
$1$-skeleton walk is a two-sided $\lambda$-spectral expander* (all nontrivial eigenvalues
$\le \lambda$ in absolute value), which requires $\lambda \ll 1/k^2$. Kaufman–Oppenheim
(2018) sharpened this to a *one-sided* hypothesis: it suffices that only the *second-largest*
eigenvalue of each link walk is $\le \lambda$ (negative eigenvalues of the links are
unconstrained), yielding the same gap $\frac{1}{k+2} - (k+1)\lambda$. Oppenheim (2017)
established the reverse "descent of spectral gaps": connectivity of all link $1$-skeletons
together with a spectral condition on the codimension-two links forces the whole complex to
be a local spectral expander. Garland's method (1973) is the classical precursor relating
link Laplacians to global ones.

**Negative correlation and balanced matroids.** Feder–Mihail (1992) isolated the *balanced*
matroids — those for which $M$ and all minors satisfy pairwise negative correlation of the
uniform basis measure, $\Pr[i, j \in B] \le \Pr[i]\Pr[j]$ — and showed the bases-exchange
walk mixes rapidly for them.

## Baselines

**Feder–Mihail balanced-matroid sampling (1992).** For a *balanced* matroid, the uniform
measure on bases and on all its minors' bases is negatively correlated. By an induction on
$|E|$ exploiting negative association, the bases-exchange walk is shown to have good
conductance, hence to mix in polynomial time, hence to sample and approximately count bases.

**Deterministic $2^{O(r)}$-approximation via log-concavity (2018).** Using log-concavity of
$g_M$ on the positive orthant (from Hodge theory) and an entropy/convex-optimization
framework, one obtains a deterministic algorithm outputting $\beta$ with
$\max\{2^{-O(r)}\beta, \sqrt\beta\} \le |\mathcal B| \le \beta$ — and, against the
$2^{\Omega(r/\log n)}$ oracle lower bound for deterministic algorithms, this is essentially
the best a deterministic method can do.

**Two-sided high-dimensional-expander mixing (Kaufman–Mass 2017, Dinur–Kaufman 2017).** If a
weighted pure complex is a *two-sided* $\lambda$-local spectral expander — every link's
$1$-skeleton walk has all nontrivial eigenvalues within $\pm\lambda$ — the top high-order
walk has spectral gap $\ge \frac{1}{k+2} - O((k+1)\lambda)$, so it mixes once
$\lambda \ll 1/k^2$.

**Other counting routes.** The popping method handles bicircular matroids; a randomized
algorithm of Barvinok–Samorodnitsky gives roughly a $\log(n)^r$ approximation for general
matroids; real-stable techniques handle the special class of real-stable (a subclass of
balanced) matroids.

## Evaluation settings

The natural yardsticks are intrinsic, not dataset benchmarks. (i) *Mixing time*: number of
steps for the bases-exchange chain started at a basis $B$ to reach total-variation distance
$\varepsilon$ of uniform — desired bound polynomial in $n$, $r$, $\log(1/\varepsilon)$.
(ii) *Approximation guarantee*: a $(1\pm\varepsilon)$ multiplicative estimate of
$|\mathcal B|$ with success probability $\ge 1-\delta$, in time polynomial in
$n, r, 1/\varepsilon, \log(1/\delta)$ — the FPRAS criterion, obtained from the sampler via
the self-reducible sampling-to-counting reduction. (iii) *Structural*: the edge expansion
$h(G_M)$ of the bases-exchange graph (the Mihail–Vazirani target value is $1$). Test
matroids span graphic matroids (spanning trees, where exact counts via the matrix-tree
theorem give a ground truth), linear/representable matroids, and abstract oracle matroids
including the non-balanced acyclic-subset matroids. Related partition functions provide
further settings: the random-cluster / Tutte polynomial and $k$-determinantal point
processes.

## Code framework

Pre-method primitives that already exist: an independence oracle, a reversible-Markov-chain
runner, and the self-reducible sampling-to-counting reduction.

```python
import numpy as np

# --- already available: the matroid, given by an independence oracle ---
class Matroid:
    def __init__(self, n, independent_fn):
        self.n = n
        self.independent = independent_fn      # independent(S: frozenset) -> bool

    def rank(self, S):
        S = list(S); basis = set()
        for e in S:
            if self.independent(frozenset(basis | {e})):
                basis.add(e)
        return len(basis)

    def contract(self, S):
        S = frozenset(S)
        return Matroid(self.n, lambda T: self.independent(frozenset(T | S)))

# --- already available: the natural local chain on top sets (one step) ---
def bases_exchange_step(M, B, rng):
    """B is a current basis (frozenset of size r). One reversible local move."""
    B = set(B)
    i = rng.choice(list(B))                    # drop a uniform element
    B.remove(i)
    addable = [j for j in range(M.n)
               if j not in B and M.independent(frozenset(B | {j}))]
    j = rng.choice(addable)                    # add a uniform completing element
    B.add(j)
    return frozenset(B)

# --- already available: generic mixing-time / counting harness ---
def run_chain(M, start, steps, rng):
    B = start
    for _ in range(steps):
        B = bases_exchange_step(M, B, rng)
    return B

def fpras_via_self_reducibility(M, eps, delta, sampler):
    """Telescoping over restrictions/contractions; turns an almost-uniform
       basis sampler into a (1±eps) estimate of the number of bases."""
    raise NotImplementedError  # standard JVV-style reduction; not the contribution

# --- THE OBJECT TO BE DESIGNED ---
def certify_rapid_mixing(M):
    """Decide / prove that the local chain on bases mixes in poly(n, r, log(1/eps))."""
    pass
```
