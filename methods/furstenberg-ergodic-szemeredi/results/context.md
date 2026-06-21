# Context

## Research question

Does *density alone* force arithmetic progressions? Concretely: if
$A \subseteq \mathbb{Z}$ has positive upper density

$$\bar{d}(A) \;=\; \limsup_{N\to\infty} \frac{|A \cap \{-N,\dots,N\}|}{2N+1} \;>\; 0,$$

must $A$ contain, for every $k$, a non-degenerate $k$-term arithmetic progression
$a, a+n, a+2n, \dots, a+(k-1)n$ (with $n \neq 0$)? This is the Erdős–Turán conjecture
(1936). Endre Szemerédi proved it in 1975 by a long combinatorial argument
built on van der Waerden's theorem and what is now called the regularity lemma. The
question taken up here is whether the same statement can be reached by a different
route — one that exhibits a mechanism forcing the progressions.

## Background

**van der Waerden's theorem (1927).** Colour $\{1,\dots,N\}$ with $k$ colours; for $N$
large (depending on $k$ and target length $\ell$) some colour class contains an
$\ell$-term arithmetic progression. The standard proof (Lukomskaya's, as presented by
Khinchin and by Swan) is a "colour-focusing" induction: assuming the result for length
$\ell$, one manufactures inside a long enough segment an *$m$-fold* arithmetic
progression $f(i_1,\dots,i_m) = a + \sum_\nu i_\nu d_\nu$ of length $\ell$, all of whose
"corner" values are forced into the same colour class; with $m = k$ colours, the
pigeonhole on the $k+1$ diagonal endpoints $f(0,\dots,0,\ell,\dots,\ell)$ produces two
in the same class, and the segment between them is the length-$(\ell+1)$ progression.
This is the *partition* (colouring) regularity statement.

**The Erdős–Turán strengthening.** A single colour class of positive density should
already force progressions: density, not finiteness of the colouring, is conjectured to
be the true cause. Roth (1953) proved the $k=3$ case by Fourier/analytic methods: a set
of positive density either correlates with a single frequency (and then a counting
argument finds a 3-AP) or is sufficiently "uniform" that the 3-AP count is close to the
random value. Szemerédi proved $k=4$ (1969) and the general case (1975), combinatorially.

**Properties of asymptotic density.** The limsup density is monotone, subadditive, and
invariant under the shift $S\colon x \mapsto x+1$. It is not finitely additive and is
defined by a limsup rather than a genuine limit. It lives on $\mathbb{Z}$, which is not
compact.

**Ergodic theory of measure-preserving systems.** A measure-preserving system (m.p.s.)
$(X,\mathcal{B},\mu,T)$ is a probability space with an invertible measure-preserving map
$T$. The classical results that exist and will be the tools:

- **Poincaré recurrence.** If $\mu(B) > 0$ then $\mu(B \cap T^{-n}B) > 0$ for some
  $n > 0$. Reason: the sets $B, T^{-1}B, T^{-2}B, \dots$ all have measure $\mu(B)$ in a
  space of total mass $1$, so they cannot be pairwise disjoint. The mean ergodic theorem
  strengthens this to a positive Cesàro average of return measures.
- **Mean ergodic theorem (von Neumann).** $\frac{1}{N}\sum_{n=0}^{N-1} T^n f \to E(f \mid
  \mathcal{I})$ in $L^2$, the conditional expectation onto the $\sigma$-algebra
  $\mathcal{I}$ of $T$-invariant sets; the system is *ergodic* iff $\mathcal{I}$ is
  trivial, i.e. the limit is the constant $\int f\,d\mu$.
- **Spectral theorem.** $\int f \cdot T^n \bar{f}\,d\mu = \int_{-\pi}^{\pi} e^{in\theta}
  \,d\rho(\theta)$ for a positive spectral measure $\rho$ on the circle; atoms of $\rho$
  are eigenvalues of $T$, eigenfunctions $\phi(Tx) = e^{i\theta}\phi(x)$.
- **Ergodic decomposition.** Any invariant measure is an integral of ergodic ones, so a
  statement about a positive-measure set reduces to the ergodic case.
- **Two extreme classes.**
  *Compact / Kronecker systems*: $X$ a compact abelian group, $T$ rotation by a
  generator, $\mu$ Haar measure. Every orbit is almost periodic; $L^2$ is spanned by
  eigenfunctions (characters). Example: an irrational circle rotation.
  *Weak-mixing systems*: $T \times T$ acting on $X \times X$ is ergodic; equivalently
  there are no non-constant eigenfunctions; equivalently $E$ and $T^n E$ decorrelate on
  average. Example: a Bernoulli shift.

**Topological dynamics as the coarse template.** vdW translates into a topological
statement: a minimal homeomorphism of a compact space has, for every non-empty open
$U$, an $n$ with $U \cap T^{-n}U \cap \dots \cap T^{-kn}U \neq \emptyset$ (multiple
recurrence in open covers).

## Baselines

- **Szemerédi's combinatorial proof (1975).** An iterated regularity /
  density-increment scheme over van der Waerden, partitioning the structure of the set
  until progressions are forced. The mechanism is the regularity lemma plus a long case
  analysis, and gives towers-of-twos bounds.

- **Roth's analytic proof of $k=3$ (1953).** A set of positive density has a Fourier
  transform that is either concentrated (a large non-trivial Fourier coefficient $\Rightarrow$
  a density increment on a sub-progression, iterate) or spread out (then the 3-AP count is
  $\approx \delta^3 N^2$, positive). It is realized in ordinary frequency space and
  established for $k=3$.

- **van der Waerden's theorem (1927) as a baseline target.** It is the colouring shadow
  of the density statement and the thing Szemerédi's proof leans on. It gives
  monochromatic progressions in *some* colour class; a positive-density class can be a
  single colour, so the density statement implies vdW.

- **Poincaré recurrence (1899) as the $k=2$ prototype.** It is the ergodic statement for
  two-term "progressions" $\{x, T^n x\}$: a *single* return of a positive-measure set
  to itself.

## Evaluation settings

This is a pure-mathematics result; the "evaluation" is logical, not empirical. The
yardsticks that exist:

- **The statement to be matched.** Erdős–Turán / Szemerédi: positive upper density
  $\Rightarrow$ arbitrarily long arithmetic progressions, for all $k$. A more general
  density hypothesis (positive upper Banach density) can also be considered; the centered
  upper-density statement here is the target.
- **Special cases for reference.** $k=2$ relates to Poincaré recurrence; $k=3$ to
  Roth's theorem; the colouring shadow to van der Waerden.
- **Worked test systems.** Irrational circle/torus rotations
  (compact/Kronecker), Bernoulli shifts (weak mixing), and skew products / group
  extensions of rotations — the standard dynamical systems on which any candidate
  argument can be checked.


