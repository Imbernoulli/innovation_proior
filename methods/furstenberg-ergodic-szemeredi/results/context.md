# Context

## Research question

Can one prove that *density alone* forces arithmetic progressions? Concretely: if
$A \subseteq \mathbb{Z}$ has positive upper density

$$\bar{d}(A) \;=\; \limsup_{N\to\infty} \frac{|A \cap \{-N,\dots,N\}|}{2N+1} \;>\; 0,$$

must $A$ contain, for every $k$, a non-degenerate $k$-term arithmetic progression
$a, a+n, a+2n, \dots, a+(k-1)n$ (with $n \neq 0$)? This is the Erdős–Turán conjecture
(1936). Endre Szemerédi proved it in 1975 by a long, intricate combinatorial argument
built on van der Waerden's theorem and what is now called the regularity lemma. The
question that matters here is not merely *whether* it is true — that is settled — but
whether there is a *structurally transparent* proof: one that exhibits the reason the
progressions are forced, instead of only certifying that they are present.

The pain point is that the existing combinatorial proof, while complete, is opaque:
it is a tour de force of nested case analysis whose mechanism is hard to see and hard
to transplant. A solution worth having should isolate the small number of "regimes"
that actually control the count of progressions, and prove the result by reducing every
situation to those regimes.

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

**Asymptotic density behaves like the shadow of a measure.** The limsup density itself is
not a finitely additive measure; it is monotone, subadditive, and invariant under the
shift $S\colon x \mapsto x+1$. The usable object appears after choosing intervals on
which the density of the set tends to a positive limit and refining the averaging scheme
so the relevant shift algebra has genuine limits. Those limits form a positive
shift-invariant linear functional, a Banach mean. This is the load-bearing analogy: a
shift-invariant mean on integers is the discrete shadow of a shift-invariant probability
measure, and the shift is the discrete shadow of a measure-preserving transformation.
What raw limsup density lacks — countable additivity, a single honest limit, an $L^2$
structure — is exactly what a probability space supplies.

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
  average. Example: a Bernoulli shift. These two classes are the "purely structured" and
  "purely random" extremes.

**Topological dynamics as the coarse template.** vdW translates into a topological
statement: a minimal homeomorphism of a compact space has, for every non-empty open
$U$, an $n$ with $U \cap T^{-n}U \cap \dots \cap T^{-kn}U \neq \emptyset$ (multiple
recurrence in open covers). This shows the *shape* of the target on the measure side.

## Baselines

- **Szemerédi's combinatorial proof (1975).** Core idea: an iterated regularity /
  density-increment scheme over van der Waerden, partitioning the structure of the set
  until progressions are forced. The actual mechanism (the regularity lemma plus a long
  case analysis) is correct but extremely intricate, gives towers-of-twos bounds, and
  does not make the forcing mechanism easy to isolate. **Gap:** no transparent "reason,"
  no reusable structural dichotomy.

- **Roth's analytic proof of $k=3$ (1953).** A set of positive density has a Fourier
  transform that is either concentrated (a large non-trivial Fourier coefficient ⇒ a
  density increment on a sub-progression, iterate) or spread out (then the 3-AP count is
  $\approx \delta^3 N^2$, positive). This is already a structure-vs-randomness split,
  but realized in ordinary frequency space and only known to close for $k=3$. **Gap:** no
  comparable linear-Fourier dichotomy is available for all longer progressions.

- **van der Waerden's theorem (1927) as a baseline target.** It is the colouring shadow
  of the density statement and the thing Szemerédi's proof leans on. **Gap:** it gives
  monochromatic progressions in *some* colour class, not progressions forced by density;
  it is genuinely weaker (a positive-density class can be a single colour, so the density
  statement implies vdW but not conversely).

- **Poincaré recurrence (1899) as the $k=2$ prototype.** It is exactly the desired
  ergodic statement for two-term "progressions" $\{x, T^n x\}$ and is essentially free
  from measure-preservation. **Gap:** it is a *single* return; the difficulty is the
  *simultaneous* return of $x, T^n x, T^{2n}x, \dots, T^{(k-1)n}x$ to the same set along
  a common gap $n$, which Poincaré's pigeonhole does not reach.

## Evaluation settings

This is a pure-mathematics result; the "evaluation" is logical, not empirical. The
yardsticks that exist:

- **The statement to be matched.** Erdős–Turán / Szemerédi: positive upper density
  $\Rightarrow$ arbitrarily long arithmetic progressions. The same correspondence
  framework can be phrased with positive upper Banach density, a more general density
  hypothesis, but the centered upper-density statement here is the target. A proof must
  recover this for all $k$, with no density-degree restriction.
- **Special cases that must drop out.** $k=2$ must reduce to Poincaré recurrence; $k=3$
  must recover Roth's theorem; the colouring shadow must recover van der Waerden.
- **Worked test systems for the ergodic statement.** Irrational circle/torus rotations
  (compact/Kronecker), Bernoulli shifts (weak mixing), and skew products / group
  extensions of rotations (distal, the first genuinely "neither" systems) — these are the
  standard systems on which any claimed multiple-recurrence argument is checked.
- **Logical economy as the metric.** Beyond correctness, the proof is judged by how few
  regimes it needs and how cleanly the general case reduces to them.

## Code framework

This is a theorem, not an algorithm, so the "scaffold" is the logical skeleton the
argument will fill, written with integer-density and measure-preserving-system
primitives. The slots are the steps a proof would have to supply.

```python
# Available primitives.

def upper_density(A):
    """limsup_N |A ∩ [-N,N]| / (2N+1).  Shift-invariant, but not a measure."""
    ...

def shift(B):
    """T: B ↦ B-1 on subsets of Z; equivalently (1_B ∘ T)(m)=1_B(m+1)."""
    return {b - 1 for b in B}

class MeasurePreservingSystem:
    """(X, B, mu, T): probability space + invertible measure-preserving T.
    Available tools: mean/pointwise ergodic theorem, spectral theorem,
    conditional expectation E(.|.), ergodic decomposition."""
    def mean_ergodic_limit(self, f): ...        # -> E(f | invariant sigma-algebra)
    def spectral_measure(self, f): ...          # rho with ∫ f·T^n f̄ = ∫ e^{inθ} dρ

def poincare_recurrence(system, B):
    """k=2: mu(B ∩ T^{-n}B) > 0 for some n."""
    ...

# ---- Open proof slots. ----

def correspondence_principle(A):
    """From a set A of positive upper density, BUILD a measure-preserving system
    (X, B, mu, T) and a set E with mu(E) > 0, so that arithmetic progressions in A
    correspond to simultaneous returns x, T^n x, ..., T^{(k-1)n}x ∈ E.
    # TODO: which compact X? which T? how to manufacture an invariant mu from
    #       a limsup density that is not itself countably additive?"""
    raise NotImplementedError

def multiple_recurrence(system, B, k):
    """The ergodic statement to prove: ∃ n>0 with
    mu(B ∩ T^{-n}B ∩ ... ∩ T^{-(k-1)n}B) > 0   (ideally Cesàro-positive).
    # TODO: what is the right way to handle a GENERAL system, given that only
    #       the two extreme regimes are directly tractable?"""
    raise NotImplementedError

def szemeredi_from_recurrence(A, k):
    """Combine the two slots above to force a k-term AP in A."""
    system, E = correspondence_principle(A)
    assert multiple_recurrence(system, E, k)     # ⇒ AP in A
    ...
```
