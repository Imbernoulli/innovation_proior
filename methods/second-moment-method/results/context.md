# Context: existence proofs in random structures, and the limits of the first moment

## Research question

I want to prove that a random structure *almost surely contains* a given configuration — a clique
of a fixed size in a random graph, a copy of a fixed subgraph, a number with about the typical number
of prime factors. The cleanest existing tool is the probabilistic/first-moment method: define a
counting random variable X (the number of configurations of the desired type), compute its
expectation E[X], and try to read off existence. This works in *one* direction. If E[X] is small —
tending to 0 with the size parameter — then X must usually be 0, so the configuration almost surely
does *not* appear. The hard and more interesting direction is the opposite: I want to conclude that
when the expected count is *large*, the configuration almost surely *does* appear.

The precise goal: given a family of nonnegative integer-valued counting variables X = X(n) indexed
by a growing parameter n, find conditions under which **Pr[X > 0] → 1**, and ideally the stronger
**X ∼ E[X]** (the count is concentrated at its mean). The pain point is that a large — even an
exploding — expectation does not deliver this by itself. A method that closes the gap would let one
pin down *thresholds*: the exact rate of the underlying parameter at which a property switches from
"almost never" to "almost always."

## Background

**The probabilistic method, first-moment form.** The foundational observation is trivial: if an
event A has Pr(A) > 0 then A is nonempty, so an object with the property exists. The averaging
refinement: if E[X] = x then some outcome has X ≥ x, and some outcome has X ≤ x. This already proves
strong existence results. For Max-Cut, a uniformly random vertex 2-coloring cuts each edge with
probability 1/2, so the expected cut size is |E|/2, hence a cut of size ≥ |E|/2 exists. For Max-3SAT,
a random assignment satisfies each 3-clause with probability 7/8, so an assignment satisfying ≥ 7m/8
clauses exists. These are pure first-moment existence arguments and they are tight as far as they go.

**Where the first moment is one-sided.** Take X ≥ 0 integer-valued, counting configurations.
Markov's inequality, Pr(X ≥ α) ≤ E[X]/α, at α = 1 gives Pr(X ≥ 1) ≤ E[X]. So **E[X] → 0 ⇒ X = 0
almost always** — a complete and clean conclusion in the "disappearance" direction. The reverse is
*false*: E[X] → ∞ does **not** imply X > 0 almost always. The standard cautionary example: let X be
a quantity that is enormous on a vanishingly rare event and 0 the rest of the time (Alon–Spencer use
"the number of deaths from nuclear war in the next year": E[X] can be argued to be very large, yet
one hopes Pr(X ≠ 0) ≈ 0). A large or even diverging mean, alone, certifies nothing about typical
existence; the expectation cannot see how the mass of X is distributed around it.

**Chebyshev's inequality.** For X with mean µ and variance σ² = Var[X] = E[(X−µ)²], Chebyshev says
Pr[|X − µ| ≥ λσ] ≤ 1/λ² for any λ > 0. The proof is Markov applied to (X−µ)²:
Pr(|X−µ| ≥ α) = Pr((X−µ)² ≥ α²) ≤ E[(X−µ)²]/α² = Var[X]/α². This is the elementary fact that ties a
small variance to concentration around the mean.

**Variance of a sum of indicators.** When X = X₁ + ⋯ + X_m is a sum, Var[X] = Σᵢ Var[Xᵢ] +
Σ_{i≠j} Cov[Xᵢ, Xⱼ], with Cov[Y,Z] = E[YZ] − E[Y]E[Z], and Cov = 0 for independent variables. For
**indicators** Xᵢ = 1_{Aᵢ} with pᵢ = Pr[Aᵢ]: Var[Xᵢ] = pᵢ(1−pᵢ) ≤ pᵢ = E[Xᵢ], so the diagonal sums
to at most E[X]; only the off-diagonal covariances between *dependent* events carry the rest of the
variance.

**The Paley–Zygmund inequality (1932).** A reverse-Chebyshev anti-concentration bound: for Z ≥ 0 and
t ∈ [0,1], Pr[Z > tE[Z]] ≥ (1−t)² E[Z]²/E[Z²]. It lower-bounds the chance that a nonnegative variable
is a constant fraction of its mean, using only its first two moments. Originally proved by Paley and
Zygmund for analytic functions on the unit circle; the probabilistic phrasing is a one-line
Cauchy–Schwarz argument.

**Random graphs and thresholds (Erdős–Rényi 1960).** The random graph G(n,p) puts each of the C(n,2)
possible edges in independently with probability p. A graph *property* is a family closed under
isomorphism. Erdős and Rényi observed that for many fundamental properties A there is a **threshold
function** r(n): when p ≪ r(n), G(n,p) almost surely lacks A; when p ≫ r(n), it almost surely has A.
They introduced this "evolution" picture and computed the first such thresholds (connectivity,
appearance of small subgraphs, etc.). The motivating phenomenon is that many natural properties do
not fade in gradually as p grows; they switch from unlikely to likely around a definite scale.
Establishing a threshold has exactly two halves: an "almost never below" half (first moment) and an
"almost always above" half (which needs more than the mean).

**The number-theory precedent (Turán 1934).** Let ν(n) be the number of distinct prime divisors of n.
Hardy and Ramanujan (1917) showed almost all n ≤ N have ≈ ln ln N prime factors by an intricate
argument. Turán re-proved it by writing ν as a sum of near-independent indicators X_p = 1_{p|x} for a
random x and arguing about the resulting count — a strikingly short proof compared with the original.

## Baselines

- **First-moment / union-bound existence (probabilistic method).** Compute E[X]; conclude an object
  with X ≥ E[X] exists, or (for a count) that E[X] → 0 ⇒ X = 0 a.a.s. *Gap*: gives nothing in the
  appearance direction — E[X] → ∞ does not imply Pr[X > 0] → 1. It is fundamentally one-sided for
  nonnegative counts.

- **Markov's inequality alone.** Pr(X ≥ α) ≤ E[X]/α. Pins the disappearance half (α = 1) but says
  nothing about a lower tail; cannot force X > 0 from a large mean. *Gap*: no two-sided control.

- **Hardy–Ramanujan (1917) for ν(n).** Establishes the "normal order ln ln n" of the number of prime
  factors, but by a long elementary-number-theory argument specific to ν. *Gap*: not a reusable
  method; gives the result without a transferable mechanism.

- **Erdős–Rényi (1960) threshold computations.** Establish thresholds for several graph properties.
  For the appearance of a fixed subgraph they identify the threshold and prove it. *Gap*: at this
  stage the "almost always above threshold" direction is handled case by case; there is no single,
  transferable criterion that mechanically yields Pr[X > 0] → 1 and X ∼ E[X] for an arbitrary
  counting variable once the mean is large.

## Evaluation settings

The natural yardsticks are existence/threshold statements one can prove and check:

- **G(n,p) subgraph appearance.** For a fixed graph H with v vertices and e edges, X = number of
  copies of H in G(n,p). The object is the threshold p(n) at which Pr[H ⊆ G(n,p)] → 1, and the
  asymptotic count of copies above threshold. Special case H = K₄ (the 4-clique).

- **Clique number of G(n, 1/2).** With X = number of k-cliques and f(k) = C(n,k) 2^{−C(k,2)} its
  expectation, the object is to locate the value k ∼ 2 log₂ n where f crosses 1 and to determine how
  sharply ω(G(n,1/2)) concentrates.

- **Number of prime factors.** For x uniform in {1,…,n}, the object is the typical size of ν(x) and
  the deviation Pr[|ν(x) − ln ln n| > λ√(ln ln n)].

Metrics throughout: whether Pr[X > 0] → 1 (existence almost surely), whether X ∼ E[X]
(concentration), and the deviation probabilities Chebyshev/Paley–Zygmund deliver.

## Code framework

A pure existence/threshold result; the only natural code is a small numerical sanity check that a
counting variable's mean and the existence/disappearance flip line up across a candidate threshold.
The pre-existing primitives are an RNG and basic graph bookkeeping. The empty slots are the count and
its mean; whatever the proof needs beyond that is supplied separately.

```python
import itertools, random

def sample_gnp(n, p):
    """Sample G(n,p): each pair an edge independently w.p. p."""
    edges = set()
    for u, v in itertools.combinations(range(n), 2):
        if random.random() < p:
            edges.add((u, v))
    return edges

def count_configurations(n, edges, k):
    """Count target configurations (e.g. k-cliques) in a sampled graph."""
    # TODO: the count X whose existence we want to certify
    pass

def expected_count(n, p, k):
    """The first moment E[X] of the counting variable."""
    # TODO: E[X] for the configuration (linearity of expectation)
    pass

def fraction_with_configuration(n, p, k, trials=300):
    """Estimate the observed frequency of X>0 for a candidate threshold."""
    hit = sum(count_configurations(n, sample_gnp(n, p), k) > 0 for _ in range(trials))
    return hit / trials
```
