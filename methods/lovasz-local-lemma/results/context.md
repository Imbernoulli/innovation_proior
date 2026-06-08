# Context

## Research question

I want to prove that a combinatorial object with a desired global property *exists*, in a regime where
the only general tool I have — the probabilistic method — runs out of steam. The probabilistic method
proves existence non-constructively: I put a probability distribution on the candidate objects, define a
family of "bad" events `A_1, …, A_n` (each one saying "the object fails in this particular spot"), and
argue that

    P(none of the A_i occurs) > 0,

because a random object avoiding every bad event is exactly a good object. Two clean situations make this
trivial. If the bad events are *mutually independent* and each has probability `< 1`, then
`P(Ā_1 … Ā_n) = ∏_i (1 − P(A_i)) > 0`, no matter how large the individual probabilities or their sum. If
instead the *union bound* applies — `Σ_i P(A_i) < 1` — then `P(⋃ A_i) ≤ Σ_i P(A_i) < 1`, so again the good
object exists.

The hard regime, and the one I care about, sits between these. I have *many* bad events; each is *individually
unlikely*; but `Σ_i P(A_i)` is far larger than `1`, so the union bound is hopeless; and the events are *not*
independent, so the product formula does not apply either. What rescues this situation is that the dependence
is *local*: each bad event is statistically entangled with only a handful of the others and is independent of
all the rest. The precise question is: **does a small amount of local dependence still leave positive
probability that no bad event occurs — and exactly how local must it be?** A theorem that answered this would
convert a hopeless *global* counting condition (a bound on the *total* number of bad events) into a mild
*local* condition (a bound on how many other bad events each one can interact with), and existence proofs that
were blocked would go through.

## Background

**The probabilistic method and where its two easy modes live.** The method, developed largely by Erdős
through the 1940s–60s, proves existence by averaging. The two regimes above are its workhorses. The
independence regime is the strongest possible — `∏(1 − p_i)` stays positive even when `Σ p_i` is enormous —
but genuine independence among "the object fails here" events is rare, because two failures usually share some
of the underlying random choices. The union-bound regime needs no independence at all, but it is wasteful: it
charges the full `P(A_i)` for every bad event with no credit for the fact that most pairs of bad events barely
interact, so it requires `Σ_i P(A_i) < 1`, a *global* budget that breaks the moment the number of bad events
grows.

**Limited dependence as a graph.** Between the two extremes is the realistic case: each `A_i` is determined by
some subset of the underlying independent random choices, and two bad events are entangled only when their
subsets overlap. This is captured by a *dependency graph*: vertices are the events `A_1, …, A_n`, and `A_i` is
joined to the events it can depend on, so that `A_i` is independent of the *set* of all events it is **not**
joined to. The load-bearing word is *set*: I need `A_i` to be independent of the entire collection of its
non-neighbours taken together (independent of every Boolean combination of them), not merely independent of
each one separately. Pairwise independence is genuinely weaker — three events can be pairwise independent yet
not mutually independent (take `x_1, x_2, x_3` uniform and independent in `Z/2Z` and let `A_i` be the event
`x_{i+1} + x_{i+2} = 0`; any two are independent, all three are not), so the empty graph is *not* a valid
dependency graph for them. The dependency graph is therefore built from genuine mutual independence, not from
pairwise correlation, and it need not be unique (the complete graph is always valid; the art is to make it
sparse).

**Conditional probability is the only fine-grained tool available.** When events are dependent, the one exact
identity that always holds is the chain rule:
`P(Ā_1 … Ā_n) = P(Ā_1) · P(Ā_2 | Ā_1) · ⋯ · P(Ā_n | Ā_1 … Ā_{n−1})`. The whole product is positive iff every
factor is, and each factor is `1 − P(A_k | the earlier events did not occur)`. So the real quantity to control
is not the *unconditional* `P(A_k)` that the union bound uses, but the *conditional* probability of a bad event
given that some of the others were avoided. That is the object the question must be reduced to.

**The motivating concrete failure: hypergraph 2-colouring (property B).** A hypergraph is a family of sets
("edges") on a vertex set; it is `2`-colourable (has *property B*) if the vertices can be coloured with two
colours so that no edge is monochromatic. For a `k`-uniform hypergraph (every edge has `k` vertices), colour
each vertex red/blue independently with probability `1/2`; the bad event `A_f` = "edge `f` is monochromatic"
has probability `2 · 2^{-k} = 2^{-(k-1)}`. The union bound proves `2`-colourability whenever the *total* number
of edges is `< 2^{k-1}` — a global count. It is a known, exasperating fact that this is essentially all the
union bound gives, even though a hypergraph with millions of edges that pairwise barely intersect is "obviously"
`2`-colourable: each edge only conflicts with the few edges it actually meets. The diagnostic is sharp — the
union-bound threshold ignores the geometry entirely, charging for edges that are disjoint and therefore
independent.

**A neighbouring structural fact.** Lovász, and independently Woodall, had shown that every `3`-chromatic
`r`-uniform hypergraph must contain a vertex of high degree (degree `> r`); more generally, a hypergraph in
which every subfamily `H'` satisfies `|⋃_{E∈H'} E| > |H'|` is `2`-colourable. These are *structural* statements
— they tie non-`2`-colourability to the existence of locally dense spots — and they suggest that the right
condition for colourability is *local* (a degree / intersection bound) rather than *global* (a total count).
That is exactly the shape of condition a probabilistic existence theorem in the middle regime would want to
produce.

## Baselines

- **Independence regime of the probabilistic method.** If `A_1, …, A_n` are mutually independent with
  `P(A_i) < 1`, then `P(Ā_1 … Ā_n) = ∏(1 − P(A_i)) > 0`. *Core idea:* multiply complementary probabilities.
  *Gap:* genuine mutual independence of "failure-here" events almost never holds, because distinct failures
  share underlying random choices; the moment two bad events overlap, the product formula is unavailable.

- **Union-bound regime.** If `Σ_i P(A_i) < 1`, then `P(⋃ A_i) ≤ Σ_i P(A_i) < 1`, so `P(Ā_1 … Ā_n) > 0`.
  *Core idea:* subadditivity of probability; requires no independence. *Gap:* it is a *global* budget. It gives
  no credit for locality — a disjoint (hence independent) bad event still costs its full `P(A_i)` — so it fails
  as soon as the number of bad events makes `Σ P(A_i) ≥ 1`, even when every event interacts with only a few
  others. In the property-B application it caps the *total* edge count at `2^{k-1}`.

- **Structural colourability conditions (Lovász; Woodall).** *Core idea:* non-`2`-colourability forces a local
  density obstruction — a high-degree vertex, or a subfamily violating `|⋃ E| > |H'|`. *Gap:* these are exact
  structural theorems for specific colourings, not a general-purpose existence tool; they hint that the right
  hypothesis is local but do not provide a probabilistic engine that turns "each bad event touches few others"
  into "a good object exists" for an arbitrary system of bad events.

## Evaluation settings

The natural yardsticks are existence statements where many local constraints must be simultaneously satisfied
and the global count is too large for the union bound:

- **Hypergraph `2`-colouring / property B.** `k`-uniform hypergraphs; the quantity of interest is how many
  other edges a single edge may intersect (a *local* degree) while `2`-colourability is still forced — to be
  contrasted with the union bound's *global* edge-count threshold `2^{k-1}`. Also `k`-uniform `k`-regular
  hypergraphs as a clean special case.
- **`k`-SAT (Boolean satisfiability).** A `k`-CNF formula: a conjunction of clauses, each a disjunction of `k`
  literals on distinct variables. Under a uniform random assignment each clause is violated with probability
  exactly `2^{-k}`. The yardstick is how many other clauses a clause may share a variable with (its local
  dependency degree) while satisfiability is still guaranteed, independent of the total number of clauses.
- **Lattice / colouring translates and Ramsey-type lower bounds**, where one must simultaneously avoid a
  monochromatic or otherwise "bad" configuration at every location, with each location's bad event entangled
  with only nearby ones.

The metric throughout is qualitative existence (does a good object provably exist?) and the *form* of the
sufficient condition (local degree vs. global count), not any numerical performance score.

## Code framework

A minimal probabilistic-existence harness has only to assemble the independent random choices, define the bad
events, record which events may depend on which others, and certify that the probability of avoiding all bad
events is positive. The existing tools fill the full-independence and union-bound cases, while the middle
regime leaves one local certification slot open.

```python
def avoid_all_bad_events_exists(setup):
    """Decide/certify that a random object avoids every bad event with positive probability.

    `setup` provides:
      - the independent random choices and their distribution,
      - the bad events A_1, ..., A_n (each a predicate on an outcome),
      - for each A_i, its probability p_i and the set of other events it can depend on
        (its neighbours in a dependency graph; A_i is independent of all non-neighbours).
    Returns True if a good object (one avoiding all A_i) is guaranteed to exist.
    """
    # Mode 1 — full independence: if all A_i are mutually independent and p_i < 1,
    #          then prod(1 - p_i) > 0.  (Rarely applicable: failures share random choices.)
    # Mode 2 — union bound: if sum_i p_i < 1, then P(union A_i) < 1, so a good object exists.
    #          (Global budget; dies once there are many bad events.)
    #
    # Middle regime: many bad events, each unlikely, sum p_i >> 1, NOT independent but only
    # locally dependent (each A_i entangled with few others).  Neither mode applies.
    # TODO: fill the local condition that certifies P(avoid all) > 0.
    raise NotImplementedError


def k_uniform_hypergraph_2_colourable(H):
    """Certify property B for a k-uniform hypergraph from a LOCAL intersection bound."""
    # Random object: colour each vertex red/blue independently with prob 1/2.
    # Bad event A_f = "edge f is monochromatic"; p = 2 ** (-(k - 1)).
    # A_f independent of all A_g with f ∩ g = ∅; local dependency degree = #edges meeting f.
    # TODO: invoke avoid_all_bad_events_exists with the dependency structure above.
    raise NotImplementedError
```
