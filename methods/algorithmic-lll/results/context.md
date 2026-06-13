## Research question

Many combinatorial objects are specified negatively: an object is good exactly when it avoids a long list of "bad" patterns. A satisfying truth assignment to a CNF formula avoids every "this clause is false" event; a proper 2-coloring of a hypergraph avoids every "this edge is monochromatic" event; a Latin transversal avoids every "these two cells collide" event. In each case there is a probability space (flip every variable, color every vertex at random) in which each bad event has small probability and each bad event depends only on a few of the underlying random choices.

The probabilistic existence question — *does a good object exist?* — has a clean sufficient condition (below). But existence is not what we ultimately want. We want to **construct** the good object, efficiently. The pain is sharp: the existence proof can guarantee that the probability of a good random point is positive yet exponentially small, so drawing random points and checking them is hopeless. The goal is an algorithm that, whenever the existence condition holds, **finds** a point avoiding all bad events in time polynomial in the number of events and variables — and ideally under the *same* condition that guarantees existence, with no loss in the allowed amount of dependence.

## Background

Fix a finite collection of mutually independent random variables P (a fair coin for each boolean variable, say). Consider a finite family of events A, each event A determined by the values of some subset vbl(A) ⊆ P. Build the dependency graph G on vertex set A with an edge between A and B whenever vbl(A) ∩ vbl(B) ≠ ∅, and write Γ(A) for the neighborhood of A. Because A is determined by vbl(A) and any event whose variables are disjoint from vbl(A) is independent of A, this Γ(A) is a valid "dependency neighborhood."

The Lovász Local Lemma (Erdős–Lovász 1975) is the existence engine. In its general form: if there is an assignment of reals x : A → (0,1) with

    Pr[A] ≤ x(A) · ∏_{B ∈ Γ(A)} (1 − x(B))    for every A,

then Pr[no event in A occurs] ≥ ∏_A (1 − x(A)) > 0, so a good point exists. Setting x ≡ 1/(d+1) yields the symmetric corollary everyone uses: if every event has Pr[A] ≤ p, depends on at most d others, and

    e · p · (d + 1) ≤ 1,

then a good point exists. In the SAT incarnation, a k-CNF formula in which every clause shares variables with at most 2^{k−2} other clauses is satisfiable: each clause is violated with probability 2^{−k} under a uniform random assignment, so p = 2^{−k}, d ≈ 2^{k−2}, and e·p·(d+1) ≤ 1.

The proof is non-constructive in a strong way. It shows, by induction on subsets S of the events, that the conditional probability Pr[A_i | ⋀_{j∈S} Ā_j] stays bounded by x_i; splitting S into the neighbors S₁ and non-neighbors S₂ of A_i, the numerator uses independence of A_i from S₂ to give Pr[A_i | ⋀ S₂] = Pr[A_i] ≤ x_i ∏(1−x_j), and the denominator is bounded below by ∏(1−x_j) via the induction hypothesis. Telescoping over all events gives Pr[⋀ Ā_i] ≥ ∏(1−x_i). The argument certifies that the good set is nonempty, but it conditions on events of tiny probability and produces no procedure for locating a point in the good set; and the good set itself can have measure ∏(1−x_i), which is exponentially small.

The diagnostic fact that frames the whole problem: under the LLL condition a good point provably exists, yet the obvious way to produce one — sample and check, repeating until success — takes an expected ∏(1−x_i)^{−1} draws, which is exponential. The conditional-probability slack the existence proof exploits is precisely what makes naive search useless.

## Baselines

**Sample-and-reject.** Draw a uniform random point; if some event is violated, throw it all away and draw again. Correct, trivially, but the expected number of draws is the reciprocal of the (exponentially small) success probability. This is the baseline any real algorithm must beat by an exponential factor.

**Beck (1991), the first constructive version.** Beck broke the long-standing barrier by giving the first polynomial-time algorithm, formulated for hypergraph 2-coloring: if every edge of a k-uniform hypergraph shares vertices with at most roughly 2^{k/48} other edges, his algorithm 2-colors the vertices with no monochromatic edge in polynomial time. The strategy is two-phase: color/assign most of the structure randomly, identify the "dangerous" part where bad events cluster, freeze it, and brute-force the small frozen components (whose total size is kept logarithmic). The gap to existence is enormous — the existential lemma allows roughly 2^{k}/e neighbors, Beck's algorithm only 2^{k/48}.

**Alon (1991).** A simpler, randomized variant of Beck's two-phase strategy, improving the threshold to essentially 2^{k/8}. Same shape (freeze a dangerous core, brute-force it); same kind of exponential gap, smaller constant in the exponent.

**Srinivasan (2008); earlier Moser (2008).** Further refinements of the freeze-and-brute-force template push the threshold to essentially 2^{k/4} and then 2^{k/2}. Each narrows the gap but none reaches the existential threshold; the constant in the exponent is an artifact of throwing away most of the available slack when freezing.

The common limitation across Beck, Alon, Srinivasan, and the 2008 line: they are genuinely constructive but lose an exponential factor in the allowed dependence, because the two-phase "freeze the hard core, brute-force it" design only works when the hard core stays tiny, which forces the neighborhood bound far below 2^{k}.

## Evaluation settings

The natural yardsticks are standard LLL applications where the task is to construct the object whose existence is guaranteed:

- **k-SAT with bounded clause overlap.** Instances of k-CNF where each clause shares variables with at most ~2^{k}/e other clauses; the metric is whether a satisfying assignment is found and how much work it takes. This is the cleanest computational testbed because sampling (flip each variable) and checking (scan each clause) are both trivial.
- **k-uniform hypergraph 2-coloring** with bounded edge degree; find a non-monochromatic coloring.
- The figures of merit are the allowed dependence (how close to the existential 2^{k}/e threshold the algorithm reaches) and the running time.

## Code framework

The available pieces are a random source, a representation of CNF clauses, the trivial "is this clause violated" check, and an outer loop that searches for a good assignment. The body that decides how to move from a violated state toward a good one is still open.

```python
import random

def is_clause_violated(clause, assignment):
    # clause: list of (var, sign) literals; satisfied iff some literal is true
    for var, sign in clause:
        if assignment[var] == sign:
            return False
    return True

def violated_clauses(clauses, assignment):
    return [i for i, c in enumerate(clauses) if is_clause_violated(c, assignment)]

def random_assignment(n_vars, rng):
    # one fair coin per variable: a uniform point of the probability space
    return [rng.random() < 0.5 for _ in range(n_vars)]

def search_good_assignment(clauses, n_vars, rng=None, max_resamples=None):
    rng = rng or random.Random()
    assignment = random_assignment(n_vars, rng)
    resamples = 0
    bad = violated_clauses(clauses, assignment)
    while bad:
        # TODO: decide what to do from the current violated state, update the
        # assignment and bad accordingly, and account for the running time.
        pass
    return assignment, resamples
```
