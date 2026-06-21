# Context: deciding consistency of universal first-order formulas, and the combinatorial regularity it demands

## Research question

The leading open problem of mathematical logic at this time is the *Entscheidungsproblem* of Hilbert and Ackermann (*Grundzüge der theoretischen Logik*, pp. 72–81): is there a mechanical procedure that, given any formula of the first-order functional calculus, decides whether it is *valid* (true under every interpretation of its predicate symbols) — equivalently, whether its negation is *consistent*, i.e. satisfiable in some domain? The two forms are interchangeable, since a formula is consistent exactly when its contradictory is not valid.

The general problem is wide open; the cases solved so far are carved out by the *syntactic shape* of the formula. A natural next target is formulas in prenex normal form whose quantifier prefix is *universal* — every apparent variable governed by "for all," possibly behind a leading block of existentials. To decide consistency of such a formula over an infinite domain one must understand the following. With finitely many predicate symbols of bounded arity, the truth-pattern of the predicates on a given *r*-tuple of individuals realizes one of finitely many "types"; assigning each *r*-subset its type is a colouring of the *r*-subsets by finitely many colours. A universal formula's behaviour is governed by how those types are distributed across tuples, and it becomes simple to read off when a single type recurs uniformly. So the question turns on a purely combinatorial fact, stripped of the logic:

> Take a large (or infinite) set of points. Look at *all* its *r*-element subsets and sort them, however one likes, into finitely many bins. **Must there be a large (or infinite) sub-set all of whose *r*-element subsets fall in one and the same bin?**

This is wanted as a *tool* — a fact about finite and infinite classes, of independent interest, to be established before the logic is attempted. And not merely existence: for the finite version one would like to know *how large* the starting set must be as a function of the number of bins, the subset size *r*, and the target size, because a computable bound is what a decision procedure could actually use.

## Background

**The pigeonhole principle (Dirichlet / Dedekind, *le principe des tiroirs*).** If *k* objects are placed into *μ* boxes and *k* > *μ*·(*h*−1), then some box contains at least *h* objects. In the homogeneity language above this is the case *r* = 1: the "subsets" are single points, sorting points into *μ* bins, and a bin with ≥ *h* points is itself a homogeneous *h*-set. The least size forcing a monochromatic *h*-set is *μ*(*h*−1)+1; for an infinite point set in finitely many bins, one bin is infinite. So the demand is elementary when *r* = 1. The interesting regime is when the things coloured are pairs, triples, *r*-subsets, where a large colour class need contain no large *internally homogeneous* set.

**Finitely many "types" of an *r*-tuple.** With finitely many predicate symbols of bounded arity, the predicates' truth-pattern on a given *r*-tuple takes one of finitely many values, so the logical question is a homogeneity question about a finite colouring of the *r*-subsets.

**Prior solved sub-cases of the Entscheidungsproblem (the field state).**
- *Behmann* (Math. Ann. 86, 1922): the case of *monadic* predicates (functions of one variable).
- *Bernays and Schönfinkel* (Math. Ann. 99, 1928): two individual apparent variables, and the prefix class ∃*∀*; they exclude the identity relation.

These advances all restrict the syntactic form.

**The infinite vs. finite contrast.** Reasoning about an infinite domain is often cleaner: "infinitely many of these are red" is a usable statement that never runs out, whereas in a finite set one must track exact counts. A regularity result is therefore often easiest to see first in the infinite, then transported to the finite — but the transport may or may not preserve quantitative information.

**The axiom of choice ("axiom of selections").** A construction in an infinite domain that picks one element after another, infinitely often, each time passing to an infinite sub-domain, requires a choice principle; it must be assumed whenever the domain is infinite. In a finite domain only finitely many selections are made and no such axiom is needed.

**The existence floor for the *r* = 2 quantitative case.** A "typical" 2-colouring of the pairs of an *N*-point set can avoid a monochromatic *s*-set whenever *N* is below roughly 2^{s/2}, since the number of *s*-subsets times the chance a fixed one is monochromatic, *C*(*N*,*s*)·2^{1−*C*(*s*,2)}, drops below 1 there. So no forcing-size bound for a monochromatic *s*-set can be smaller than exponential in *s*; the open question is how close a proof's bound comes to this floor. (This is an existence statement about colourings, not a construction.)

## Baselines

The prior art a regularity theorem for *r*-subset colourings would be measured against or reuse:

**Pigeonhole (the *r* = 1 case).** Stated above.

**The six-person folklore for pairs.** For two-colouring the pairs of a set, it is known by hand that six people always contain three who pairwise match: focus on one person *P*; of the other five, *P* relates the same way to at least three; among those three, either some pair matches *P*'s relation (a monochromatic triple with *P*) or all three pairwise take the other relation.

**Behmann; Bernays–Schönfinkel decision procedures.** Substantive procedures for their syntactic classes (monadic; two variables; ∃*∀*).

**Compactness-style transfer from infinite to finite (the König-lemma idea).** It is a standard pattern that a statement about all finite initial segments follows from a statement about the infinite object: organize finite counterexamples into a finitely-branching tree and extract an infinite branch (König, 1927), or nest infinitely-agreeing sub-families of finite colourings into one infinite colouring.

## Evaluation settings

The deliverable is a *theorem with a proof*, so the "evaluation" is mathematical, not experimental.

- **Objects.** A finite or infinite class Γ; a fixed subset-size *r* ≥ 1; a number of colours *μ* ≥ 1; a colouring assigning each *r*-element subset of Γ one of the classes *C*₁,…,*C*_μ. A sub-class Δ ⊆ Γ is *homogeneous* (monochromatic) if all its *r*-element subsets receive a single colour.
- **What a solution must establish.** Infinite case: an infinite homogeneous Δ always exists. Finite case: for every *r*, *μ*, target *n* there is a threshold *m*₀ so any Γ with ≥ *m*₀ members forces a homogeneous Δ of ≥ *n* members; ideally an explicit *m*₀(*r*, *μ*, *n*).
- **Yardsticks.** (a) Generality — all *r*, all finite *μ*, both finite and infinite Γ. (b) For the finite case, the size of *m*₀ against the existence floor (exponential in the target for *r* = 2). (c) Hand-checkable small instances: for *r* = 2, *μ* = 2, target 3, that 6 points always force a homogeneous triple while 5 do not (the 5-cycle is the witness for 5).
- **Protocol.** Prove the infinite statement; prove the finite statement, preferably with a computed bound rather than pure existence; sanity-check the smallest cases; note any gap to the existence floor. No dataset, no train/test split — the artifact is a theorem and its proof.

## Code framework

The natural artifact is a proof, not a program, so the only code that belongs is a small *checker* one could run on tiny instances to confirm the statement (does a given size really force a homogeneous set; is a smaller size insufficient?). Written purely in pre-method terms — colourings of *r*-subsets and a brute-force homogeneity test — it leaves an empty slot exactly where the forcing argument and bound will go.

```python
from itertools import combinations

def r_subsets(vertices, r):
    """All r-element subsets of the ground set — the things being coloured."""
    return list(combinations(vertices, r))

def is_homogeneous(subset, r, coloring):
    """True iff every r-element subset of `subset` gets a single colour under `coloring`."""
    colors = {coloring[s] for s in combinations(subset, r)}
    return len(colors) == 1

def has_homogeneous_set(vertices, r, n, coloring):
    """Brute force: does this one colouring contain a monochromatic n-set?"""
    for cand in combinations(vertices, n):
        if is_homogeneous(cand, r, coloring):
            return True
    return False

def forces_homogeneous_set(m, r, n, mu):
    """
    Does EVERY mu-colouring of the r-subsets of an m-point set contain a
    monochromatic n-set?  (Brute force over all colourings — only tiny m.)
    Used to check, on small instances, that some threshold m0 forces it.
    """
    vertices = list(range(m))
    subs = r_subsets(vertices, r)
    # iterate over all mu^|subs| colourings ... (feasible only for tiny m, r)
    raise NotImplementedError  # enumeration harness, not the mathematics

# ---- the actual contribution: not a program but a theorem + proof ----
def forcing_threshold(r, n, mu):
    """
    Return a size m0 such that any m >= m0 forces a monochromatic n-set
    when the r-subsets are mu-coloured.
    """
    # TODO: the result we will establish, and the argument that yields it
    pass
```
