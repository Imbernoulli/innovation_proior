# Ramsey's theorem

## Problem solved

Partition all the *r*-element subsets of a set into finitely many colour classes. Is there always a large *homogeneous* sub-set — one all of whose *r*-element subsets receive a single colour? Pigeonhole answers this for *r* = 1 (colouring points) but says nothing once pairs, triples, or larger subsets are coloured, because a large colour class of *pairs* need contain no large set whose *every* pair is that colour. The theorem says homogeneity is nonetheless unavoidable: an infinite set always has an infinite homogeneous sub-set, and a large enough finite set always has a homogeneous sub-set of any prescribed size — with an explicit, computable threshold. It was needed as a tool to decide consistency of universal first-order formulas (find an infinite sub-domain on which every *r*-tuple realizes one truth-type), but the combinatorial statement stands on its own.

## Key idea

Use the pigeonhole principle not once globally but **locally and repeatedly**: fix a vertex, pass to the sub-population that is the same colour to it on the relevant subsets (the "majority neighbourhood"), and recurse, lowering the subset-size *r* by one each time a vertex is absorbed. The infinite case runs this forever and then pigeonholes the resulting sequence of vertex-colours; the finite case runs it a bounded number of times, paying a bookkeeping cost that is packaged as a recursion for the threshold. Ramsey's theorem is the pigeonhole principle lifted from points to *r*-subsets.

## The infinite theorem

**Theorem A.** Let Γ be infinite and *r*, *μ* positive integers. Partition all *r*-element subsets of Γ into μ classes *C*₁, …, *C*_μ. Then (using the axiom of choice) Γ has an infinite sub-class Δ all of whose *r*-element subsets lie in a single *C_i*.

**Proof.** Induct on *r*.

*Base r = 1:* infinitely many points in μ classes force one class to be infinite (pigeonhole).

*Step, μ = 2:* assume the theorem for *r* = *p*−1. Given a 2-colouring of the *p*-subsets, try to build a sequence: find *a*₁ and an infinite Γ₁ ⊆ Γ\{*a*₁} with every *p*-subset {*a*₁} ∪ (*p*−1 of Γ₁) in *C*₁; then *a*₂ ∈ Γ₁ with infinite Γ₂ ⊆ Γ₁\{*a*₂} likewise; and so on.
- *If the process never fails*, Δ = {*a*₁, *a*₂, …} works: the *a*'s are distinct (each lies in all earlier Γ's, which exclude the earlier *a*'s), and any *p*-subset of Δ, taken with its least-index element *a*_s, has its other *p*−1 members in Γ_s, hence is *C*₁.
- *If it fails at stage n*, let *y*₁ ∈ Γ_{n−1}. Colour the (*p*−1)-subsets of Γ_{n−1}\{*y*₁} by whether adjoining *y*₁ gives *C*₁ or *C*₂; by induction there is an infinite Δ₁ on which this is constant. That constant colour cannot be *C*₁ (else *y*₁, Δ₁ would have served at stage *n*), so all *p*-subsets {*y*₁} ∪ (*p*−1 of Δ₁) are *C*₂. Repeat inside Δ₁ with *y*₂, *y*₃, …, each forced into *C*₂. Then {*y*₁, *y*₂, …} is an infinite *C*₂-homogeneous set.

*Step, μ > 2:* merge two colours into one and apply the (μ−1)-colour case; if the homogeneous set lands in the merged class, apply the μ = 2 case inside it. ∎

## The finite theorem, with an explicit bound

**Theorem B.** For all *r*, *n*, *μ* there is *m*₀ = *h*(*r*, *n*, μ) such that any μ-colouring of the *r*-subsets of any *m*-set with *m* ≥ *m*₀ contains a homogeneous *n*-set.

Compactness deduces Theorem B from Theorem A but yields **no** value for *m*₀; a usable bound requires a direct recursion. The induction goes through a stronger, self-feeding statement.

**Theorem C (the (n, k) reservoir form, μ = 2).** For *n* + *k* ≥ *r* there is *m*₀ such that any 2-colouring of the *r*-subsets of an *m*-set, *m* ≥ *m*₀, yields disjoint sub-classes Δ_n and Λ_k with all *r*-subsets of Δ_n ∪ Λ_k that meet Δ_n lying in a single colour. (For fixed *r*: C for *n*, *k* asserts more than B for *n* but less than B for *n*+*k*.)

Define, with an auxiliary *g*:

```
f(1, n, k) = max(2n − 1, n + k)
f(r, 1, k) = f(r − 1, k − r + 2, r − 2) + 1            (k + 1 > r)
g(r, 0, k) = max(r − 1, k)
g(r, n, k) = f(r, 1, g(r, n − 1, k))                   (n ≥ 1)
f(r, n, k) = f(r, n − 1, g(r, n, k))                   (n > 1)
```

*Theorem C holds with m₀ = f(r, n, k).* Induction on *r*, then on *n*. The base *r* = 1 is pigeonhole plus a fit condition. The base *n* = 1 peels one vertex *x*, recolours the (*r*−1)-subsets of the rest by adjoining *x*, and invokes the *r*−1 form with parameters (*k*−*r*+2, *r*−2) — chosen so the resulting *k*-set has every (*r*−1)-subset monochromatic (its reservoir of *r*−2 < *r*−1 members is too small to fill one alone). The step *n* > 1 secures a Δ_{n−1} with a large reservoir *g*(*r*, *n*, *k*), then promotes reservoir vertices into the *n*-th core point one at a time via the *n* = 1 result; *g* is sized so that if a promotion lands in the wrong colour there is room for another, and after at most *n* peels either the core grows in the original colour or the peeled vertices themselves form a homogeneous set in the other colour. ∎

**From C to B.** For μ = 2 take *m*₀ = *f*(*r*, *n*−*r*+1, *r*−1) =: *h*(*r*, *n*, 2): an *r*-subset of that ((*n*−*r*+1)+(*r*−1)) = *n*-point set must meet the core, so the whole *n*-set is homogeneous. For μ > 2, peel colours pairwise:

```
h(r, n, 2) = f(r, n − r + 1, r − 1)
h(r, n, μ) = h(r, h(r, n, μ − 1), 2)                   (μ > 2)
```

merging *C*₂, …, *C*_μ into one super-colour, applying the 2-colour bound, and recursing inside the super-colour class. ∎

## The graph case (r = 2): the sharp, iterated-pigeonhole form

For pairs the generic bound is *h*(2, *n*, 2) = 2^{n(n−1)/2}, "altogether excessive." A direct local pigeonhole gives a factorial: for *k* colours, *m*₀ = *k*·(*n*+1)! forces a homogeneous *n*-set (a fixed vertex has *m*−1 incident pairs in *k* colours, so a same-colour neighbourhood of size ≥ *k*·*n*! exists; recurse and append the vertex).

For two colours the cleanest object is the **Ramsey number** *R*(*s*, *t*): the least *N* forcing, in any red/blue colouring of all pairs, a red *K_s* or a blue *K_t* (equivalently, every *N*-vertex graph has a clique of size *s* or an independent set of size *t*).

- *R*(*s*, *t*) = *R*(*t*, *s*); *R*(2, *t*) = *t*.
- **Recursion (fix a vertex, pass to the majority-colour neighbourhood):**
  *R*(*s*, *t*) ≤ *R*(*s*−1, *t*) + *R*(*s*, *t*−1).
  With *n* = *R*(*s*−1, *t*) + *R*(*s*, *t*−1) vertices, a fixed *v* has either ≥ *R*(*s*−1, *t*) red-neighbours (yielding a red *K*_{s−1} → red *K_s* with *v*, or a blue *K_t*) or ≥ *R*(*s*, *t*−1) blue-neighbours (symmetric).
- **Closed bound (Pascal's identity):**
  *R*(*s*, *t*) ≤ *C*(*s*+*t*−2, *s*−1), since *C*(*s*+*t*−3, *s*−2) + *C*(*s*+*t*−3, *s*−1) = *C*(*s*+*t*−2, *s*−1).
- **Diagonal:** *R*(*s*, *s*) ≤ *C*(2*s*−2, *s*−1) ≤ 4^s/√s < 4^s. Equivalently every 2-colouring of an *N*-point set's pairs has a monochromatic set of size ≳ (1/2)·log₂ *N* — exponential, matching the existence floor.

**Worked instance: *R*(3, 3) = 6.** The bound gives *R*(3, 3) ≤ *R*(2, 3) + *R*(3, 2) = 3 + 3 = 6. Tightness: among six people fix *P*; of the other five, *P* matches at least three the same way (pigeonhole, ⌈5/2⌉ = 3); among those three, either a matching pair completes a monochromatic triple with *P*, or all three pairwise take the other colour — a monochromatic triple. Five is not enough: the 5-cycle (red on the cycle, blue on the diagonals) has neither a red nor a blue triangle. So *R*(3, 3) = 6.

## Optional checker

A brute-force verifier on tiny instances, confirming the threshold forces homogeneity and that one less does not.

```python
from itertools import combinations, product

def is_homogeneous(subset, r, coloring):
    """True iff every r-subset of `subset` gets one colour."""
    return len({coloring[s] for s in combinations(subset, r)}) == 1

def has_homogeneous_set(vertices, r, n, coloring):
    return any(is_homogeneous(c, r, coloring) for c in combinations(vertices, n))

def forces_homogeneous_set(m, r, n, mu):
    """Does EVERY mu-colouring of the r-subsets of an m-set have a mono n-set?
    Brute force over all colourings — feasible only for tiny m, r, mu."""
    verts = list(range(m))
    subs = list(combinations(verts, r))
    for assign in product(range(mu), repeat=len(subs)):
        coloring = dict(zip(subs, assign))
        if not has_homogeneous_set(verts, r, n, coloring):
            return False   # found a colouring with no homogeneous n-set
    return True

# R(3,3) = 6:  6 forces a mono triangle (pairs, 2 colours, target 3); 5 does not.
assert forces_homogeneous_set(6, r=2, n=3, mu=2) is True
assert forces_homogeneous_set(5, r=2, n=3, mu=2) is False
```
