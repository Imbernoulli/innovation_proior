# Cantor's Diagonal Argument

## The problem it solves

Are all infinite sets the same size? Taking "same size" to mean *one-to-one
correspondence* (bijection), many infinite sets that look enormous are in fact
enumerable — the rationals, and even the real algebraic numbers, can be written as a
single list. The question is whether **any** infinite set resists listing, and, more
generally, whether there is a largest possible size. The diagonal argument answers
both: the reals are uncountable, and for every set X the set of its subsets is
strictly larger — so there is no maximum cardinality.

## The key idea

Given **any** purported enumeration of the objects in question, construct a single new
object that is made to **differ from the n-th listed object at the n-th place**. Then
the new object can equal none of them — it differs from object n exactly at position n
— so it is missing from the list, contradicting the claim that the list was complete.
The construction is a self-referential "anti-diagonal": read down the diagonal of the
list-as-array and reverse each entry. It uses nothing about the objects except that
they are indexed by the list-positions and offer a readable, reversible binary choice
at each position, which is why the same one-line move proves both the uncountability
of the reals and the general power-set theorem.

## Predecessor: the 1874 nested-intervals uncountability proof

**Theorem (second theorem, 1874).** For any sequence of reals x₁, x₂, x₃, … and any
interval [a, b], there is a point of [a, b] not in the sequence.

**Proof.** If (a, b) contains at most one sequence term, choose a different real in it.
Otherwise, in the sequence order, let the first two terms lying in (a, b) occur at
positions k₁ < k₂, and sort their values as a₁ < b₁. In (a₁, b₁), take the first two
list-terms lying strictly inside it, at positions k₃ < k₄, and sort them as a₂ < b₂.
Repeat, producing nested open intervals (a₁,b₁) ⊃ (a₂,b₂) ⊃ …, with aₙ increasing and
bₙ decreasing.

*Lemma (induction).* (aₙ, bₙ) excludes x₁, …, x_{2n}. More strongly, it excludes every
term through x_{k_{2n}}. Base: before k₂, no term lies strictly between a₁ and b₁
except the two selected endpoint terms, which the open interval excludes. Step:
(a_{n+1},b_{n+1}) ⊂ (aₙ,bₙ) inherits the exclusions through k_{2n}; because
k_{2n+1}, k_{2n+2} are the first two later positions whose terms lie inside
(aₙ,bₙ), the new open interval also excludes every term through k_{2n+2}. Since
k_{2n} ≥ 2n, in particular xₙ ∉ (aₙ,bₙ) for all n.

*Cases.* (1) If the scan stops at a last (a_L,b_L), at most one term lies inside it, so
some η in that interval — hence in every interval — avoids the whole sequence. (2) If it
never stops and a_∞ = lim aₙ equals b_∞ = lim bₙ (the limits exist by completeness of
ℝ), then η = a_∞ lies strictly between aₙ and bₙ for every n while xₙ does not, so
η ≠ xₙ for all n. (3) If a_∞ < b_∞, every point of [a_∞,b_∞] lies strictly inside every
(aₙ,bₙ) and so avoids the sequence. ∎

**Corollary (uncountability).** If the reals in [a,b] could be listed, applying the
theorem to that list yields a real in [a,b] off the list — contradiction. Combined with
the enumerability of the algebraic numbers, this re-proves Liouville's theorem that
every interval contains transcendentals, by a pure size mismatch.

This proof works but rests on completeness of ℝ, splits into cases, and is tied to the
order/interval structure of the real line — it says nothing about non-numeric sets.

## The diagonal argument (1891)

**Theorem.** Let M be the set of all infinite sequences over two symbols {m, w}. M is
uncountable.

**Proof.** Let E₁, E₂, E₃, … be any sequence of elements of M, with
Eμ = (a_{μ,1}, a_{μ,2}, …) and each a_{μ,ν} ∈ {m, w}. Define E₀ = (b₁, b₂, …) by

  b_ν = w if a_{ν,ν} = m, and b_ν = m if a_{ν,ν} = w  (so b_ν ≠ a_{ν,ν} for all ν).

Then E₀ ∈ M. If E₀ = Eμ for some μ, then b_ν = a_{μ,ν} for all ν, in particular
b_μ = a_{μ,μ} — contradicting b_μ ≠ a_{μ,μ}. So E₀ differs from every Eμ; M is not of the
form E₁, E₂, …. ∎ The two-symbol strings inject into [0,1] by mapping m/w to decimal
digits 1/2; the first differing digit is too large to be canceled by the remaining
tail. If [0,1] were countable, this injection would make M countable too, impossible.
Every non-degenerate real interval follows by rescaling.

No completeness, no order, no limits, no case split — one construction.

## The general theorem (no maximum cardinality)

**Theorem (Cantor's theorem).** For every set X, |X| < |𝒫(X)|, where 𝒫(X) is the set of
all subsets of X (equivalently all functions X → {0,1}).

**Proof.** The map x₀ ↦ {x₀} injects X into 𝒫(X), so |X| ≤ |𝒫(X)|. For strictness, let
f : X → 𝒫(X) be any function and form the anti-diagonal subset

  T = { x ∈ X : x ∉ f(x) }.

For each x: if x ∈ T then x ∉ f(x), so T and f(x) disagree at x, hence T ≠ f(x); if
x ∉ T then x ∈ f(x), so again T ≠ f(x). Thus T is in the image of no x, so f is not
surjective; no bijection X → 𝒫(X) exists. With the injection above, |X| < |𝒫(X)|. ∎

T is the diagonal flip wearing subset clothing: index x, diagonal entry "x ∈ f(x)?",
reversed. Equivalently, in Cantor's 1891 function form: with L a set, M all functions
f : L → {0,1}, and a supposed indexing φ(x,z) of M by z ∈ L, the function
g(x) = 1 − φ(x, x) lies in M but equals no φ(·, z₀) since g(z₀) ≠ φ(z₀, z₀).

Consequently the cardinalities of well-defined sets **have no maximum**: every set is
strictly smaller than its power set, so the hierarchy of infinities climbs without end
(|ℕ| < |𝒫(ℕ)| = |ℝ| < |𝒫(ℝ)| < …).
