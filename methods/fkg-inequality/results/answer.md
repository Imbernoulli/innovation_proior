# The FKG inequality

**Problem.** Across percolation and statistical mechanics, increasing quantities are observed to be
positively correlated ("good events help each other"), but each known case (Griffiths' inequality for
ferromagnets, Harris's inequality for independent percolation) was tied to its model's specific
structure. Find the minimal hypotheses — on the order structure of the configuration space and on the
measure — that force this, with the physics removed.

**Key idea.** On a **finite distributive lattice** (a coordinatewise order structure closed under meet
and join, such as subsets under ⊆ or products of chains), positive correlation of increasing functions
holds for *any* positive measure when the lattice is a chain (the Chebyshev sum inequality). For genuine
partial orders, the measure must not anti-associate the coordinates; the exact condition is
**log-supermodularity** — the *FKG lattice condition*. The independent (product) measure satisfies it
with equality, so it is the borderline case, and "positive association" is the side of the inequality.

## The theorem

**Theorem.** Let Γ be a finite distributive lattice (operations ∧ = meet, ∨ = join), and let μ be a
positive measure on Γ satisfying the **FKG lattice condition**

  (A)   μ(x ∧ y) · μ(x ∨ y) ≥ μ(x) · μ(y)   for all x, y ∈ Γ.

Define the average ⟨f⟩ = Z⁻¹ Σ_{x∈Γ} μ(x) f(x) with Z = Σ_{x∈Γ} μ(x). If f and g are both increasing
(or both decreasing) real functions on Γ, then

  ⟨fg⟩ − ⟨f⟩⟨g⟩ ≥ 0.

(Equivalently μ is *positively associated*: ⟨fg⟩ ≥ ⟨f⟩⟨g⟩ for increasing f, g.) In log form, writing
λ = log μ, condition (A) is supermodularity λ(x∧y) + λ(x∨y) ≥ λ(x) + λ(y).

## Proof

Reduce to strictly positive μ: its support Γ₀ = {x : μ(x) > 0} is a sublattice by (A) (if μ(x), μ(y) > 0
then μ(x∧y)μ(x∨y) > 0), and the average only sees Γ₀. Induct on the length l(Γ) (length of the longest
chain).

**Base case** l(Γ) = 0: Γ is one point, ⟨fg⟩ − ⟨f⟩⟨g⟩ = 0. (More generally, on any *chain*,
⟨fg⟩ − ⟨f⟩⟨g⟩ = (2Z²)⁻¹ Σ_{x,y} μ(x)μ(y)(f(x) − f(y))(g(x) − g(y)) ≥ 0, since comparable x, y give
(f(x) − f(y))(g(x) − g(y)) ≥ 0 termwise — the Chebyshev sum / rearrangement inequality.)

**Inductive step.** Assume the result for all lattices of length ≤ n − 1; let l(Γ) = n ≥ 1, μ > 0,
f, g increasing. Fix an atom a and set
- Γ'_a = {x : x ≥ a},  Γ''_a = {x : x ≱ a}  (equivalently x ∧ a = O),  and
- Γ_a = {x' ∨ a : x' ∈ Γ''_a} ⊆ Γ'_a.

By distributivity all three are distributive sublattices; x' ↦ x' ∨ a is an isomorphism Γ''_a ≅ Γ_a;
and Γ_a is a semi-ideal (down-set) of Γ'_a. Since Γ'_a is the interval [a, I], adding O below a extends
any maximal chain in Γ'_a by one step inside Γ; hence Γ'_a has length ≤ n − 1, and Γ_a ⊆ Γ'_a with
Γ''_a ≅ Γ_a gives length ≤ n − 1 for the other two.

Let Ω = Z²(⟨fg⟩ − ⟨f⟩⟨g⟩) = Σ_{x,y} μ(x)μ(y)(f(x)g(x) − f(x)g(y)). Splitting x, y over the partition
Γ = Γ'_a ⊔ Γ''_a, the two same-block sums equal Ω computed on Γ'_a and on Γ''_a, both ≥ 0 by induction.
Writing C for the symmetrized cross block,
C = Σ'_x Σ''_y μ(x)μ(y)(f(x) − f(y))(g(x) − g(y)), the inductive hypothesis on Γ'_a and Γ''_a gives

  C ≥ (Σ'μ · Σ''μ)⁻¹ (Σ'μf · Σ''μ − Σ'μ · Σ''μf)(Σ'μg · Σ''μ − Σ'μ · Σ''μg).

The same-block terms are nonnegative, so the same lower bound applies to Ω. Here Σ' sums over Γ'_a and
Σ'' over Γ''_a. Thus Ω ≥ 0 follows from ⟨f; Γ''_a⟩ ≤ ⟨f; Γ'_a⟩ (and same for g), which is shown via the
bridge Γ_a:

  ⟨f; μ, Γ''_a⟩ ≤ ⟨f; μ, Γ_a⟩ ≤ ⟨f; μ, Γ'_a⟩.

- **Right inequality.** Γ_a is a semi-ideal of Γ'_a, so its indicator χ is *decreasing* on Γ'_a. Applying
  the inductive hypothesis on Γ'_a to the two increasing functions f and −χ gives ⟨fχ⟩ ≤ ⟨f⟩⟨χ⟩, i.e.
  ⟨f; μ, Γ_a⟩ = ⟨fχ; μ, Γ'_a⟩ / ⟨χ; μ, Γ'_a⟩ ≤ ⟨f; μ, Γ'_a⟩.
- **Left inequality.** Define μ̃(x') = μ(x' ∨ a) and f_a(x') = f(x' ∨ a) on Γ''_a. For y ≤ x in Γ''_a,
  (A) applied to x and y ∨ a gives μ(y)μ(x∨a) ≥ μ(x)μ(y∨a) (using x ∧ (y∨a) = y, x ∨ (y∨a) = x ∨ a), so
  μ̃/μ is *increasing* on Γ''_a. By the inductive hypothesis, the increasing f and increasing μ̃/μ are
  positively correlated under μ; after dividing by ⟨μ̃/μ; μ, Γ''_a⟩, this is
  ⟨f; μ̃, Γ''_a⟩ ≥ ⟨f; μ, Γ''_a⟩. Since f ≤ f_a (because x' ≤ x' ∨ a, f increasing),
  ⟨f; μ, Γ''_a⟩ ≤ ⟨f; μ̃, Γ''_a⟩ ≤ ⟨f_a; μ̃, Γ''_a⟩ = ⟨f; μ, Γ_a⟩.

Hence both factors are ≥ 0, Ω ≥ 0, and the induction closes. ∎

## Harris's product-measure corollary

A product measure μ(x) = ∏_i μ_i(x_i) satisfies (A) with **equality** (meet/min and join/max repackage
the same per-coordinate factors). Hence:

**Corollary (Harris, 1960).** If f and g are increasing functions of independent random variables, then
E[fg] ≥ E[f]E[g]; for increasing events A, B, P(A ∩ B) ≥ P(A)P(B).

A self-contained proof, induction on the number of coordinates by conditioning: for n = 1, with x, y
i.i.d. on a chain, E[(f(x) − f(y))(g(x) − g(y))] ≥ 0 gives E[fg] ≥ E[f]E[g]. For n ≥ 2, condition on
x_1: the marginals f_1(y_1) = E[f | x_1 = y_1], g_1 are increasing in x_1, and for each y_1 the inner
(n − 1)-variable inequality gives E[fg | x_1 = y_1] ≥ f_1(y_1)g_1(y_1). Then
E[fg] = E[E[fg | x_1]] ≥ E[f_1 g_1] ≥ E[f_1]E[g_1] = E[f]E[g], the last by the n = 1 case.

Further corollaries: two decreasing events also satisfy P(A∩B) ≥ P(A)P(B); an increasing and a
decreasing event satisfy P(A∩B) ≤ P(A)P(B); and for A_1, …, A_k all increasing,
P(A_1 ∩ ⋯ ∩ A_k) ≥ ∏_i P(A_i).

