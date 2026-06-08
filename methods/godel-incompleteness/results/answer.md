# Gödel's Incompleteness Theorems

## The problem

For a formal system `P` strong enough to express the arithmetic of the natural numbers (the Peano axioms under the logic of *Principia Mathematica*; equally, Peano arithmetic, or set theory) with a fixed alphabet, axiom list, and mechanical inference rules: can `P` be both **consistent** (never proving a formula and its negation) and **complete** (deciding every closed formula `A` by proving `A` or `¬A`), and can `P` prove its own consistency? The hope of Hilbert's program was yes on all counts.

## The key idea

Arithmetize syntax. A truth predicate for arithmetic cannot exist — the diagonal construction turns it into the Liar ("this sentence is false") and forces inconsistency, so arithmetical truth is undefinable in arithmetic. But **provability** is different from truth: a proof is a finite combinatorial object, so "x is a proof of y" is a *decidable* relation and "y is provable" is a single existential quantifier over it. Encode every symbol, formula, and proof as a natural number (Gödel numbering); then the syntactic relations become arithmetical relations, the decidable ones are *representable* inside `P` (true facts about proofs become theorems of `P`), and a diagonal sentence `G` can be built that asserts "I am not provable." Running the diagonal against provability instead of truth yields not a paradox but a **gap**: consistency makes `G` unprovable, and under the standard interpretation it is true because no proof number exists.

## Construction

**Gödel numbering.** Constants `0,f,¬,∨,∀,(,) ↦ 1,3,5,7,9,11,13`; a type-`n` variable `↦ p^n` for a prime `p > 13`. A string of symbol-codes `n₁,…,n_k ↦ 2^{n₁}·3^{n₂}·⋯·p_k^{n_k}`; unique factorization makes decoding effective. A proof (a finite series of formulas) is coded the same way one level up. Distinguish a number `n` from its **numeral** `Z(n) = f⋯f0` (n copies of `f`), the syntactic name substituted into formulas.

**Syntactic relations are primitive recursive** (built by composition, primitive recursion, and bounded quantification/search): substitution `Sb(x; v/y)`, `Form(x)`, `Ax(x)`, immediate consequence `Fl(x,y,z)`, proof-schema `Bw(x)`, and
- `x B y ≡ Bw(x) & [l(x)]Gl x = y` — "x is a proof of formula y" (decidable),
- `Bew(y) ≡ ∃x (x B y)` — provability, the lone non-primitive-recursive relation in this list (a single unbounded existential over the decidable proof-relation; provability is semi-decidable).

**Representability (Proposition V).** For every primitive recursive relation `R(x⃗)` there is a formula `r` with free variables `u⃗` such that
- `R(x⃗) ⟹ ⊢ Sb(r; u⃗/Z(x⃗))`,
- `¬R(x⃗) ⟹ ⊢ Neg Sb(r; u⃗/Z(x⃗))`.

Proof by induction on the build-up of the defining function: each computation step (composition, successor, an instance of the recursion equations) is mirrored by a provable arithmetical axiom.

**Diagonalization.** For recursive `c`, `Q(x,y) ≡ ¬(x B_c Sb(y; 19/Z(y)))` is primitive recursive and represented by `q` (free variables 17, 19). Set `p = 17 Gen q` and `r = Sb(q; 19/Z(p))`. Since substitution and generalization on different variables commute,
`Sb(p; 19/Z(p)) = 17 Gen Sb(q; 19/Z(p)) = 17 Gen r`,
so `17 Gen r` is the formula got by feeding `p` its own numeral, and it **asserts its own unprovability**. Substituting `p` into the represented implications gives the engine:
- (15) `¬(x B_c (17 Gen r)) ⟹ ⊢_c Sb(r; 17/Z(x))`,
- (16) `(x B_c (17 Gen r)) ⟹ ⊢_c Neg Sb(r; 17/Z(x))`.

`ω-consistency`: there is no formula `a` with `⊢_c Sb(a; v/Z(n))` for every `n` *and* `⊢_c Neg(v Gen a)` — "you cannot prove every instance and also deny the universal." It implies consistency but is strictly stronger.

## Theorem (First Incompleteness)

For every ω-consistent, recursively specified class of axioms `c` over `P`, the sentence `17 Gen r` is **undecidable**: neither `17 Gen r` nor `Neg(17 Gen r)` is `c`-provable.

*Proof.* **(a) `17 Gen r` unprovable — needs only consistency.** If `⊢_c 17 Gen r`, some `n` codes a proof: `n B_c (17 Gen r)`. By (16), `⊢_c Neg Sb(r; 17/Z(n))`. But `17 Gen r` is `∀(17)r`, so universal instantiation gives `⊢_c Sb(r; 17/Z(n))`. A formula and its negation are proved — `c` is inconsistent. So consistency ⇒ `17 Gen r` is not `c`-provable.
**(b) `Neg(17 Gen r)` unprovable — needs ω-consistency.** Suppose `⊢_c Neg(17 Gen r)`. Since ω-consistency implies consistency, (a) gives that `17 Gen r` is not `c`-provable, so `¬(n B_c (17 Gen r))` for every `n`; by (15), `⊢_c Sb(r; 17/Z(n))` for every `n`. Together with `⊢_c Neg(17 Gen r)` this is exactly the configuration ω-consistency forbids (take `a := r`, `v := 17`). So ω-consistency ⇒ `Neg(17 Gen r)` is not `c`-provable. ∎

The extra hypothesis in (b) is doing real work: adjoining `Neg(17 Gen r)` to a consistent `c` gives a consistent but **not** ω-consistent system (every instance `Sb(r;17/Z(n))` remains provable while `Neg(17 Gen r)` is now an axiom), in which this negation is provable.

Under plain consistency alone one still gets a primitive recursive `r` such that every instance `Sb(r; 17/Z(n))` is provable yet `17 Gen r` is not. Under ω-consistency, the undecidable sentence is a genuine statement `∀x F(x)` about ordinary integers with `F` decidable, because every primitive recursive relation is arithmetical (finite sequences can be coded by a pair `(n,d)` via the Chinese remainder theorem, using residues `n mod (1+(i+1)d)`). Hence elementary arithmetic, the *Principia* system, and set theory are incomplete under Gödel's ω-consistency hypothesis.

## Theorem (Second Incompleteness, sketch)

Direction (a) above used only consistency: `Wid(c) ⟹ ¬Bew_c(17 Gen r)` (23), i.e. `Wid(c) ⟹ (∀x) Q(x,p)` (24). This argument is itself formalizable in `P`. Let `w` express consistency, `Wid(c) ≡ ∃x(Form(x) & ¬Bew_c(x))` ("some formula is unprovable"). Then (24) formalizes to `⊢_P w Imp (17 Gen r)`, hence also in the extension by `c`. If `c` proved `w`, it would prove `17 Gen r`, contradicting (a) when `c` is consistent. Therefore

**a consistent `P` cannot prove its own consistency** (`P ⊬ Con(P)`; take `c` empty), and likewise for set theory and classical analysis.

This is no formal contradiction of Hilbert's standpoint per se — a finitary consistency proof not formalizable in `P` is not excluded — but it ends the hope that a single system rich enough for arithmetic can be complete under the stated consistency standard or certify its own consistency from within.

## Causal chain

undefinability of arithmetical truth (Liar, via diagonalization) → the culprit is truth's rigid `True(⌜S⌝)↔S`, not self-reference → provability is decidable-plus-one-existential, hence arithmetizable where truth is not → Gödel-number the syntax → syntactic relations are primitive recursive → primitive recursive ⇒ representable in `P` → diagonalize provability to build `G = 17 Gen r` asserting its own unprovability → consistency makes `G` unprovable, ω-consistency makes `¬G` unprovable → incompleteness for a statement about plain integers → the unprovability-of-`G` argument is formalizable, so a consistent `P` cannot prove `Con(P)`.
