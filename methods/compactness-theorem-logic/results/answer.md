# Compactness Theorem for First-Order Logic

## The Problem It Solves

An infinite first-order theory can contain arbitrarily many constraints. The compactness theorem identifies the exact local test for satisfiability: there is no special infinite obstruction. If a theory fails to have a model, the failure is already visible in some finite fragment.

## Theorem

Let `T` be a set of first-order sentences. Then `T` has a model if and only if every finite subset of `T` has a model.

## Proof by Maximal Henkin Theories

The forward direction is immediate. If a structure `M` satisfies every sentence in `T`, then `M` satisfies every finite subset of `T`.

For the converse, assume every finite subset of `T` has a model. Then `T` is syntactically consistent. If `T` proved a contradiction, that proof would use only finitely many premises from `T`; those premises would form a finite unsatisfiable subset, contradicting the hypothesis and soundness.

Use the model-existence construction for first-order logic. Expand the language by adding Henkin constants so that each existential formula has a named witness: for each relevant formula `exists x phi(x)`, add a fresh constant `c_phi` and the sentence

`exists x phi(x) -> phi(c_phi)`.

The Henkin extension can be arranged consistently, because any contradiction from added witness axioms would occur at a finite stage, and the constants added at that stage are fresh. Extend the resulting consistent Henkin theory to a maximal consistent theory `H`.

Build the term model `M_H`:

- Its elements are equivalence classes `[t]` of closed terms, where `s ~ t` iff `s = t` is in `H`.
- Each constant `c` is interpreted as `[c]`.
- Each function symbol `f` is interpreted by `f([t1],...,[tn]) = [f(t1,...,tn)]`.
- Each relation symbol `R` holds of `[t1],...,[tn]` iff `R(t1,...,tn)` is in `H`.

Equality axioms in `H` make these interpretations well defined. The truth lemma follows by induction on formulas:

`M_H |= phi([t1],...,[tn])` iff `phi(t1,...,tn) in H`.

Atomic formulas hold by construction. Boolean connectives follow from maximal consistency. For existential formulas, the Henkin witness property gives the crucial direction: if `exists x phi(x,t)` is in `H`, then some witness term `c` has `phi(c,t)` in `H`, so the model realizes the existential. The reverse direction follows because every element of the term model is represented by a closed term.

For every sentence `sigma in H`, the truth lemma gives `M_H |= sigma`. Since the original theory `T` is contained in `H`, `M_H` is a model of `T`. Therefore every finitely satisfiable first-order theory is satisfiable.

## Ultraproduct Proof

Assume again that every finite subset of `T` is satisfiable. Let `I` be the set of finite subsets of `T`. For each `Delta in I`, choose a model `M_Delta` of `Delta`.

For each sentence `sigma in T`, let

`A_sigma = { Delta in I : sigma in Delta }`.

The family `{A_sigma : sigma in T}` has the finite intersection property, because any finite intersection contains all finite fragments that include the corresponding finite set of sentences. Extend this family to an ultrafilter `U` on `I`.

Form the ultraproduct

`M = prod_{Delta in I} M_Delta / U`.

For any `sigma in T`, the set of indices where `sigma` is true contains `A_sigma`, since every `M_Delta` with `sigma in Delta` satisfies `sigma`. Hence

`{ Delta in I : M_Delta |= sigma } in U`.

By Los's theorem, `M |= sigma`. This holds for every `sigma in T`, so the ultraproduct `M` is a model of `T`.

## Connection to Completeness

Compactness is tightly tied to completeness. Completeness says that every consistent first-order theory has a model. Since proofs are finite, if every finite subset of `T` is satisfiable, no finite proof can derive a contradiction from `T`; hence `T` is consistent. Completeness then supplies a model. Conversely, compactness can be used as a model-theoretic reflection of the same finitary proof fact: semantic failure of an infinite first-order theory must be witnessed finitely.

## Consequence: Nonstandard Models

Add a new constant `c` to arithmetic and include the sentences

`c > 0`, `c > 1`, `c > 2`, ...

alongside the chosen first-order arithmetic axioms. Every finite subset is satisfiable in the standard natural numbers by interpreting `c` as a sufficiently large natural number. By compactness, the whole theory has a model. In that model, `c` is greater than every standard numeral, so the model contains a nonstandard element.

This is the central force of compactness: first-order logic cannot block such an element by infinitely many finite lower-bound requirements, because each finite fragment is satisfiable.
