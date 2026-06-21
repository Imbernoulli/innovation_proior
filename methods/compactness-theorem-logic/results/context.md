## Research question

First-order logic lets a theory contain infinitely many sentences. The basic semantic question is whether such a theory has a model. For a finite theory, the question is local: there are only finitely many constraints to satisfy at once. For an infinite theory, every sentence must be made true simultaneously.

The question is what local-to-global principle governs this. If every finite subset of a theory can be satisfied, is that already enough to force a model of the whole theory?

The same question matters beyond bare existence. If finite fragments control infinite satisfiability, then it bears on whether first-order logic can distinguish standard finiteness from certain infinite or nonstandard configurations using a single infinite collection of axioms.

## Background

A first-order language has relation, function, and constant symbols, variables, terms, formulas, and sentences. A structure interprets the nonlogical symbols over a domain, and a sentence is satisfiable if some structure makes it true. A theory is a set of sentences; its models are the structures satisfying every sentence in the set.

Syntactic consequence and semantic consequence are connected by the completeness theorem: if a sentence is true in every model of a theory, then it is derivable from the theory by a formal proof system. Formal proofs are finite objects: a derivation from an infinite theory can use only finitely many premises.

Consistency is syntactic, while satisfiability is semantic. A consistent theory is one from which no contradiction is derivable. Completeness turns consistency into satisfiability by building a model, usually after extending the language with witnesses and extending the theory to a maximally consistent Henkin theory. The resulting term model interprets terms as elements and makes exactly the selected sentences true.

There is also a semantic product construction. Given structures indexed by a set and an ultrafilter on that index set, the ultraproduct identifies sequences that agree on a large set of indices. Los's theorem says that a first-order formula holds in the ultraproduct exactly when the set of indices where it holds is large. This makes ultraproducts a way to assemble many models into one global model.

Nonstandard models arise from the same setting. If a first-order theory has arbitrarily large finite models, or if arithmetic is expanded with a constant required to exceed every standard numeral, then each finite piece can be realized. A model satisfying all the first-order requirements at once may then contain elements not intended in the original standard picture.

## Baselines

**Direct infinite satisfaction.** One can try to build a structure that satisfies the sentences in the theory one after another, extending the structure as each new constraint is added.

**Finite satisfiability checks.** Verifying that each finite subset has a model is a local test, carried out fragment by fragment, with a possibly different model for each finite fragment.

**Syntactic consistency alone.** If no contradiction is derivable, the theory is formally coherent. Consistency is a property of the proof system, stated without reference to any particular domain or interpretation.

**Completeness theorem.** Completeness converts syntactic consistency into a model by a Henkin construction: a consistent theory is extended to a maximal consistent Henkin theory in an expanded language, and the term model of that theory realizes it.

**Ultraproducts.** Products modulo ultrafilters preserve first-order truth in the Los-theorem sense. The construction takes an index set, a family of models, and an ultrafilter, and yields a single structure whose first-order truths are those holding on large index sets.

## Evaluation settings

The artifact is a theorem and proof in first-order model theory. The natural input is an arbitrary first-order language and a set `T` of sentences. Success means proving the equivalence between satisfiability of `T` and satisfiability of every finite subset of `T`.

The proof should handle arbitrary infinite theories, not just countable ones. It should make clear where first-order logic is being used: formal proofs are finite, and Los's theorem applies to first-order formulas. It should also distinguish the theorem from stronger logics where compactness can fail.

Stress cases include adding a constant `c` to arithmetic with axioms `c > 0`, `c > 1`, `c > 2`, and so on; every finite part has a standard natural-number interpretation, while a model of all the sentences must contain a nonstandard element. Another stress case is a theory saying that there are at least `n` distinct elements for every natural number `n`; each finite fragment has a finite model, while the whole theory forces an infinite model.

## Proof artifact

The final artifact should state compactness for first-order logic:

`T` has a model if and only if every finite subset of `T` has a model.

The main proof should run through the Henkin/maximal-consistent-set route: finite satisfiability implies syntactic consistency, consistency extends to a complete Henkin theory in an expanded language, and the term model of that theory satisfies the original `T`. A companion proof can show the same idea semantically with ultraproducts: index models by finite fragments, choose an ultrafilter containing the sets of fragments that include each sentence, and use Los's theorem to make every sentence true in the ultraproduct.

The artifact should close with consequences: compactness links directly to completeness, proves that infinite first-order satisfiability is controlled by finite fragments, and explains the existence of nonstandard models in settings where every finite approximation is standard-looking.
