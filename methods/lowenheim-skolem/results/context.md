## Research question

First-order theories can describe many infinite structures, but they do not by themselves determine a unique intended infinite size. A theory may force infinitude by saying that every finite list misses something, yet each such sentence is still finite and only quantifies over first-order variables. The question is how far that limitation goes: if a first-order theory has an infinite model in a countable language, what can be said about the range of infinite cardinalities in which it can be satisfied, and to what extent does first-order truth depend on the size of the domain?

## Background

A first-order language has relation, function, and constant symbols, and formulas quantify only over elements of a structure. In a countable language there are only countably many formulas with finitely many free variables. This syntactic countability matters: even if a model is uncountable, first-order formulas can make only countably many schematic demands once their finite variable patterns are fixed.

Elementary substructure is the preservation notion that matches first-order expressivity. A substructure `N subset M` is elementary, written `N prec M`, when for every first-order formula `phi(x_1,...,x_n)` and every tuple `a` from `N`, `N |= phi(a)` exactly when `M |= phi(a)`. The hard direction is existential formulas. Universal, Boolean, and atomic behavior is stable once the interpretations of functions and relations agree on the smaller domain; existential truth can fail if `M` has a witness outside `N`.

The Tarski-Vaught test isolates the witness issue. A substructure `N` of `M` is elementary if whenever `M |= exists y phi(y,a)` for parameters `a` from `N`, there is already some `b` in `N` with `M |= phi(b,a)`. Thus elementary preservation can be reduced to closing the small set under enough choices of witnesses.

Skolemization and Henkin-style witness choices make this closure idea explicit. For each existential demand, one may add a function symbol or choose a function on tuples that returns a witness whenever a witness exists. A set closed under those functions cannot lose existential truths whose parameters are inside it.

Skolem's paradox is the visible pressure point. Set theory may prove, internally, that uncountable sets exist; nevertheless, if it has an infinite model in a countable first-order language, there can be a countable model of the same first-order theory. The apparent conflict is resolved by distinguishing external size from internal first-order statements about bijections. A countable model of set theory may contain an object it regards as uncountable because no bijection inside that model witnesses countability.

## Baselines

- **Plain substructure.** Taking an arbitrary subset and closing it under the language's original function symbols gives a substructure that satisfies universal and atomic formulas when those hold in the larger structure.

- **Complete elementary equivalence.** Compactness and Henkin constructions can build models of a theory with controlled size, producing a model satisfying the same sentences as a given theory.

- **Finite approximations to infinity.** First-order theories can include schemes saying there are at least `n` distinct elements for each finite `n`, encoding infinitude through finitely many first-order sentences.

- **Cardinality by naming elements.** Adding constants to a language can force a model to contain any chosen small parameter set.

## Evaluation settings

The artifact is a theorem and proof in model theory. The natural setting is a first-order language `L`, an `L`-structure `M`, and a chosen subset `A subseteq M`. For the countable case, `L` is countable and `A` is finite or countable.

Stress cases include uncountable models in countable languages, structures with many definable existential choices, and set-theoretic structures where the smaller elementary submodel is externally countable while still satisfying first-order statements that internally describe uncountability.

