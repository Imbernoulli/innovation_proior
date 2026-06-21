## Research question

First-order theories can describe many infinite structures, but they do not by themselves determine a unique intended infinite size. A theory may force infinitude by saying that every finite list misses something, yet each such sentence is still finite and only quantifies over first-order variables. The question is how far that limitation goes: if a first-order theory has an infinite model in a countable language, must it also have a much smaller infinite model satisfying exactly the same first-order sentences?

The useful target is stronger than merely finding any smaller model of the same axioms. If a large structure `M` is already given, a sharp result would carve from it a substructure `N` of controlled cardinality such that every first-order formula with parameters from `N` has the same truth value in `N` as in `M`. This would show that first-order truth is preserved even after shrinking the domain, so long as the smaller domain contains enough witnesses.

## Background

A first-order language has relation, function, and constant symbols, and formulas quantify only over elements of a structure. In a countable language there are only countably many formulas with finitely many free variables. This syntactic countability matters: even if a model is uncountable, first-order formulas can make only countably many schematic demands once their finite variable patterns are fixed.

Elementary substructure is the preservation notion that matches first-order expressivity. A substructure `N subset M` is elementary, written `N prec M`, when for every first-order formula `phi(x_1,...,x_n)` and every tuple `a` from `N`, `N |= phi(a)` exactly when `M |= phi(a)`. The hard direction is existential formulas. Universal, Boolean, and atomic behavior is stable once the interpretations of functions and relations agree on the smaller domain; existential truth can fail if `M` has a witness outside `N`.

The Tarski-Vaught test isolates the witness issue. A substructure `N` of `M` is elementary if whenever `M |= exists y phi(y,a)` for parameters `a` from `N`, there is already some `b` in `N` with `M |= phi(b,a)`. Thus elementary preservation can be reduced to closing the small set under enough choices of witnesses.

Skolemization and Henkin-style witness choices make this closure idea explicit. For each existential demand, one may add a function symbol or choose a function on tuples that returns a witness whenever a witness exists. A set closed under those functions cannot lose existential truths whose parameters are inside it.

Skolem's paradox is the visible pressure point. Set theory may prove, internally, that uncountable sets exist; nevertheless, if it has an infinite model in a countable first-order language, there can be a countable model of the same first-order theory. The apparent conflict is resolved by distinguishing external size from internal first-order statements about bijections. A countable model of set theory may contain an object it regards as uncountable because no bijection inside that model witnesses countability.

## Baselines

- **Plain substructure.** Taking an arbitrary subset and closing it under the language's original function symbols gives a substructure. Gap: existential formulas may be true in the large structure only because of witnesses outside the chosen substructure.

- **Complete elementary equivalence.** Compactness and Henkin constructions can build models of a theory with controlled size. Gap: this can produce a model satisfying the same sentences without necessarily preserving the given structure's parameter-by-parameter truth.

- **Finite approximations to infinity.** First-order theories can include schemes saying there are at least `n` distinct elements for each finite `n`. Gap: no single first-order sentence in the ordinary language says "the domain has exactly this infinite cardinality," so finite expressibility does not pin down intended infinite size.

- **Cardinality by naming elements.** Adding constants can force a model to contain a chosen small parameter set. Gap: naming elements alone does not ensure existential witnesses for formulas using those parameters are also present.

## Evaluation settings

The artifact is a theorem and proof in model theory. The natural setting is a first-order language `L`, an `L`-structure `M`, and a chosen subset `A subseteq M`. For the countable case, `L` is countable and `A` is finite or countable. The proof should produce an elementary substructure containing `A` whose size is no larger than `max(|A|, |L|, aleph_0)`.

Stress cases include uncountable models in countable languages, structures with many definable existential choices, and set-theoretic structures where the smaller elementary submodel is externally countable while still satisfying first-order statements that internally describe uncountability.

## Proof artifact

The final artifact should state the downward theorem in Skolem-hull form: for any `L`-structure `M` and any `A subseteq M`, there is `N prec M` with `A subseteq N` and `|N| <= max(|A|, |L|, aleph_0)`. In a countable language, every infinite structure has a countable elementary substructure containing any prescribed countable set.

The proof should choose witness functions for all formulas, close `A` under the original functions and these witness choices by an omega-stage construction, bound the size at each stage, and then apply the Tarski-Vaught test. The size control should come from the fact that only `|L| + aleph_0` many formulas and finite tuples have to be handled at each stage.
