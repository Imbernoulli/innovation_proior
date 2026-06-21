# Robinson Resolution

## Core insight

Robinson resolution's distinctive insight is that first-order logical proof search can be reduced to refutation search over clauses using one general inference rule: resolution, made first-order by unification.

The method takes axioms plus the negation of a goal, converts them into clause form, and searches for the empty clause. Each proof step resolves two clauses by finding complementary literals. If those literals are not syntactically identical, unification computes a substitution that makes them match; the resolvent is then the remaining literals under that substitution.

In schematic form:

`C or L`

`D or not L'`

If `theta` is a most general unifier of `L` and `L'`, derive:

`(C or D)theta`

Deriving the empty clause means the original clause set is unsatisfiable, so the negated goal is impossible and the goal follows.

## Why it changed automated reasoning

Before this idea, first-order theorem proving could rely on many human-oriented rules or on generating large numbers of ground instances. Both approaches made search hard to mechanize cleanly. Robinson's use of most general unifiers avoided blind instantiation: the prover instantiated variables only when an actual inference required it.

That turned "clever reasoning" into an executable symbolic calculus. A theorem prover could normalize knowledge into clauses, repeatedly apply one inference schema, and use search strategy rather than human proof style to find derivations.

This was a central breakthrough for AI and automated theorem proving because it gave symbolic reasoning a general engine. Modus ponens, syllogistic chains, case reasoning, and quantifier instantiation could all appear as special cases of the same resolution process after clause conversion.

## Importance and limits

Resolution with unification is refutation-complete for first-order logic in the intended clause setting: if a first-order clause set is unsatisfiable, a suitable fair resolution search can derive contradiction.

It is not a magic decision procedure. First-order validity is semi-decidable, so search may not terminate for non-theorems. Clause growth can be explosive, and equality reasoning requires extensions such as paramodulation or superposition. Practical success depends on indexing, simplification, clause selection, and heuristics.

The conceptual achievement remains sharp: Robinson showed how to represent first-order deduction as a uniform machine procedure. AI did not need to hand-code every kind of logical cleverness; it could run a complete syntactic calculus and spend its intelligence budget on search control.
