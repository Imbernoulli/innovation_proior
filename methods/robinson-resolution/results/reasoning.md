The useful way to understand Robinson resolution is to start from the mismatch between ordinary logic and machine search. First-order logic has a clean semantics, but a human proof can use a large repertoire of transformations, lemmas, instantiations, and tactical choices. A machine needs a small set of operations that can be applied repeatedly without understanding the proof at a human level.

A direct search in a natural deduction or sequent system exposes too many choices. The prover has to choose which connective rule to apply, which quantifier instance to use, which intermediate lemma might help, and how to combine facts. That is not impossible, but it makes "proof search" look like an imitation of human cleverness.

Robinson's move was to change the representation before doing the search. If the goal is to prove `G` from axioms `A`, combine `A` with `not G` and ask whether the resulting set is unsatisfiable. This turns theorem proving into refutation. The prover no longer has to build a direct proof of the desired theorem; it has to drive the negated problem to contradiction.

The second move is clause form. Once formulas are converted into clauses, much of the logical surface syntax disappears. Implications, nested connectives, and explicit universal quantifiers are removed or normalized. Existential quantifiers are replaced by Skolem functions so that witness choices become symbolic terms. What remains is a set of disjunctions of literals.

At that point a very simple propositional idea becomes visible. If one clause says `A or L` and another says `B or not L`, then the two clauses together support `A or B`. The complementary literal has been eliminated. Repeating this elimination can eventually yield the empty clause, meaning that the original assumptions cannot all be satisfied.

The first-order version is harder because the complementary literals may not be textually identical. `P(x)` and `not P(a)` are complementary only after substituting `a` for `x`. More complex examples require matching structured terms such as `f(y)` and `f(g(z))`. This is where unification supplies the missing operation.

Unification computes a substitution that makes symbolic expressions identical, when such a substitution exists. The most general unifier matters because it postpones unnecessary commitments. If a prover can use `x := y` instead of prematurely choosing `x := a`, it keeps more future proof paths available while still making the current inference valid.

This is the crucial algorithmic insight. Earlier Herbrand-style approaches could preserve completeness by considering ground instances, but that means instantiating formulas across a potentially enormous universe of terms before knowing which instances matter. Robinson's calculus performs instantiation on demand, at the moment two literals are selected for resolution.

So the resolution inference is not merely a rule of logic. It is a package: put formulas into clauses, standardize variables so separate clauses do not accidentally share names, find complementary literals, compute their most general unifier, apply that substitution to the remaining literals, and add the resolvent back to the clause set.

The proof procedure is then a search loop. Given a clause set, select pairs of clauses, resolve them where possible, optionally factor clauses to merge unifiable literals, and keep deriving consequences. If the empty clause appears, the original negated goal is refuted. Completeness says that for an unsatisfiable first-order clause set, a fair enough resolution search can find such a refutation.

This completeness result is what makes the method more than a heuristic. Resolution does not just imitate some examples of syllogistic reasoning; it gives a general refutation calculus for first-order logic. The prover can miss a proof because of bad strategy or resource limits, but the calculus itself is adequate for the valid first-order consequences it targets.

The AI significance follows from this reduction. A system that reasons with first-order knowledge no longer needs a separate implementation of every familiar inference pattern. Modus ponens, case analysis, chained syllogisms, and many quantifier instantiations appear as consequences of the same resolution-and-unification loop after clause conversion.

That does not mean proof search became easy. First-order validity remains only semi-decidable, and search can diverge when no proof exists. Clause sets can grow rapidly, and equality needs stronger machinery such as paramodulation or superposition. Practical theorem provers still depend heavily on heuristics, term indexing, simplification, and selection strategies.

The breakthrough is therefore more precise than "Robinson made theorem proving automatic." He identified a representation and a rule that made automatic theorem proving programmable in a uniform way. The hard choices became engineering choices inside a single calculus: which clauses to keep, which inferences to prioritize, how to index terms, and how to control saturation.

This also explains the connection to logic programming. In restricted forms such as Horn clauses, resolution becomes goal-directed; SLD-resolution and Prolog inherit the same idea that computation can be viewed as proof search driven by unification. The same mechanism that proves theorems can also execute logical specifications.

The unique insight, then, is the unification of two senses of "unification." Technically, unification matches first-order terms. Conceptually, Robinson unified first-order proof search itself: diverse reasoning steps were collapsed into a clause-level refutation calculus with one core inference rule.
