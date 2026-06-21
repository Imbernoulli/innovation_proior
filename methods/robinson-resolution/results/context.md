## Research question

Robinson resolution asks how first-order logical proof can be made genuinely mechanical. The hard part is not only checking a finished proof, but searching for one: a theorem prover must decide which formulas to combine, which terms to instantiate, and when a contradiction has been reached.

The central question is whether first-order reasoning can be reduced to a uniform executable calculus rather than a collection of many human-facing inference patterns. Robinson's answer was to move the problem into clause form and use resolution, guided by unification, as a single refutation rule.

The distinctive claim is that "smart" logical reasoning can be represented as systematic search over symbolic clauses. The intelligence is shifted out of bespoke proof steps and into normalization, unifier computation, and search control.

## Background

In first-order logic, formulas contain predicates, variables, quantifiers, functions, and constants. Traditional proof systems such as Hilbert calculi, natural deduction, or sequent calculi are expressive and mathematically elegant, but they expose many inference choices to a proof search procedure.

Resolution works by proving a goal through contradiction. Put the axioms together with the negation of the goal; transform the result into a set of clauses; then try to derive the empty clause. The empty clause represents contradiction, so its derivation refutes the negated goal.

The first-order obstacle is variables. A clause such as `P(x) or Q(x)` can interact with `not P(a)`, but only after finding the substitution `x := a`. Robinson's contribution was to make that matching step algorithmic through unification, especially by using most general unifiers instead of enumerating all ground instances in advance.

## Baselines

Earlier resolution-related procedures could rely on propositional reasoning or on ground instances produced from first-order formulas. That approach is complete in principle through Herbrand-style ideas, but it creates a massive combinatorial blow-up because the search space is filled with many irrelevant instantiations.

Human-oriented proof systems can derive the same results, but their rule sets are not directly optimized for automatic search. A prover using many rules has to choose not only the next formula, but also the proof style and the instantiation pattern.

Tableaux, model elimination, and other calculi offer alternative automated proof methods. Robinson resolution's special force is its extreme compression: after clause conversion, the core first-order inference step is resolution plus unification.

## Core mechanism

The input is converted into clausal form: implications are eliminated, negations are pushed inward, variables are standardized apart, existential quantifiers are Skolemized, and the formula is represented as a conjunction of disjunctions of literals. Universal quantifiers become implicit in clauses.

The resolution step takes two clauses containing complementary literals that can be made identical by a substitution. If `L` and `not L'` have a most general unifier `theta`, then from `C or L` and `D or not L'` the prover derives `(C or D)theta`. Factoring can also merge unifiable literals within a clause.

Search then becomes saturation or refutation search. The prover repeatedly chooses clauses, computes unifiers for complementary literals, adds resolvents, and stops successfully when the empty clause is derived. For first-order logic this is a semi-decision procedure: if the theorem is valid, fair search can eventually find a refutation, but non-theorems may lead to nontermination.

## Stakes for AI

Robinson resolution was a breakthrough for AI because it converted logical intelligence into a concrete symbolic engine. Instead of encoding many domain-specific syllogisms, the system could normalize knowledge into clauses and run one general inference loop.

Unification is the key to that shift. It lets the prover instantiate variables only when a proof step demands it, and at the most general useful level. This is why resolution avoided much of the blind ground-instance explosion that made earlier first-order search impractical.

The result shaped automated theorem proving, logic programming, and symbolic AI. It showed that a machine could perform nontrivial deductive reasoning by manipulating syntax under a complete calculus, while leaving performance to heuristics, indexing, clause selection, and later refinements such as paramodulation and superposition for equality.
