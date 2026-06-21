## Research question

The question is how first-order logical proof can be made genuinely mechanical. The hard part is not only checking a finished proof, but searching for one: a theorem prover must decide which formulas to combine, which terms to instantiate, and when a contradiction has been reached.

The central question is whether first-order reasoning can be reduced to a uniform executable calculus rather than a collection of many human-facing inference patterns.

## Background

In first-order logic, formulas contain predicates, variables, quantifiers, functions, and constants. Traditional proof systems such as Hilbert calculi, natural deduction, or sequent calculi are expressive and mathematically elegant, with many inference rules each corresponding to a natural reasoning pattern.

Resolution works by proving a goal through contradiction. Put the axioms together with the negation of the goal; transform the result into a set of clauses; then try to derive the empty clause. The empty clause represents contradiction, so its derivation refutes the negated goal.

The first-order obstacle is variables. A clause such as `P(x) or Q(x)` can interact with `not P(a)`, but only after finding the substitution `x := a`. The question is how to make that matching step algorithmic — how to compute substitutions that allow clauses with variables to interact without enumerating all possible ground instances in advance.

## Baselines

Earlier resolution-related procedures could rely on propositional reasoning or on ground instances produced from first-order formulas, following Herbrand-style ideas.

Human-oriented proof systems such as Hilbert calculi and natural deduction have rich expressive rule sets designed for mathematical elegance and human readability.

Tableaux, model elimination, and other calculi offer alternative automated proof methods with varying degrees of compression and mechanizability.

## Core mechanism

The input is converted into clausal form: implications are eliminated, negations are pushed inward, variables are standardized apart, existential quantifiers are Skolemized, and the formula is represented as a conjunction of disjunctions of literals. Universal quantifiers become implicit in clauses.

The resolution step takes two clauses containing complementary literals that can be made identical by a substitution. If `L` and `not L'` have a most general unifier `theta`, then from `C or L` and `D or not L'` the prover derives `(C or D)theta`. Factoring can also merge unifiable literals within a clause.

Search then becomes saturation or refutation search. The prover repeatedly chooses clauses, computes unifiers for complementary literals, adds resolvents, and stops successfully when the empty clause is derived. For first-order logic this is a semi-decision procedure: if the theorem is valid, fair search can eventually find a refutation, but non-theorems may lead to nontermination.

## Stakes for AI

Automated theorem proving for AI requires that a system handle domain knowledge represented in logic and derive consequences without human guidance. The goal is to replace domain-specific syllogisms and hand-coded inference chains with a general, machine-executable deduction procedure.

Unification computes substitutions that make symbolic terms match, and most general unifiers are the least-committing such substitutions. This machinery underlies automated reasoning and logic programming.

A machine performing first-order deduction by manipulating syntax under a complete calculus can leave performance to heuristics, indexing, clause selection, and refinements such as paramodulation and superposition for equality.
