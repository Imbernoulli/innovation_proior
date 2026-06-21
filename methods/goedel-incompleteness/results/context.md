# Context

## Research question

Can a sufficiently strong, effectively axiomatized formal system for arithmetic be
both complete and able to certify its own reliability from within? In the Hilbert
program setting, the hope was that formal mathematics could be made precise enough
that every arithmetical truth of the intended domain would be derivable, while the
system's consistency could be proved by finitary means. Godel's incompleteness
theorems show that this hope breaks at a precise point: once a system can represent
enough elementary arithmetic and its axioms are effectively recognizable, there are
arithmetical statements the system cannot decide, and the system cannot prove its own
consistency if it is in fact consistent.

The distinctive question is not merely whether self-reference can cause trouble.
The real question is how a formal theory, whose formulas appear to talk only about
natural numbers, can be made to talk about formulas, derivations, and provability
inside that same numerical language. Godel's breakthrough is the arithmetization of
syntax: proofs become finite numerical objects, and metamathematical predicates
about those proofs become arithmetical predicates.

## Background

Before the incompleteness theorem, metamathematics seemed external to the formal
systems it studied. A logician could stand outside a calculus and say that a string
is a formula, that one string follows from earlier strings by a rule, or that a
formula has a proof. Those are syntactic facts about finite inscriptions, not
arithmetical facts on their face. Godel's move was to assign natural numbers to
symbols, sequences of symbols, formulas, and whole proofs. With a sufficiently
careful coding, basic syntactic operations become primitive recursive numerical
relations.

This matters because theories such as Peano Arithmetic can represent primitive
recursive relations. Therefore the external statement "p is a proof, in T, of the
formula with code n" can be mirrored by a formula Proof_T(p, n) of arithmetic, and
"there exists a T-proof of n" can be mirrored as Prov_T(n). The system has not gained
a magical semantic truth predicate. It has gained something more disciplined and more
powerful: an internal arithmetical proxy for its own formal provability relation.

## Baselines

The first baseline is the liar-style paradox: "this sentence is false." It uses
semantic truth and informal self-reference, and it yields contradiction if treated
naively. Godel's construction is different. It avoids a truth predicate and talks
about formal proof, a syntactic relation on finite strings encoded as numbers. The
outcome is not explosion but an undecidable sentence, under the relevant consistency
conditions.

The second baseline is Hilbert-style consistency proof from the outside. One can
inspect a formal calculus and try to prove, in a trusted metatheory, that it never
derives a contradiction. Godel internalizes enough of that inspection into arithmetic
itself. The surprise is that the internal version cannot be completed by the very
system under inspection, unless the system is inconsistent.

The third baseline is ordinary diagonalization, such as Cantor's anti-diagonal. Godel
uses a related fixed-point pattern, but the object being diagonalized is not merely a
list of reals or sets. It is the theory's own proof predicate, represented inside
arithmetic. The diagonal step produces a sentence G_T whose content is, in effect,
"G_T is not provable in T."

## Evaluation settings

The setting is any formal theory T that is effectively axiomatized, consistent in the
relevant sense, and strong enough to represent enough elementary arithmetic. The
original 1931 theorem used conditions such as omega-consistency for the full
undecidability claim; later refinements weaken or separate these assumptions. For
the conceptual point here, the essential ingredients are effective proof checking,
representability of primitive recursive syntax, and a diagonal or fixed-point lemma.

Success is measured by whether the argument produces a concrete arithmetical sentence
that the theory cannot decide, not by whether it creates a semantic paradox. The
sentence must be a formula of arithmetic, its code must be a natural number, and the
claim about its unprovability must be expressed through the theory's formal proof
predicate. The second theorem is evaluated by whether the theory can prove the
arithmetized statement Con(T), usually formulated as the nonexistence of a T-proof of
contradiction.

## Code framework

```text
# External syntax:
#   formula, proof, derivation, substitution, theoremhood

# Godel coding:
#   each symbol/string/formula/proof -> a natural number

# Primitive recursive syntax predicates:
#   Formula(n)        : n codes a well-formed formula
#   Proof_T(p, n)     : p codes a valid T-proof of formula-code n
#   Prov_T(n)         : exists p Proof_T(p, n)

# Representability:
#   if a syntactic relation is primitive recursive,
#   arithmetic can express it by a formula.

# Fixed point:
#   for a formula A(x), construct sentence S such that
#   T proves S <-> A(code(S)).

# Incompleteness instance:
#   choose A(x) = not Prov_T(x)
#   obtain G_T such that T proves G_T <-> not Prov_T(code(G_T)).
#
# If T proved G_T, then T would prove a sentence asserting its own nonprovability.
# Under consistency assumptions this cannot happen.
# If T could prove enough of Con(T), it would prove G_T, so a consistent T cannot
# prove its own Con(T).
```
