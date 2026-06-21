# Context

## Research question

Can a sufficiently strong, effectively axiomatized formal system for arithmetic be
both complete and able to certify its own reliability from within? In the Hilbert
program setting, the aim is to make formal mathematics precise enough that every
arithmetical truth of the intended domain is derivable, while the system's
consistency is established by finitary means.

A connected question concerns how metamathematics relates to the object theory. The
formulas of a theory such as Peano Arithmetic appear to talk only about natural
numbers. Statements about those formulas — that a string is a well-formed formula,
that one formula follows from earlier ones by a rule, that a formula has a proof —
are made in a separate metalanguage about finite inscriptions. The setting in which
these questions are posed is the relationship between a formal calculus and the
finitary reasoning carried out about it.

## Background

Metamathematics is conducted external to the formal systems it studies. A logician
stands outside a calculus and states that a string is a formula, that one string
follows from earlier strings by a rule of inference, or that a formula has a proof.
These are syntactic facts about finite sequences of symbols. They are checked
mechanically: given a candidate derivation, one inspects each step against the
finitely many axioms and rules and confirms it is valid.

Theories such as Peano Arithmetic are formulated with a fixed alphabet of symbols, a
recursive set of axioms, and finitely many rules of inference. Such theories can
express and prove a wide range of statements about natural numbers, including
statements built from the basic operations of arithmetic and from bounded and
unbounded quantification. The classes of numerical relations definable in arithmetic,
and in particular those built up by the schema of primitive recursion, are an active
subject of study.

## Baselines

The first baseline is the liar-style paradox: "this sentence is false." It uses
semantic truth and informal self-reference, and treated naively it yields a
contradiction. It operates at the level of meaning and truth rather than at the level
of formal proof.

The second baseline is the Hilbert-style consistency proof from the outside. One
inspects a formal calculus and tries to prove, in a trusted finitary metatheory, that
the calculus never derives a contradiction. The reasoning is carried on in the
metalanguage, about the strings and derivations of the object theory.

The third baseline is diagonalization, as in Cantor's anti-diagonal argument. Given an
enumeration of objects of a certain kind — for instance an enumeration of reals or of
sets — one constructs a new object that differs from the n-th listed object in its
n-th component, so it cannot appear anywhere in the list. The construction proceeds by
indexing a family of objects against the same index set.

## Evaluation settings

The setting is any formal theory T that is effectively axiomatized, consistent in the
relevant sense, and strong enough to express and prove a substantial fragment of
elementary arithmetic. Relevant consistency notions include plain consistency and
omega-consistency. The available ingredients are: effective, mechanical checking of
whether a derivation is a valid proof; the definability within arithmetic of various
classes of numerical relations, including those given by primitive recursion; and the
diagonal pattern of construction.

A statement about the theory is taken to bear on it when it can be phrased as a claim
about the theory's formulas, derivations, and proofs. The second part of the setting
concerns the theory's own consistency, expressed as a statement about whether the
theory derives a contradiction.

## Code framework

```text
# External syntax (stated in the metalanguage):
#   formula, proof, derivation, substitution, theoremhood

# Mechanical proof checking:
#   given a candidate derivation, check each step against
#   the axioms and rules; accept iff every step is valid.

# Numerical relations available in arithmetic:
#   relations built by primitive recursion and by bounded/unbounded
#   quantification can be expressed by formulas of arithmetic.

# Diagonal pattern (e.g. Cantor):
#   given a family indexed by the same set it ranges over,
#   construct an element differing from each indexed member.

# Consistency of T:
#   T is consistent iff it does not derive a contradiction.
```
