# Godel Incompleteness

## The problem it solves

Godel's incompleteness theorems answer a Hilbert-program question: can a strong
formal theory of arithmetic be complete, consistent, effectively axiomatized, and
able to prove its own consistency? The answer is no. Any sufficiently expressive,
effectively axiomatized consistent theory of arithmetic leaves some arithmetical
sentence undecided, and it cannot prove its own standard consistency statement.

## The key idea

The unique insight is the arithmetization of syntax. Godel assigns natural numbers to
symbols, formulas, and proofs, so facts that look metamathematical from the outside
become facts about natural numbers. In particular, the relation
"p is a proof in T of the formula with code n" can be expressed by an arithmetical
formula Proof_T(p, n), and provability becomes:

```text
Prov_T(n) := exists p Proof_T(p, n)
```

That is the fundamental breakthrough: arithmetic is made to talk about its own
formal derivations without leaving arithmetic.

## The self-referential sentence

Once the proof predicate has been internalized, the fixed-point lemma constructs a
sentence G_T such that T proves:

```text
G_T <-> not Prov_T(code(G_T))
```

Informally, G_T says "I am not provable in T." Formally, it says that no natural
number codes a valid T-proof of the formula whose code is code(G_T). If T proved G_T,
then T would prove a sentence asserting the nonexistence of that very proof. Under
the relevant consistency assumptions, this cannot happen. If T also proved its own
standard consistency statement Con(T), then T would be able to prove G_T; hence a
consistent T cannot prove Con(T).

## Why this is not just a paradox

The construction is not the ordinary liar paradox in mathematical clothing. The liar
uses semantic truth: "this sentence is false." Godel avoids a truth predicate and
uses formal provability, which is a syntactic, mechanically checkable relation on
finite strings. The sentence does not generate contradiction; it exposes
incompleteness. A consistent theory is not made inconsistent. Instead, it is shown to
contain enough arithmetic to encode its own proof theory, yet not enough strength to
settle every arithmetical fact about that proof theory.

The conceptual shift is therefore from external metamathematics to internal
arithmetic. The system's syntax is no longer merely an object of outside commentary;
it becomes part of the system's numerical subject matter. That internalization is
what makes the theorem a structural limit on formal systems rather than a clever
restatement of a semantic puzzle.

## Minimal takeaway

Godel's decisive move was:

```text
formal syntax -> numerical coding -> arithmetical proof predicate -> fixed point
```

This pipeline lets arithmetic express "this very arithmetical sentence has no proof
in T." The result is a precise undecidable sentence and, from it, the impossibility of
a sufficiently strong consistent theory proving its own consistency.
