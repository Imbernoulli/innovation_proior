# Karp-Lipton Theorem

## Core Claim

In its standard modern form, the Karp-Lipton theorem says:

`NP subseteq P/poly implies PH = Sigma_2^p`.

Equivalently, if `SAT` has polynomial-size nonuniform circuits, then the polynomial hierarchy collapses to its second level.

## Distinctive Insight

The theorem turns a nonuniform upper-bound assumption into a uniform collapse theorem.

The assumption "NP has small circuits" looks weaker than "NP has polynomial-time algorithms," because the circuit for length `n` need not be constructible. Karp-Lipton shows that this weakness is still too powerful: in a higher-level quantified computation, the small circuit can be existentially guessed as advice, and universal quantification can test it across all relevant inputs.

## Proof Skeleton

Take a `Pi_2^p` statement:

`forall x exists y R(z, x, y)`.

The inner `exists y` predicate can be encoded as a SAT instance. If `SAT` has polynomial-size circuits, then by self-reducibility those circuits can be used to produce satisfying assignments, not merely decide satisfiability.

A `Sigma_2^p` machine can therefore guess one polynomial-size witness-producing circuit `D`, then universally check:

`forall x, R(z, x, D(z, x))`.

If the original statement is true, a correct `D` exists. If it is false, some `x` has no valid witness, so every guessed `D` fails. This places `Pi_2^p` inside `Sigma_2^p`, yielding the collapse of `PH`.

## Why It Is Not Just A Lower Bound

Karp-Lipton does not prove `SAT notin P/poly`. Instead, it explains what would go wrong structurally if `SAT in P/poly` were true.

The theorem says small nonuniform circuits for `NP` would let one guessed advice object replace many existential witnesses spread across universal branches. That would make alternation in the polynomial hierarchy much less expressive than expected.

## Final Takeaway

Karp-Lipton's unique contribution is the pressure argument: small circuits for `NP` are not just a failed lower-bound target, but a condition strong enough to collapse `PH`. The result reframes circuit lower bounds as evidence about the structural incompatibility between nonuniform advice and a robust polynomial hierarchy.
