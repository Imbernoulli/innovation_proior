# PAC Learning - Method Artifact

## Protocol

Represent a concept as a Boolean recognizer over variables `p_1, ..., p_t`, extended to partial vectors in `{0,1,*}^t`: a vector is positive when every total completion makes the target true.

The learner has access to:

- `EXAMPLES`, which returns a positive vector drawn from an arbitrary fixed distribution `D`;
- optionally `ORACLE(v)`, which reports whether `v` positively exemplifies the target.

The learner does not know `D`. Error is measured under that same `D`.

## Learnability Criterion

A concept class is learnable when there is a deduction algorithm that, for every target in the class and every distribution `D`, uses polynomially many examples and polynomial time and outputs, with probability at least `1-delta`, a hypothesis whose distributional error is at most `epsilon`.

For Valiant's main positive classes, the guarantee is one-sided: the hypothesis never accepts outside the target; its only allowed error is rejecting a set of true positives of `D`-mass at most `epsilon`.

## Progress Bound

Let `L(h,S)` be the number of independent trials needed so that, if each trial succeeds with probability at least `1/h`, the probability of fewer than `S` successes is below `1/h`.

Valiant's bound is:

```text
L(h,S) <= 2h(S + ln h).
```

This is the sample-complexity engine: if high current error makes the next random example produce a structural repair with probability at least `1/h`, and at most `S` repairs are possible, then polynomially many examples make high residual error unlikely.

## Core Results

**Bounded CNF.** For fixed `k`, initialize `g` as the conjunction of all clauses with at most `k` literals. For each positive example, delete every clause the example does not satisfy. There are fewer than `(2t)^(k+1)` candidate clauses, so `L(h,(2t)^(k+1))` examples learn `k`-CNF from examples alone. Conjunctions are the `k=1` case.

**Finite-class cross-check.** If `H` is finite and a learner returns a hypothesis consistent with the sample, then

```text
m >= (1/epsilon)(ln |H| + ln(1/delta))
```

suffices for error at most `epsilon` with probability at least `1-delta`. For monotone conjunctions `|H|=2^n`; for general conjunctions `|H|=3^n`.

**Monotone DNF.** Initialize `g=false`. When a positive example is missed by `g`, use `ORACLE` to remove inessential positive coordinates until a prime implicant remains, then add that monomial to `g`. With degree `d`, at most `d` monomials are added, so `L(h,d)` examples and at most `dt` oracle calls suffice.

**Boundary.** Unrestricted DNF over partial vectors is harder because testing whether a partial vector implies a DNF formula includes the tautology problem. This is why the monotone concept result and the total-vector unrestricted-DNF function result are separated.
