# Kleene Recursion Theorem

## Core statement

Fix an acceptable effective numbering of partial computable functions, and write `phi_e` for the function computed by index `e`.

Kleene's second recursion theorem says:

```text
For every partial computable Q(e, x), there is an index p such that
phi_p(x) ~= Q(p, x) for every x.
```

Rogers' fixed-point form says:

```text
For every total computable transformer f on program indices,
there is an index p such that phi_p ~= phi_{f(p)}.
```

Both statements formalize the same idea: an effective system of programs contains programs that can use their own descriptions as parameters.

## Key idea

The theorem's distinctive move is to separate self-reference from paradox. A program does not need a primitive "current source code" operation. The construction uses two ordinary computable facts:

- Programs have effective descriptions, usually natural-number indices.
- The `s-m-n` theorem gives an effective specialization operation `s(q, a)` that turns a two-input program `q` into a one-input program with `a` hard-coded.

Given a desired behavior `Q(self, x)`, build a helper:

```text
B(q, x) = Q(s(q, q), x)
```

Let `b` be an index for `B`, and set:

```text
p = s(b, b)
```

Then:

```text
phi_p(x)
~= B(b, x)
~= Q(s(b, b), x)
~= Q(p, x)
```

So `p` is a concrete program index whose behavior is exactly the behavior of `Q` when `Q` is handed `p` itself.

## Why this is not a paradox

The construction is staged and effective. First, define `B`. Second, take an index `b` for `B`. Third, compute `p = s(b, b)`. Only after this finite construction does the resulting program run. When it runs, it can use `p` because the construction placed `p` there through specialization.

There is no inconsistent truth condition, no halting oracle, and no assumption that semantic equality is decidable. The theorem proves an equality of partial functions by unfolding how the produced code was assembled. Self-reference enters as a computable fixed point of a program-description transformer.

## Consequences

Quines are a special case: choose `Q(e, x)` to output a representation of `e`. Recursive definitions are another case: choose `Q(e, x)` to call the function named by `e` on smaller arguments. Fixed points of program transformers follow by taking `Q(e, x) = phi_{f(e)}(x)`.

The broad lesson is that self-reference is not an informal trick added to computation from the outside. In any acceptable programming system with effective numbering and specialization, there is a general mechanism for constructing programs whose own descriptions are available as data.
