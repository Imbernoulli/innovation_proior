# Graffiti Removal: Recovering the Core of a Scrambled Logic Circuit

An archaeologist has uncovered an ancient logic machine. Its original wiring was
small and elegant, but centuries of "restorers" scrawled graffiti over it: harmless
padding, dead loops that never fire, and the same computation copied out in a second,
uglier hand. Your job is to scrape the graffiti off and hand back the smallest machine
that still computes exactly the same thing.

## The machine

A **circuit** over `n` boolean inputs `x0..x_{n-1}` is a straight-line list of gates.
Node ids `0..n-1` are the inputs; each gate has the next id in order. A circuit is:

```
n g
<gate 1>
...
<gate g>
OUTPUT r
```

Each gate is one of (operands must be **earlier** node ids):

```
AND a b     OR a b     XOR a b     NOT a     CONST0     CONST1
```

`OUTPUT r` names the node whose value is the circuit's output (`0 <= r < n+g`).

## Input (stdin)

The scrambled circuit. It computes some hidden boolean function `f: {0,1}^n -> {0,1}`
(the value of its `OUTPUT` node over all `2^n` assignments). `n <= 16`.

## Output (stdout)

**Any** circuit, in the same schema, over the **same** `n` inputs, whose `OUTPUT` node
computes the **same** function `f` on all `2^n` inputs. It may reuse none, some, or a
completely different set of gates — only the function must match. Keep it small.

## Feasibility

Your circuit is accepted only if, for every one of the `2^n` input assignments, its
output equals `f`. Any parse error, out-of-range operand (an operand id `>=` the gate's
own id, or an `OUTPUT` out of range), unknown token, or wrong truth table scores `0`.

## Objective (minimize)

Let `F` be the number of gate lines in your circuit and `B` the number of gate lines in
the scrambled input. Fewer gates is better. The score is

```
score = min(1.0, 0.1 * B / max(1, F))
```

Resubmitting the input verbatim (`F = B`) scores `0.1`; a circuit `10x` smaller caps at
`1.0`. Only the **gate count** is scored — gate type does not matter.

## What the graffiti is (and why it resists local cleanup)

The scramble is built from four independent layers on top of a compact core:

- **Identity padding** — wrappers like `w^0`, `w&w`, `~~w`, `w&1`. Locally removable.
- **Dead logic** — a subcircuit whose value is globally constant `0`, then OR-ed in.
  It is assembled from **distant, structurally different nodes**, so no local `x & ~x`
  rule fires: only knowing each wire's actual behaviour reveals it contributes nothing.
- **Redundant recomputation** — a subfunction of the core, copied in a *different shape*
  and folded back. Two distant nodes compute the same thing; spotting it needs their
  functional signatures, not their text.
- **Structural blow-ups** — De Morgan / XOR re-expressions that genuinely enlarge the
  wiring and are not identities you can pattern-match away.

A purely **syntactic** cleanup (peephole rewrites + merge-identical-gates + drop-unused)
removes the padding and plateaus — the dead logic and the redundant copy look alive and
distinct. The win comes from treating the circuit as a **function, not a text**: compute
each wire's truth table, fold the ones that are constant, merge the ones with an equal
signature however far apart they sit, then drop what is no longer reachable. Even that is
not the true minimum — factoring the core further is left open.

## Example

Input (a 2-input machine, `n=2`, padded):

```
2 4
AND 0 1
CONST0
XOR 2 3
OR 4 4
OUTPUT 5
```

This computes `x0 AND x1`. A valid smaller answer:

```
2 1
AND 0 1
OUTPUT 2
```

Here `B = 4`, `F = 1`, so `score = min(1, 0.1*4/1) = 0.4`.

## Constraints

`n <= 16`; the scrambled input has a few hundred gates; time limit 5s; memory 512 MB.
Scoring is exact and deterministic.
