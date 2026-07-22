# Nonna's Ledger

A commercial kitchen has run the same undocumented batch-prep routine for
decades: a catering order of `n` portions is split into a "stockpot batch"
and a "garnish batch", each of which is prepped by calling the SAME routine
recursively on a smaller portion count, plus some fixed extra knife-and-plate
operations, plus a constant number of operations per portion. Every order the
kitchen has ever run is logged, EXACTLY, in the paper ledger (that's how the
grocery bills get reconciled). Your job: recover the routine well enough to
predict the exact operation count for future mega-orders. The kitchen's new
digital invoicing system REJECTS any prediction that isn't bit-for-bit exact.

## Input (stdin)

```
n_train test_id
0  T(0)
1  T(1)
...
n_train  T(n_train)
```

`test_id` fixes a hidden routine (structurally the same shape for every test,
but with different numeric parameters). The rows give the EXACT operation
count `T(n)` for every order size `0..n_train`. The routine's parameters and
`test_id`'s seed are never printed.

## Output (stdout): a tiny recursive-program DSL

Emit exactly two lines:

```
BASE k v0 v1 ... vk
REC  <expr>
```

`BASE` gives `T(0..k)` directly (`k+1` integers, `0 <= k <= 2000`). `REC` is
an integer arithmetic expression (used for `n > k`) over `+ - *`,
parentheses, unary minus, integer constants (magnitude `<= 10^9`), the
variable `n`, and:

- `MOD(n, m)` — `n mod m` (constant `2 <= m <= 30`).
- `TAB(MOD(n, m), v0, ..., v_{m-1})` — looks up `v_{n mod m}` (`m` values,
  matching the modulus).
- `T(FLOORDIV(n, k))` / `T(CEILDIV(n, k))` — a RECURSIVE call to your own
  program, evaluated at `floor(n/k)` or `ceil(n/k)` (constant `2 <= k <= 12`).

Hard limits (feasibility): `REC` is at most 80 AST nodes; the whole program
is at most 300000 bytes.

The grader evaluates `T(n)` for held-out sizes by rolling out your program
(memoized): if `n <= k` it reads `BASE`; otherwise it evaluates `REC`,
recursing through any `T(...)` calls. Program size `<= 80` expression nodes.

**Illustrative FORM only — NOT the hidden routine:**
```
BASE 0 3
REC  TAB ( MOD ( n , 4 ) , 1 , -1 , 2 , 0 ) + 5 * n
```
This just shows the syntax (no recursion at all here); the real routine
recurses and you must discover its shape from the ledger.

## Feasibility

The program must parse under the grammar above (known names/functions only,
integer constants, in-range moduli/divisors, size within bounds, `BASE`'s
value count matching its declared `k`). Any violation scores `0`.

## Objective (maximize)

The grader regenerates a held-out sample of large "mega-order" sizes (far
beyond `n_train` — genuine extrapolation, never shown to you) and their TRUE
operation counts, rolls your program forward on them, and forms:

```
EXACT     = fraction of held-out sizes matched EXACTLY (integer equality)
CLOSE     = mean of 1 / (1 + 7 * |pred - true| / max(1,|true|))  over held-out sizes
PARSIMONY = max(0, 1 - footprint / 60), footprint = (BASE value count) + (REC node count)
Ratio = 0.04 + 0.55 * EXACT + 0.12 * CLOSE + 0.08 * PARSIMONY     (clipped to [0,1])
```

A prediction that's merely in the right ballpark earns some `CLOSE` credit;
only genuinely EXACT recovery earns the large `EXACT` term. `PARSIMONY`
rewards a leaner encoding of whatever law you emit (a smaller `BASE` table
and a smaller `REC` expression), so two programs that both recover the exact
law can still be told apart. A point where your program hits a cycle,
exceeds the recursion-depth guard, or otherwise fails to evaluate (including
producing a value too large to score) counts as a miss for `EXACT`/`CLOSE`
(not a global failure).

## Why "close" isn't good enough

On the training range, the operation count looks almost perfectly linear in
`n`, so a straight-line (or straight-line-plus-periodic-correction) fit looks
excellent. But `T(n)` is truly defined via `T` at smaller arguments — every
recursive call injects its own rounding and remainder correction. The total
un-modeled correction mass GROWS with `n` (it does not stay a small constant
"noise" term), so smooth fits drift further and further from exact as order
sizes grow, while the ledger's true generative law never does.

## Example (toy, illustrative numbers only)

Suppose (hypothetically) a held-out sample had 4 sizes with true counts
`[100, 250, 900, 4000]` and your program predicted `[100, 251, 900, 3000]`,
using a program with footprint 30 (e.g. `BASE 1 ...` plus a 28-node `REC`).
Then `EXACT = 2/4 = 0.5` (sizes 1 and 3 matched exactly); for the misses,
relative errors are `1/250=0.004` and `1000/4000=0.25`, giving
`CLOSE = (1 + 1/(1+7*0.004) + 1 + 1/(1+7*0.25))/4 ≈ 0.865`;
`PARSIMONY = 1 - 30/60 = 0.5`; so
`Ratio ≈ 0.04 + 0.55*0.5 + 0.12*0.865 + 0.08*0.5 ≈ 0.460`.

## Constraints

Time limit 5 s, memory 512 MB. `n_train` is a few hundred rows; held-out
sizes reach up to 2,000,000. Scoring is fully deterministic (all randomness
is seeded from `test_id`).
