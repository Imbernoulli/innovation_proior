# The Line's Diary: Compressing a Grid-Crossing Sequence

## Problem

Draw the straight segment from (0,0) to (P,Q) over the integer grid, where P and Q are
coprime positive integers. As you travel along the segment, record one symbol every time
you cross a grid line: `V` when you cross a vertical line `x = 1, 2, ..., P-1`, `H` when
you cross a horizontal line `y = 1, 2, ..., Q-1`, in the order these crossings occur.
Because gcd(P,Q) = 1, no vertical and horizontal crossing ever coincide, so this gives a
well-defined string `S` over `{H,V}` of exact length `L = P + Q - 2` (the two endpoints
are not counted).

Your job is not to output `S` itself, but a **compact grammar that expands to exactly
`S`**. The prize for finding structure in the slope P/Q is a much smaller description
than storing the crossings one by one.

## Input (stdin)

One line: two integers `P Q` with `2 <= P, Q <= 2000`, `gcd(P,Q) = 1`.

## Output (stdout) -- the artifact

```
R
id_0 sym0_0 c0_0 sym1_0 c1_0
...
id_{R-1} sym0_{R-1} c0_{R-1} sym1_{R-1} c1_{R-1}
ANSWER k
```
- `R` (1 <= R <= 3000): number of grammar rules.
- Each rule line has 5 whitespace-separated tokens: `id sym0 c0 sym1 c1`, where `id` must
  equal the rule's 0-based line index. Each `sym` is either the literal character `H`/`V`,
  or a decimal integer referencing an **earlier** rule id (`0 <= ref < id`) -- no cycles,
  no forward references. Each `c` is a non-negative decimal integer count, `0 <= c <=
  10^7`. Rule `i` expands to `expand(sym0)` repeated `c0` times, followed by
  `expand(sym1)` repeated `c1` times (`expand('H')="H"`, `expand('V')="V"`,
  `expand(ref)=expand(rule[ref])`).
- `ANSWER k` (`0 <= k < R`): the rule whose expansion must equal `S` exactly.

Blank lines are ignored; anything else malformed (wrong token count, bad reference, out-
of-range count, non-ASCII, non-integer field, missing `ANSWER`, `R` out of range) is an
immediate feasibility failure.

## Feasibility

The grammar must be a valid DAG as described above, and `expand(rule[k])` must equal `S`
**character for character** (checked by exact reconstruction, not just length). Any
violation scores `0.0`.

## Objective & Scoring

Define the description cost of your grammar as
`cost = sum over ALL R rules of (2 + digits(c0) + digits(c1))`
(2 fixed structural tokens per rule, plus the decimal length of each repeat count --
references cost nothing extra beyond the fixed 2, so sharing is free but every count you
write down costs roughly its number of digits). Let `F = L / (L + cost)` (bigger `F` =
smaller grammar, relative to the sequence length).

The checker also builds its own naive reference grammar for `S` -- one rule per symbol,
chained left to right (`rule_i = ref(rule_{i-1})*1 + S[i]*1`) -- and computes its `F`
the same way, call it `B`. Your printed score is `min(1.0, 0.1 * F / B)` (so tying the
naive chain exactly gives ~0.10, and a 10x-smaller-than-naive grammar caps the score at
1.0).

Storing `S` literally (or symbol-by-symbol) ties `B` exactly (score ~0.10). Plain run-
length encoding (one rule per maximal run of the same symbol, chained together) helps
when one crossing type dominates in long unbroken stretches -- but it only sees the
*first* level of structure. Look at how the crossing pattern for slope P/Q relates to
the **continued fraction of P/Q**: nothing stops you from representing the sequence at
multiple nested scales, one grammar rule per level, instead of one rule per run or symbol.

## Constraints

- `2 <= P, Q <= 2000`, `gcd(P,Q) = 1`.
- Time limit 5s, memory 512MB, `R <= 3000`, counts `<= 10^7`, output `<= 2MB`.

## Example (worked score, illustrative only -- not a test case)

`P=5, Q=2`: crossings in order are `V V H V V`, so `S = "VVHVV"`, `L = 5`.

A valid (non-optimal) artifact:
```
3
0 V 2 H 0
1 0 1 H 1
2 1 1 V 2
ANSWER 2
```
Rule 0 = `"VV"`, rule 1 = `"VV"+"H" = "VVH"`, rule 2 = `"VVH"+"VV" = "VVHVV"` = `S`. Cost
`= (2+1+1) + (2+1+1) + (2+1+1) = 12`. The checker's own naive chain (5 rules, one per
symbol) costs `20`, giving `B = 5/25 = 0.2` and `F = 5/17 ≈ 0.294`; score `=
min(1.0, 0.1 * 0.294/0.2) ≈ 0.147`.
