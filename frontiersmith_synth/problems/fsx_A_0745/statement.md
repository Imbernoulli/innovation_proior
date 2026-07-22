# Multi-Scale Grammar Compression

## Problem
You are given a string `S` of length `n` over the digit alphabet `{0,...,7}`. It was built by
nesting short repeated fragments across several structural scales: small "atoms" are concatenated
into mid-size blocks, blocks are concatenated into sections, and sections (with occasional local
point-mutations sprinkled in as noise) are concatenated many times to form `S`. You do not know
this recipe -- you only see `S`.

Your job: output a **straight-line grammar** -- a numbered list of rules, each a short sequence of
symbols -- such that the LAST rule expands (by recursively substituting references) to exactly
`S`. Rules are indexed `0..R-1` and must be **acyclic**: a rule may reference only *strictly
earlier* rules, so the grammar is automatically a DAG. Your artifact is scored by how cheaply it
can be encoded under a fixed bit-cost model that charges *more* for referencing an existing rule
as the rule table grows (see Scoring). Reusing one rule inside many different larger rules --
genuine hierarchical factoring -- is what actually pays off; naming only the single biggest
literal repeat you can spot is not enough once noise breaks up the largest exact matches.

## Input (stdin)
```
n
S
```
`S` is exactly `n` characters, each a digit `0`-`7`.

## Output (stdout)
```
R
rhs_0
rhs_1
...
rhs_{R-1}
```
`R` is the number of rules. Each `rhs_i` is a space-separated list of 1 or more tokens. A token is
either a single digit `0`-`7` (a **terminal**), or `r<j>` for an integer `j` with `0 <= j < i` (a
**reference** to the earlier rule `j`). Rule `R-1` is the START rule.

## Feasibility
The output is valid iff ALL hold:
- `1 <= R <= 8n + 2000`; every `rhs_i` has between 1 and 13000 tokens (inclusive);
- every reference `r<j>` satisfies `0 <= j < i` (strictly earlier rule -- acyclic);
- every terminal token is a single digit `0`-`7`;
- expanding rule `i` (substitute each reference by the already-computed expansion it names, then
  concatenate) never produces a string longer than `n` for ANY rule, and the TOTAL characters
  produced while expanding all rules never exceeds `60n + 20000` (a finite compression-work
  budget -- also caps how far you may over-provision unused rules);
- the expansion of rule `R-1` equals `S` exactly (same length, same characters).
Any violation scores `Ratio: 0.0`.

## Objective
Minimize the encoded size `F`, in bits, of your grammar under this fixed cost model:
a `16`-bit fixed header, plus, for every rule `i` with token count `L`: a length field costing
`len_cost(L) = 2*floor(log2(L)) + 1` bits (a cheap universal code -- short rules pay very little,
long ones pay proportionally more), plus, for each token in `rhs_i`: **3 bits** if it is a terminal
(`ceil(log2(8))`), or **`max(1, ceil(log2(i)))` bits** if it is a reference `r<j>` (`i` = how many
rules existed *before* this one was defined, i.e. the number of distinct earlier rules you could
have referenced -- so a rule defined late in the table pays more per reference than one defined
early).

## Scoring
Let `B = 16 + len_cost(n) + 3n` be the checker's own baseline: the cost of the trivial one-rule
grammar that just lists `S` as `n` literal terminals, with no reuse at all. With `F` your grammar's
cost:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the literal baseline scores `Ratio = 0.1`; a grammar `10x` cheaper caps at `1.0`.

## Constraints
`n <= 12000`. Time limit 5s, memory 256m.

## Example
`n = 64`, `S` = `"01"` repeated 32 times. `len_cost(64) = 13`, so the literal baseline costs
`B = 16+13+64*3 = 221` bits (`Ratio = 0.1`). A doubling grammar -- `rhs_0="0 1"`, `rhs_1="r0 r0"`,
`rhs_2="r1 r1"`, `rhs_3="r2 r2"`, `rhs_4="r3 r3"`, `rhs_5="r4 r4"` (start) -- expands `r0` to 2
characters, ..., `r5` to all 64, reconstructing `S` exactly. Every rule has `L=2` tokens
(`len_cost=3`). Rule 0 is two terminals: cost `3+2*3=9`. Rules 1-5 each hold two references, with
`ref_cost(i)` for `i=1..5` equal to `1,1,2,2,3`, giving rule costs `5,5,7,7,9`. Total rule cost
`9+5+5+7+7+9=42`, plus the `16`-bit header: `F=58`. `sc = 100*221/58 = 381.0`, `Ratio = 0.3810` --
almost four times the baseline's score, from noticing the string doubles in size at every scale
instead of copying `64` literal digits.
