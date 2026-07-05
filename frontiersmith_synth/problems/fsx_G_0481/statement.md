# Conflict-Free Cache Color Selection

## Problem

A set-associative cache is partitioned into `n` candidate *color slots* numbered
`1, 2, ..., n` (a color is a residue class of physical addresses that maps to a fixed
group of cache sets). The allocator wants to hand out as many colors as possible to a
latency-critical arena, but it must avoid a pathological *strided-conflict* pattern:

> if three selected colors `x < y < z` are **evenly spaced** — i.e. `y - x == z - y`
> — then a power-of-two strided sweep can be steered to evict all three in a repeating
> 3-way conflict, defeating the coloring.

So a set of colors is **admissible** iff it contains **no three-term arithmetic
progression** (no `x, x+d, x+2d` all selected, for any `d >= 1`). Note the three colors
need not be adjacent in the selected set — *any* three chosen colors that are evenly
spaced are forbidden.

Your job: select an admissible set of colors that is **as large as possible**.

## Input (stdin)

A single integer `n` (`2 <= n <= 30000`).

## Output (stdout)

The selected colors, as whitespace-separated integers in `[1, n]` (spaces and/or
newlines, in any order). Output **only** these integers — no count header, no other text.
Every value must be distinct and lie in `[1, n]`.

## Feasibility

The output is admissible iff:
- every token is an integer in `[1, n]`,
- all values are distinct,
- the set contains **no** three-term arithmetic progression.

Any violation (out-of-range, duplicate, non-integer, `nan`/`inf`, or a 3-term
progression) scores `0`.

## Objective (maximize)

`F = ` the number of selected colors (`|S|`).

## Scoring

Let `B` be the size of a reference admissible construction that the checker builds
itself (the "base-5 two-digit" set: colors whose base-5 representation uses only the
digits 0 and 1). The score is

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the reference construction scores `Ratio ~= 0.1`; you must build denser
admissible sets to score higher. Reaching `Ratio = 1.0` would require a set ten times
the reference size, which is not known to be attainable.

## Constraints

- `2 <= n <= 30000`.
- Deterministic scoring; exact integer arithmetic only.

## Example

For `n = 9`, the set `{1, 2, 4, 5}` is admissible (its progressions? `1,2,3`? 3 not
chosen; `2,4,6`? no; `1,3,5`? 3 not chosen; no three chosen colors are evenly spaced),
with `F = 4`. The set `{1, 2, 3}` is **not** admissible because `1, 2, 3` is an
arithmetic progression, so it scores `0`.
