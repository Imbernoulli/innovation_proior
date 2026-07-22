# Ripple Window: Minimum Tile Types for a Counting Rectangle

## Problem

You are designing an **abstract Tile Assembly Model (aTAM)** system at
temperature tau = 2. A tile type is a unit square with a **glue** on each of
its four sides (N/S/E/W); a glue is either "none" or a `(label, strength)`
pair with strength 1 or 2. Adjacent tiles bind on a shared edge iff the
facing glues have the **same label**, contributing that glue's strength.
Growth starts from one pre-placed **seed** tile and repeatedly attaches an
empty cell to any declared tile type whose **total binding strength**
(summed over already-occupied N/S/E/W neighbors whose facing glue matches)
is **>= 2**. One strength-2 glue to a single neighbor is enough; two
different neighbors each contributing a matching strength-1 glue is
*cooperative* binding and also works.

Index grid cells by `(row, col)`: `row` = a bit index `0..W-1` (0 = least
significant), `col` = a counter value `0..n-1`. `n` is given in the input; `W`
is the number of bits needed to write `n-1` in binary (`W = 1` if `n <= 2`,
otherwise the smallest `W` with `2^W >= n`). The seed is always placed at
`(row=0, col=0)`. The **target**: every cell `(row=r, col=c)` with `0<=r<W`,
`0<=c<n` must end up occupied by a tile type whose declared **value** equals
`(c >> r) & 1` -- i.e. reading column `c` bottom-to-top spells out `c` in
binary. (Growth may spill outside this window; only the window itself is
checked.)

## Input (stdin)

One line: the integer `n` (`2 <= n <= 16`).

## Output (stdout)

```
K
id  Nlabel Nstr  Slabel Sstr  Elabel Estr  Wlabel Wstr  value
...                                                          (K lines total)
seed_id
```

`K` is how many tile types you declare, `1 <= K <= 2000`. Each of the next `K`
lines defines tile type `id` (the `K` ids must be exactly `0..K-1`, any
order): four `label str` pairs for N, S, E, W, then `value` (0 or 1) -- the
bit this tile type represents wherever it is placed. `label` is letters,
digits, underscore (max 32 chars) or the sentinel `.` ("no glue here"),
which must pair with `str = 0`; a real label must pair with `str` in
`{1, 2}`, and every occurrence of the same label text anywhere in your tile
set must use the same strength. The last line, `seed_id`, names the tile
type pre-placed at `(0,0)`; its value must be 0.

## Feasibility

Your submission is rejected (score 0) if: the output is malformed; a label
text is used with two different strengths; the seed's value isn't 0; the
target window is wrong (some `(row,col)`, `0<=row<W, 0<=col<n`, ends up
unoccupied or holds a value other than `(c >> r) & 1`, after simulating
growth from the seed over a generous neighborhood around the window); or
your system isn't **directed**: two declared types must never be
simultaneously attachable at one empty cell under any valid attachment
order, not just the order the checker happens to grow first -- a competing
type only counts if its matching neighbors are reachable *independently* of
that cell (if they could only ever exist because the cell was already
placed, deferring it can't conjure them into existing sooner).

## Objective

Minimize `K`, the number of declared tile types.

## Scoring

The checker builds its own baseline: `B = n * W` (one bespoke, unshared tile
type per target cell -- always trivially valid). Your score is
```
Ratio = min(1.0, n * W / (10 * K))
```
so matching the per-cell baseline scores 0.1, and every genuine reduction in
`K` raises your score, capped below 1.0 to leave headroom. Your program is run
independently on 10 hidden instances (`n` growing across the set); your final
score is the mean ratio.

## Constraints

`2 <= n <= 16` (so `W <= 4`), `1 <= K <= 2000` declared tile types, time
limit 5s, memory 512MB. Your program is invoked independently once per
hidden instance and must work for any `n` in range -- it does not know in
advance which of the 10 instances it is given.

## Worked example (illustrative form only)

For `n=2` (`W=1`, target: `(0,0)=0`, `(0,1)=1`):
```
2
0 . 0 . 0 g 2 . 0 0
1 . 0 . 0 . 0 g 2 1
1
```
Tile 0 (seed, value 0) offers glue `g` (strength 2) East; tile 1 (value 1)
requires `g` (strength 2) West -- a single non-cooperative strength-2 match.
`K=2`, `B=n*W=2`, so `Ratio = min(1, 2/20) = 0.1`. Larger `n` (hence `W`)
let cooperative, glue-reusing designs pay off far more than this tiny
example can show.
