# Temple Roof That Drinks the Rain: Ridgeline Drainage

## Problem
A temple roof is an `R x C` grid of tiles, each with an integer height. Some
tiles are fixed stone **pillars** (obstacles): their height can never change
and rain cannot sit on or cross them. Every tile on the outer border of the
grid that is marked as a **gutter** collects rain that reaches it; other
border tiles are just ordinary (non-draining) eaves.

You may re-tile the roof: choose a new height for every non-pillar tile,
subject to two budgets:
- **Edit budget**: the total absolute change, summed over all non-pillar
  tiles, must be at most `B`.
- **Max-slope**: in your FINAL heightfield, any two orthogonally adjacent
  non-pillar tiles must differ in height by at most `S`.

After you re-tile, one raindrop is dropped on every non-pillar tile. Each
drop independently rolls downhill by **steepest descent**: from its current
tile it looks at its up/left/down/right neighbors that are not pillars, and
if at least one is strictly lower than the current tile, it moves to the
lowest such neighbor (ties broken by preferring, in order, Up, then Left,
then Down, then Right). It stops the instant it lands on a gutter tile
(counted as drained) or has no strictly-lower neighbor to move to (it pools
there forever, uncounted). This routing is completely deterministic.

Because the routing rule is fixed, your re-tiled roof partitions into
drainage basins exactly like a real watershed map: which basin a tile
belongs to is a piecewise-constant function of the heightfield, and it only
changes where an edit flips which side of a **ridge** (a local high point
between two basins) a tile's downhill path exits through. Editing tiles deep
inside a basin that stays a basin cannot change a single drop's fate.

## Input (stdin)
```
R C
S B
H[0][0] ... H[0][C-1]
...
H[R-1][0] ... H[R-1][C-1]
<R lines, C chars each: '#' = pillar, '.' = ordinary tile>
<R lines, C chars each: 'G' = gutter, '.' = non-draining tile>
```
The given heightfield already satisfies the max-slope bound, so changing
nothing is always feasible.

## Output (stdout)
`R` lines of `C` integers: your re-tiled heightfield `H'`.

## Feasibility
All of the following must hold, or the submission scores `0`:
- exactly `R*C` finite integer tokens are printed;
- `H'[r][c] == H[r][c]` for every pillar tile;
- `sum(|H'[r][c] - H[r][c]|)` over non-pillar tiles is at most `B`;
- `|H'[r][c] - H'[r2][c2]| <= S` for every orthogonally adjacent pair of
  non-pillar tiles.

## Objective
Maximize `F`, the number of raindrops that reach a gutter tile after being
dropped (once each) on your re-tiled roof `H'` and routed as above.

## Scoring
Let `Bnatural` be the number of drops that already reach a gutter on the
**original, unedited** heightfield (always well defined and feasible, since
leaving everything unchanged is a valid submission). The scored reference
`Bref` is a deliberately pessimistic **37%** of that natural count --
`Bref = max(1, round(0.37 * Bnatural))` -- so leaving the roof unchanged is
still feasible but is not treated as "the target": only reconnecting basins
that the unedited roof leaves trapped shows up as real score. With `F` as
above:
```
sc = min(1000.0, 100.0 * F / max(1e-9, Bref))
Ratio = sc / 1000.0
```
Leaving the roof unchanged scores `Ratio = sc/1000` with `F = Bnatural`,
i.e. `Ratio = min(1.0, 0.1 / 0.37) ~= 0.270`.

## Constraints
- `90 <= R <= 320`, `100 <= C <= 330`.
- `S = 5` (fixed), `B >= 1` (both given in the input, per test).
- Time limit 5s, memory 512MB.

## Example
Suppose a small basin of 12 tiles currently pools behind a raised rim
instead of reaching the nearby gutter that the rest of the roof already
drains to. Tilting the *entire* roof toward that gutter would touch every
tile and blow the edit budget, and it still would not clear the basin --
its low point stays a low point no matter how the rest of the surface tilts.
Instead, lowering the single rim tile that borders both the basin and the
already-draining terrain, by just enough to make it lower than the basin's
edge while staying within `S` of its other neighbors, redirects all 12 drops
at once for a handful of edit budget. `F` rises by 12, `Bref` is unchanged,
and the ratio increases accordingly.
