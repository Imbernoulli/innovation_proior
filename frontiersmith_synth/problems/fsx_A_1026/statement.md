# Trichroic Ice Lattice: One Lens, Three Colors

## Problem
An ice wizard channels a single beam of light through a rectangular lattice of ice
cells: `W` columns `x = 0..W-1` (the direction of travel) by `H` rows `y = 0..H-1`
(the transverse position). Each cell `(x,y)` has an integer **refractive level**
`Lvl[x][y]` in `[0, LMAX]`, chosen by the wizard subject to a total **budget**:
`sum of all Lvl[x][y] <= K`. Undesigned cells default to level `0`.

The beam carries three colors, indexed `0,1,2`, each with its own **dispersion
coefficient** `CHROMA[c]`. A cell's refractive index for color `c` is
`n(x,y,c) = 1 + ALPHA * CHROMA[c] * Lvl[x][y]` -- the same lattice produces a
*different* index for each color (chromatic splitting).

Each color enters as a small bundle of `R` parallel rays from column `x=0`, at
given real-valued rows `y0`. A ray's state is `(y, s)` (transverse position,
slope), starting at `s=0`. For each step `x -> x+1` (`x=0..W-2`):
1. Let `r = floor(y)` (clamped to `[0,H-1]`), `ru=min(r+1,H-1)`, `rd=max(r-1,0)`.
   **Gradient-index bending**: `s += KAPPA * (n(x,ru,c) - n(x,rd,c))`.
2. Let `y' = y+s`, `r' = floor(y')` (clamped). If `r' != r` the ray crosses into
   a new row's cell: apply **discrete Snell steering**, conserving the
   tangential invariant `n / sqrt(1+s^2)` across the boundary: with
   `n1=n(x,r,c)`, `n2=n(x,r',c)`, let `rhs = clamp((n1/n2)/sqrt(1+s^2), 1e-9,
   1-1e-9)`, then `s <- sign(s) * sqrt(1/rhs^2 - 1)` (`sign(0)=+1`).
3. `y += s`; clamp `s` to `[-4,4]` and `y` to `[0, H-1e-9]` (the lattice walls
   absorb overshoot deterministically).

After `W-1` steps, `y` is the ray's **exit row** at column `W-1`.

## Input (stdin)
```
W H
LMAX K
ALPHA KAPPA
CHROMA_0 CHROMA_1 CHROMA_2
T_0 T_1 T_2
R
y0 values for color 0 (R ints)
y0 values for color 1 (R ints)
y0 values for color 2 (R ints)
```
`T_c` is color `c`'s target exit row.

## Output (stdout)
Print `Lvl[x][y]` as a flat stream of whitespace-separated integers in
row-major order: row `y=0`'s `W` values (`x=0..W-1`), then row `y=1`'s, ...,
then row `H-1`'s. (Printing them one row per line is the natural layout and
is what the reference solutions do, but only the token count and order
matter -- line breaks are not otherwise significant.)

## Feasibility
The output is valid iff **all** hold: exactly `H*W` whitespace-separated
tokens, each a plain integer literal (no floats/`nan`/`inf`/extra or missing
tokens); every level is an integer in `[0,LMAX]`; the sum of all levels is
`<= K`. Any violation -> `Ratio: 0.0`.

## Objective
For every ray (each of the `3*R` rays), score `1/(1+|exit_row - T_c|)`
(`c` = that ray's color). Maximize `F`, the sum of these scores over all rays.

## Scoring
Let `B` be the same sum `F` computed for the **all-zero lattice** (no budget
spent -- the checker's own trivial construction). Then
`sc = min(1000, 100*F/B)`, printed as `Ratio: %.6f` (`sc/1000`). A design that
does no better than doing nothing scores `~0.1`; one that is `10x` better caps
at `1.0`.

## Constraints
`12 <= W,H <= 40`, `LMAX=6`, `K` scales with `W*H`, `R=3`. Time limit 5s,
memory 512MB.

## Example (worked score, illustrative values only)
`W=3,H=5,LMAX>=4,ALPHA=0.1,KAPPA=0.6`, one ray per color at `y0=2`,
`CHROMA=[0.5,1.0,2.0]`, `T=[0,2,4]`. Submitted lattice columns (rows
`y=0..4`) `x=0: [0,0,0,4,0]`, `x=1: [0,0,0,1,0]`; `x=2` is never read (only
`x=0,1` are the "current column" of the `W-1=2` steps).
- Color 0 (chroma 0.5): step1 `grad=0.20,s=0.12,y=2.12`; step2 `grad=0.05,
  s=0.15,y=2.27` (no crossing). Exit `2.27`, target `0`, score
  `1/(1+2.27)=0.3058`.
- Color 1 (chroma 1.0): step1 `s=0.24,y=2.24`; step2 `grad=0.10,s=0.30,
  y=2.54` (no crossing). Exit `2.54`, target `2`, score `1/(1+0.54)=0.6494`.
- Color 2 (chroma 2.0): step1 `s=0.48,y=2.48`; step2 `grad=0.20,s=0.60`;
  provisional `y'=3.08` crosses row `2->3`. Snell: `n1=1.0, n2=1.2,
  rhs=(1.0/1.2)/sqrt(1.36)=0.7146, s<-sqrt(1/0.7146^2-1)=0.979`; exit
  `y=2.48+0.979=3.459`, target `4`, score `1/(1+0.541)=0.6489`.
`F=0.3058+0.6494+0.6489=1.6041`. All-zero baseline: all three rays stay at
`y=2`, giving `B=1/3+1+1/3=1.6667`. `Ratio = min(1, 0.1*1.6041/1.6667)
=0.0962`.
