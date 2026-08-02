# Echo Room: Wall Reconstruction from Unlabeled First-Order Echoes

## Problem

A speaker sits at a fixed, known point **S** inside a room with **W** flat
walls and emits a single pulse. Sound travels at speed 1, so the length of
any path equals its travel time. Besides the direct sound, each wall
produces exactly one **first-order (single-bounce) echo**: by the
image-source method, reflecting **S** across wall *k*'s supporting line
gives an *image point* `I_k`, and the length of the source→wall-*k*→receiver
path, for any receiver point **R**, equals `|I_k − R|` exactly. Higher-order
(multi-bounce) reflections are not modeled.

You are given **K** microphones at known positions. Each reports only the
*unlabeled* multiset of first-order echo arrival times it heard — you are
never told which number came from which wall, and the same wall can rank
differently (1st-nearest, 2nd-nearest, ...) at different microphones. Some
instances also splice in **decoy** numbers close to a real echo time but
not equal to the distance from any single, globally consistent point to
that microphone — clutter that a naive "every number is a genuine echo"
assumption will misuse.

Your job: reconstruct the **W** wall lines.

## Input (stdin)

```
W K testId
Sx Sy
mx_1 my_1 L_1 t_1 t_2 ... t_{L_1}
...
mx_K my_K L_K t_1 t_2 ... t_{L_K}
```
`W` walls, `K` given microphones, `testId` is an opaque ladder index. Each
of the `K` lines gives a microphone's position and its `L_i >= W` unlabeled
echo readings (order carries no information; some instances splice in a
few decoys, so `L_i` can exceed `W`).

## Output (stdout)

```
W
x1_1 y1_1 x2_1 y2_1
...
x1_W y1_W x2_W y2_W
```
First the wall count `W` (must match the input), then exactly `W` lines,
each given as two distinct points `(x1,y1)`–`(x2,y2)` that lie on that
wall's supporting line (order of the `W` walls in your output does not
matter, and the two points need not be the actual wall endpoints — only
the infinite line through them is used).

## Feasibility

Your output is rejected (score 0) if: the wall count differs from `W`; any
coordinate is non-finite or exceeds `1e4`; a wall's two points coincide
(distance `< 1e-3`); `S` lies closer than `0.05` or farther than `60` from
some wall's line (degenerate/absurd reflection); or two walls reflect `S`
to (near-)identical image points (`< 0.05` apart — no faking several walls
with one good guess).

## Objective

We reflect `S` across each of your `W` lines to get candidate image points,
then forward-simulate first-order echo times at **two held-out microphone
positions never shown to you**. At each held-out microphone we sort your
`W` predicted times and the true `W` times and take the mean absolute
difference between the two sorted lists, then average over the two
microphones to get `MAE`. Quality is `Q = max(0, 1 − MAE / 6.0)`. A wall
assignment that merely reproduces the *given* microphones' numbers (there
are usually several such assignments — echo-labeling is combinatorial) but
is not the geometrically correct room will generally mispredict the
held-out microphone and score low, even if it looked perfect on everything
you could see.

## Scoring

The checker also computes `Q_base`, the same quality measure for a fixed
generic room (walls placed at a small constant radius from `S`, evenly
spread in angle, ignoring all echo data). Your ratio is
`min(1000, 100 * Q / Q_base) / 1000`, printed as `Ratio: <value>` — a data-
blind guess scores near 0.1; better rooms score higher; the ratio never
exceeds 1.0.

## Constraints

`4 <= W <= 7`, `4 <= K <= 5`, coordinates and readings are real numbers of
modest magnitude (room extent on the order of 10 units). Time limit 5s,
memory 512MB.

## Example (worked score, illustrative shape only)

Suppose `W=1`, `S=(0,0)`, one microphone at `(1,0)` reports a single
reading `4.0`. The true wall's image point is unknown to you, but any line
whose reflection of `S` lands on the circle of radius 4 around `(1,0)` is
consistent with this one microphone — e.g. image point `(5,0)` (wall line
`x=2.5`) or `(1,-4)` are both locally consistent; only a held-out
microphone elsewhere would tell them apart. This illustrates the *shape* of
the ambiguity only — it is not the construction used to build the actual
test data.
