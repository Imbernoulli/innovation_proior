# Nested Dyadic Quantization Under a Split Budget

## Problem
You are given `C` independent **channels**. Each channel is a weighted empirical
distribution: a set of real values in `[0,1)`, each carrying a positive integer weight
(a multiplicity/mass). Every channel starts as ONE bin covering the whole interval
`[0,1)`. You may **split** a bin into two children at its **exact midpoint**
`(lo+hi)/2` -- this is the only place a split may occur. Once created, a bin's
boundaries are permanent: it can later be split again (subdividing it further), but it
can never be moved, merged, or undone. This is a **dyadic** tree: the root is depth 0,
each split increases depth by 1, and a node at depth `d`, position `p`
(`0 <= p < 2^d`) covers `[p/2^d, (p+1)/2^d)`.

You have a single **shared split budget** `S` (an integer), spent across ALL channels
combined. Every split, in every channel, costs exactly one unit of `S`. A node may be
split only if it is currently a *leaf* (not yet split) and its depth is `< D` (the
input's maximum depth).

After spending your splits, every remaining leaf must be given a **reconstruction
value** (any real number). Every original point in a channel falls into exactly one of
that channel's final leaves (by its value, using the leaf's `[lo,hi)` interval); it is
reconstructed as that leaf's declared value.

## Input (stdin)
```
C S D
P_0
v w        (P_0 lines: value in [0,1), positive integer weight)
...
P_1
...
```
`C` channels follow in order; channel `c` lists `P_c` weighted points.

## Output (stdout)
```
K
c depth pos        (K lines: split node (depth,pos) of channel c -- IN ORDER; each
                     node must currently be an open leaf of channel c, i.e. depth < D
                     and its parent, if any, was already split earlier in this list
                     or it is the root)
L
c depth pos value   (L lines: every remaining leaf of every channel, each declared
                     EXACTLY ONCE, with a real reconstruction value)
```
`0 <= K <= S`. `L` must equal the number of leaves that remain after the `K` splits.

## Feasibility
A split must target a currently-open leaf of its channel at depth `< D`. The final
leaf declarations must be a perfect bijection with the actual open-leaf set (no
missing leaf, no extra/duplicate declaration, no split of an already-split node).
Reconstruction values must be finite. Any violation scores `0`.

## Objective
Minimise the total weighted squared error:
`sum over channels, over each (value v, weight w): w * (v - reconstruction(v))^2`.

## Scoring
Let `B` be the SSE achieved by the trivial construction (every channel left as one
leaf, reconstructed by its own weighted mean -- zero splits used). Let `F` be your
total SSE. Score `= min(1, 0.1 * B / F)`. Lower `F` scores higher; matching the
zero-split baseline scores `0.1`.

## The trap
A channel that looks unpromising to split right now can be the *only* gate to a much
richer split hiding just behind it: since a coarse boundary, once fixed, can never be
repositioned, whichever fine detail lies on the wrong side of it is lost for good. A
solver that always takes whichever single split looks best *this instant* will spend
the whole budget on channels with immediately attractive (but individually capped)
gains, and may never afford the unglamorous first split that would have unlocked a far
larger downstream reduction elsewhere.

## Constraints
`1 <= C <= 100`, `1 <= S <= 200`, `1 <= D <= 6`, each channel has `2 <= P_c <= 8`
points, weights are positive integers `<= 20000`. Time limit 5s, memory 512MB.

## Example (illustrative only)
Channel with 4 points `{0.001 (w=1), 0.499 (w=1), 0.501 (w=1), 0.999 (w=1)}`, `D=3`.
Splitting the root (`0,0`) alone barely changes the mean on each side (small gain).
But following it with a split of `(1,0)` (separating `0.001` from `0.499`) and of
`(1,1)` (separating `0.501` from `0.999`) drives the SSE to near zero -- the root split
was necessary, even though its own gain looked small in isolation.
