# No-Fridge Banquet: Minimal Re-Prep Under a Burner Cap

## Problem

You are cooking a banquet with **no fridge**. The recipe is a strict chain of
`N` prep steps `1..N`: step `i` can only be cooked once step `i-1`'s
component is physically sitting warm on a burner (step `1` has no
prerequisite). Cooking step `i` costs `c[i]` (kitchen effort units) *every
single time* you cook it — recompute it twice, pay twice.

Your kitchen has exactly `M` burners, so **at most `M` distinct step-outputs
may be warm at once**. Anything not on a burner is gone for good (no fridge)
— if you need it again later you must re-cook it, which means re-cooking the
whole chain from whatever earlier step is still warm.

The head chef will call for `K` specific steps to be plated, one at a time,
in a fixed order given in the input (**strictly decreasing** step indices).
When step `r` is called, its output must be warm on a burner at that exact
moment.

## Input (stdin)

```
N M K
c[1] c[2] ... c[N]
r[1] r[2] ... r[K]        (strictly decreasing, each in [1,N])
```
`1 <= K <= N`, `M >= 2`, each `c[i] >= 1`.

## Output (stdout)

A sequence of actions, one per line, each `X i` with `X in {C, E, U}`:
- `C i` — cook (or re-cook) step `i`. Requires `i == 1` or step `i-1`
  currently warm. Costs `c[i]`, and step `i` becomes warm.
- `E i` — let step `i` go cold (free a burner). Requires `i` currently warm.
- `U i` — plate step `i` for the chef. Requires `i` currently warm. The
  full sequence of `U` actions, in order, must equal `r[1..K]` exactly.

## Feasibility

Reject (score 0) if: any `C i` fires with `i>1` and `i-1` not warm; more than
`M` steps are warm at once after any action; any `E`/`U i` targets a step
that is not currently warm; the emitted `U` sequence is not exactly
`r[1],...,r[K]` in order; or the output is malformed/unparseable.

## Objective

Minimize the total cost of all executed `C` actions (recooking counts every
time). This is a pure store-vs-recompute trade-off: keeping a step warm
(spending a burner) can save you from re-paying an expensive stretch of the
chain later, but burners are scarce.

## Scoring

The checker computes your total cost `F` and its own baseline `B`: cost of
re-cooking each requested step from scratch (steps `1..r`) using only 2
burners, ignoring caching entirely. For minimization,
`ratio = min(1, 0.1 * B / F)`, printed as `Ratio: <value in [0,1]>`. Lower
`F` (fewer wasted re-cooks) scores higher, capped below 1 so there is always
room to do better.

## Constraints

`N` up to a few thousand, `M` moderate (single/low double digits relative to
`N`), time limit 5s, memory 512MB.

## Example (worked, illustrative shape only — costs are NOT indicative of the
real instances, which plant sharp cost bursts)

`N=4 M=3`, `c=[1,1,1,1]`, requests `r=[4,2]`. One feasible schedule: `C1 C2
E1 C3 C4 U4 E4 E3` then re-cook `C1 C2 U2`... — many action sequences are
valid; the checker only cares that dependencies/capacity/order hold and that
total `C` cost is small. The exact coefficients and where the chain's costly
stretches sit are only visible in the input's `c[]` array — read it, don't
assume it is uniform.
