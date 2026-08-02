# Thermostable Enzyme Redesign: Epistasis-Screened Mutation Stacking

## Problem
You are redesigning an enzyme to survive higher temperatures without losing its catalytic
function. A computational scan proposes `n` candidate point mutations, each at a distinct
residue site `i = 0..n-1`. For site `i` you are given:

- `dstab[i]` -- the mutation's **individual** contribution to thermostability if installed alone.
- `dact[i]` -- the mutation's **individual** effect on catalytic activity if installed alone
  (usually negative: stabilizing a fold often costs some activity -- the core tradeoff).
- `dist[i]` -- an integer structural distance from the catalytic active site.

You may install a **subset** `S` of at most `K` mutations (a fixed lab budget: each mutation
costs a round of cloning). Naively you might assume effects just add up. They do not: a sparse
table lists **pairwise epistasis corrections** -- extra stability and activity terms that apply
ONLY when both mutations of a listed pair are installed together. Pairs absent from the table are
genuinely additive (correction `0`). Separately, packing too many mutations into the active-site
**neighbourhood** (`dist[i] <= R`) causes a *nonlinear* activity penalty from steric/electrostatic
crowding, on top of any tabulated pairwise terms: if more than `C` of your installed mutations lie
in the neighbourhood, activity drops by `alpha * (count - C)^2`.

The enzyme is useless if it loses too much activity. Your redesign must keep predicted activity at
or above a floor `ActMin`. Subject to that, maximize total stability gain.

## Input (stdin)
```
n K C R
A0 ActMin alpha
dstab_0 dact_0 dist_0
dstab_1 dact_1 dist_1
...
dstab_{n-1} dact_{n-1} dist_{n-1}
m_epi
i_1 j_1 e_stab_1 e_act_1
...
i_{m_epi} j_{m_epi} e_stab_{m_epi} e_act_{m_epi}
```
`A0` is wild-type activity. Each epistasis row (`i<j`) applies its `e_stab`/`e_act` correction
only when both `i` and `j` are in your installed set `S`; unlisted pairs contribute `0`.

## Output (stdout)
```
m
idx_1 idx_2 ... idx_m
```
`m` = number of mutations you install (`0 <= m <= K`), followed by `m` distinct site indices in
`[0, n-1]`. If `m = 0`, the second line may be empty.

## Feasibility
Rejected (score `0`) if: the output has any non-integer token; token count != `1+m`; `m` is
negative or `> K`; any index repeats or lies outside `[0, n-1]`; or the resulting activity
```
act(S) = A0 + sum_{i in S} dact[i] + sum_{(i,j) in S, tabulated} e_act[i,j]
              - alpha * max(0, |{i in S : dist[i] <= R}| - C)^2
```
falls below `ActMin`.

## Objective (maximize)
For a feasible `S`:
```
stab(S) = sum_{i in S} dstab[i] + sum_{(i,j) in S, tabulated} e_stab[i,j]
F = max(0, stab(S))
```

## Scoring
Let `B` = the checker's own baseline: the best single mutation `i` whose activity alone (with its
own crowding term) already meets `ActMin`. Then
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
Installing nothing beyond that one mutation scores `0.1`. Stacking several mutations that avoid
antagonistic pairs, exploit synergistic pairs, and respect the neighbourhood cap can score
substantially higher.

## Constraints
`8 <= n <= 18`, `3 <= K <= 6`, `1 <= C <= 2`, `R = 3`, `0 <= dist[i] <= 9`, `A0 = 100`,
`ActMin = 55`, `alpha = 4`. All numeric input values are given to 3 decimals.

## Example
`n=4, K=2, C=1, R=3`; `A0=100, ActMin=55, alpha=4`; sites: `(2.0,-1.0,1) (1.8,-1.0,5) (1.5,-0.8,7)
(1.2,-0.6,8)`; one epistasis row `0 1 -3.6 -2.0` (sites 0,1 are both near the active site and
clash). Installing `{0,1}` (the two best individually) gives `stab = 2.0+1.8-3.6 = 0.2`, wiped out
by the clash. Installing `{0,2}` instead gives `stab = 3.5`, a feasible, far better choice.
