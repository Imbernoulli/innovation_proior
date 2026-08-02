# Copolymer Sequence Design for a Target Glass Transition

## Problem

You are designing a copolymer chain of length `N` built from `K=3` monomer
types. Type `i` has a pure-component glass transition temperature `tg[i]`.
When two monomer units of types `i` and `j` sit next to each other in the
chain, that **junction** (an ordered pair of adjacent units, called a dyad)
has its own local glass transition `M[i][j]` (symmetric, `M[i][i]=tg[i]`).
`M[i][j]` need **not** be close to the average of `tg[i]` and `tg[j]`: a
junction can be strongly *plasticizing* (local mobility rises, `M[i][j]`
well below both pure components) or strongly *reinforcing* (interfacial
packing stiffens the chain, `M[i][j]` well above both).

The chain's predicted glass transition is the bond-frequency-weighted
harmonic mean of the dyad values over all `N-1` adjacent junctions:

```
1 / Tg_pred  =  (1 / (N-1)) * sum over adjacent pairs (s_k, s_{k+1}) of  1 / M[s_k][s_{k+1}]
```

Two chains with the *identical monomer composition* (same counts of each
type) can have very different `Tg_pred`, because the formula depends on
*which* pairs of monomers actually end up adjacent -- i.e. on whether the
chain is arranged in long blocky runs or finely alternated, not merely on
the overall ratio of monomers used.

Feedstock is limited: at most `caps[i]` units of monomer type `i` are
available, so the chain you build must respect `count[i] <= caps[i]` for
every type (a feasible chain of length exactly `N` respecting all caps
always exists in every test).

## Input (stdin)

```
N K
tg[1] tg[2] tg[3]
M[1][1] M[1][2] M[1][3]
M[2][1] M[2][2] M[2][3]
M[3][1] M[3][2] M[3][3]
caps[1] caps[2] caps[3]
target
```
All values are integers. `M` is symmetric with `M[i][i]=tg[i]`.

## Output (stdout)

`N` whitespace-separated integers `s_1 ... s_N`, each in `[1,K]`: the
monomer type placed at each position of the chain, in order.

## Feasibility

- Exactly `N` tokens, each a base-10 integer in `[1,K]`.
- For every type `i`, the number of times it appears is `<= caps[i]`.
Any violation scores `Ratio: 0.0`.

## Objective

Let `Tg_pred` be computed from your sequence by the formula above, and let
`err = |Tg_pred - target|`. Your raw quality is
`F = exp(-err^2 / (2*sigma^2))`, `sigma=10`, so an exact hit scores `F=1`
and quality decays smoothly as you miss the target. The checker also builds
its own naive reference chain `B`: it fills monomer types, in blocks, in
order of how poorly their *pure* `tg[i]` matches `target` (worst match
first) -- a construction that never looks at the interaction matrix at all
-- and reports `Ratio = min(1, F / (10*B))`.

**Maximize the printed `Ratio`.**

## Constraints

`20 <= N <= 220`, `3 <= K <= 3`, `60 <= M[i][j] <= 600`, `1 <= caps[i] <= N`,
time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

For `N=4, K=3`, `tg=[300,300,300]`, `M` all-equal `300` everywhere, `caps=[4,4,4]`,
`target=300`: any sequence gives `Tg_pred=300` exactly (`err=0`, `F=1`) --
composition and arrangement are both irrelevant when every dyad has the same
value. The real test cases plant meaningfully different `M[i][j]` values, so
arrangement stops being irrelevant.
