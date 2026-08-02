# Solvent Blend and Sacrificial Additive Under a Window-Decomposition Cliff

## Problem
You are formulating a liquid electrolyte. A library of `N` candidate solvents
is given; you choose how to blend them into volume fractions `x_1..x_N`
(non-negative, summing to exactly 1). A separate library of `M` candidate
sacrificial additives is given; you choose a loading `a_1..a_M` (non-negative
volume fraction) for each, subject to a small shared budget `A_max` on their
total loading.

Each solvent `i` has a viscosity `eta_i`, a conductivity coefficient
`kappa_i`, and a native anodic-stability threshold `thr_i` — the highest
voltage window that solvent alone can survive at the anode before it
decomposes. Each additive `j` has an SEI-forming strength `p_j` per unit of
loading (useful only up to its own cap `cap_j` — loading it beyond `cap_j`
buys no more protection but still costs viscosity/conductivity), a viscosity
penalty `etapen_j` per unit loading, and a conductivity-dilution penalty
`kappapen_j` per unit loading.

**Electrochemical window (hard gate).** Look at every solvent actually used
in your blend (`x_i > 0`) and take the smallest of their `thr_i`. If that
minimum is at least the instance's target window `V_target`, the blend is
natively safe. Otherwise the blend is safe ONLY if your additives' combined
SEI coverage — `sum_j min(a_j, cap_j) * p_j` — reaches the instance's
`cov_target`. If neither holds, the electrolyte decomposes at the anode and
the whole formulation scores **zero conductivity** for that test case: a
sacrificial additive layer can substitute for native solvent stability, but
nothing else can.

**Conductivity/viscosity trade-off (the scored quantity, when the window
holds).** Additives always cost you something even when they save you:
```
F = max(0, sum_i x_i*kappa_i - sum_j a_j*kappapen_j) /
    (sum_i x_i*eta_i + sum_j a_j*etapen_j)
```
This is a Walden-quotient-style ratio: raise the blend's conductivity,
lower its viscosity, or both. The library is built so the solvents with the
best raw `kappa_i` are exactly the ones with the weakest native `thr_i` —
optimizing the ratio in isolation walks straight toward a solvent that then
fails the window gate.

## Input (stdin)
```
N M
A_max V_target cov_target Kconst
eta_1 kappa_1 thr_1
...
eta_N kappa_N thr_N
p_1 etapen_1 kappapen_1 cap_1
...
p_M etapen_M kappapen_M cap_M
```
All values are positive reals (`Kconst` is a fixed scaling constant folded
into the objective: multiply `F` by it before scoring).

## Output (stdout)
Exactly `N + M` whitespace-separated finite numbers: the `N` solvent
fractions `x_1..x_N` first, then the `M` additive loadings `a_1..a_M`.

## Feasibility
`x_i >= 0` for all `i`, `sum_i x_i = 1` (tolerance `1e-4`). `a_j >= 0` for
all `j`, `sum_j a_j <= A_max`. Wrong token count, non-numeric or non-finite
tokens, negative values, a fraction sum off from 1, or an additive-budget
overrun make the whole case score `0`.

## Scoring
Let `F` be the objective above (already 0 if the window gate fails). The
checker also builds its own reference: 100% of the single solvent with the
highest native `thr_i` (ties broken by lowest index), zero additive — always
window-safe by construction. Call its value `B`. Then
`Ratio = min(1000, 100 * F / B) / 1000.0`. Matching the reference scores
≈0.1; the target window is placed so no reachable blend saturates the score.
Your final score is the mean `Ratio` over 10 test cases of varying library
size, including harder cases where the window gate actively bites.

## Constraints
`4 <= N <= 8`, `2 <= M <= 4`, `0.02 <= cap_j <= 0.06`, `A_max = 0.12`. Time
limit 5s.

## Example (illustrative FORM only, small made-up numbers)
`N=2, M=1`: solvent A (`eta=1, kappa=6, thr=3.8`), solvent B (`eta=3,
kappa=3, thr=5.2`); additive (`p=6, etapen=2, kappapen=0.5, cap=0.05`).
Suppose `V_target=4.2, cov_target=0.25, A_max=0.12`. Using 100% A bare:
window fails (`3.8 < 4.2`) → `F=0`. Using 100% B bare: safe, `F=3/3=1.0`.
Using 100% A plus additive at `0.05` (near its cap): coverage
`0.05*6=0.30 >= 0.25`, window passes; `F=(6-0.05*0.5)/(1+0.05*2)=5.975/1.1
≈5.43` — far above either bare option, because the additive rescues the
fast solvent instead of the formulation retreating to the slow one.
