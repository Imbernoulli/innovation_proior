# Enzyme Staging for Target Flux

## Problem

A small metabolic network has `R` reactions and `R` internal metabolites,
plus one external, unlimited-supply source metabolite (id `0`, fixed
concentration `X0`, given). Reaction `i` (`1..R`) consumes its designated
substrate metabolite `parent_i` (either `0`, the external source, or an
already-produced internal metabolite `< i`) and produces `yield_i` units of
its **own** metabolite `i`. Reaction `i`'s rate follows Michaelis-Menten
kinetics:

```
v_i = e_i * kcat_i * x_{parent_i} / (Km_i + x_{parent_i})
```

where `e_i in [0, e_max_i]` is the enzyme (Vmax multiplier) you choose, and
`x_{parent_i}` is the steady-state concentration of `i`'s substrate.

Every internal metabolite `i` sits at whatever concentration balances its
production against its uptake by the reactions that consume it (its
"children" in the production tree): at steady state,

```
x_i = tau_i * max(yield_i * v_i - sum_{k consumes i} v_k, 0)
```

Because the children's uptake `v_k` itself depends on `x_i`, this is a
genuine fixed-point equation at every metabolite, not a one-shot formula —
some reactions end up **limiting** (`x << Km`, rate nearly proportional to
`x`) and others **saturated** (`x >> Km`, rate nearly capped, insensitive
to further concentration changes). Which regime a reaction lands in is
determined only by solving the network, not by inspection.

**Your task:** choose enzyme levels `e_1..e_R` (each bounded in
`[0, e_max_i]`) so that the resulting steady-state flux vector matches a
given **target** `v_target_1..v_target_R` as closely as possible, while
not spending more enzyme than necessary.

**Objective (minimize):**
```
F = mean_i( |v_ss_i - v_target_i| / (|v_target_i| + 0.05) )   +   0.15 * mean_i( cost_i * e_i )
```
where `v_ss` is the flux vector obtained by actually solving the network's
steady state at your chosen `e`. The first term is mean relative flux
error; the second is a mild enzyme-cost penalty (`cost_i` given per
reaction).

## Input (stdin)
```
R X0
parent_1 yield_1 kcat_1 Km_1 tau_1 e_max_1 cost_1
...
parent_R yield_R kcat_R Km_R tau_R e_max_R cost_R
x_ref_1 x_ref_2 ... x_ref_R
v_target_1 v_target_2 ... v_target_R
```
`parent_i` is `0` or an integer `< i`. `x_ref` is the steady-state
concentration vector obtained at the **nominal** enzyme profile `e_i = 1`
for every reaction — a reference point, not necessarily anywhere near the
concentration the target flux actually requires.

## Output (stdout)
`R` numbers `e_1 ... e_R`, each satisfying `0 <= e_i <= e_max_i`.

## Feasibility
Output must be exactly `R` finite numbers, each within `[0, e_max_i]`
(tolerance `1e-6`). Any violation (wrong count, out of range,
non-finite) scores `Ratio: 0.0`.

## Scoring
The checker re-solves the network's steady state at your submitted `e`
(the same deterministic per-metabolite fixed-point solve described
above), computes `F` as defined, and compares it to `B`, the checker's own
baseline `F` at `e_i = e_max_i/2` for every reaction. Your score is
`min(900, 100*B/F)` printed as `Ratio: <score/1000>` on the final line
(so the naive baseline scores `~0.1`; the score cap leaves headroom above
what the reference solutions reach).

## Worked example (illustrative shape only)
Two reactions, `R=2`, `X0=10`: reaction 1 consumes the source and produces
metabolite 1; reaction 2 consumes metabolite 1. If `v_target_1 = 1.0` and
`x` at that flux level works out to `2.0` while `Km_2 = 8.0`, reaction 2 is
deep in the **limiting** regime — a small change in `x` moves its rate a
lot, so a `v_target_2` set at nominal `x_ref` (say `x_ref=5.0`) will need a
very different `e_2` than the true regime implies. Getting `e_2` right
means solving for what `x` the *target* actually implies, not reusing
`x_ref`.

## Constraints
`4 <= R <= 15`, `1 <= e_max_i <= 10`, time limit 5s, memory 512MB.
