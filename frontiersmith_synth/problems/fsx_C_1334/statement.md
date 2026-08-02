# Pruning a Kinetic Reaction Network Under an Accuracy Bound

## Problem
A first-order (linearized) kinetic reaction network has `n` chemical species and `m`
reactions. Reaction `i` converts species `r_i` into species `p_i` at rate constant
`k_i > 0`: `dc[r_i]/dt -= k_i*c[r_i]`, `dc[p_i]/dt += k_i*c[r_i]`. Given an initial
concentration vector, integrating this linear ODE system over a fixed time horizon `T`
produces the trajectory of every species, in particular of one designated **target
species**.

You are given `P` independent **held-out operating conditions** — different initial
concentration vectors, representing different experimental startups of the same
mechanism — and an accuracy tolerance `epsilon`. Output a **subset of the `m`
reactions** (a reduced mechanism) such that, when each condition is re-integrated using
only the kept reactions, the target species' trajectory stays within `epsilon` of the
full-mechanism trajectory for **every one of the `P` conditions**. Minimize the number
of reactions you keep.

This is mechanism reduction as it appears in combustion/atmospheric chemistry and, more
generally, stiff-network model-order reduction: most of a large mechanism's reactions
are irrelevant to any one observable, but *which* ones are irrelevant can depend on
which starting mixture feeds the network. A reaction that carries almost no flux under
one startup can be the *only* path by which another startup's precursor ever reaches
the target — so a reaction's raw rate constant is not a reliable proxy for whether it
matters; what matters is its actual contribution to the target's trajectory, checked
against every condition you are given, not just the one you happen to have tried first.

## Input (stdin)
```
n m target P T_horizon N_steps epsilon
r_1 p_1 k_1            (m lines; 0-indexed reactant, product, rate constant)
...
c_0 c_1 ... c_{n-1}    (P lines; condition p's initial concentration of every species)
```
Integrate with fixed-step RK4 using exactly `N_steps` steps of size `T_horizon/N_steps`
— this exact scheme is what the checker uses to grade you.

## Output (stdout)
```
k
i_1 i_2 ... i_k
```
`k` = number of kept reactions, followed by `k` pairwise-distinct reaction indices in
`[0, m-1]` (whitespace/newline separated). No other tokens may appear.

## Feasibility
- Output must parse exactly: a valid integer `k`, then exactly `k` valid distinct
  integer indices in `[0, m-1]`, and nothing else. `k = 0` is always infeasible.
- For **every** condition `p` in `0..P-1`: `max_t |c_target,full(t) - c_target,reduced(t)|
  <= epsilon`, using the same fixed RK4 scheme on the full network and on the network
  built from only your kept reactions.
- Any violation (parse error, out-of-range/duplicate index, or a condition that busts
  the tolerance) scores `Ratio: 0.0`.

## Objective
Minimize `F = k`, the number of kept reactions.

## Scoring
`B = m` (keeping every reaction — always feasible, exact). `Ratio = min(1, 0.1*B/F)`.
Keeping everything scores ≈0.1; halving the mechanism scores ≈0.2; a 5×-smaller
feasible mechanism caps the ratio at 1.0.

## Constraints
`n <= 60`, `m <= 30`, `P = 5`, `N_steps <= 100`, all rate constants in `(0, 10]`,
`0 < epsilon <= 0.2`. Time limit 5s, memory 512MB.

## Example
2 species, `target = 1`, one reaction `0 -> 1` at rate `1.0`, one condition
`c = [1.0, 0.0]`, `T = 1, N_steps = 10, epsilon = 0.01`. Keeping the only reaction
reproduces the full trajectory exactly: `F = 1`, `B = 1`,
`Ratio = min(1, 0.1*1/1) = 0.1`. Dropping it freezes the target at `0`, missing the true
rising trajectory by far more than `0.01` — infeasible, `Ratio: 0.0`.
