# Culture Flask Scheduling: Reserve the Catalyst, Don't Chase the Brightest Reaction

## Problem

A flask holds two shared, irreversibly-consumed resources: raw nutrient
**R** (initial count `N_R`) and a catalyst **Z** (initial count `N_Z`,
never regenerated). There are `m` independent reaction groups, indexed
`0..m-1`. Group `g` is described by seven integers `px py dx dv cx cy cw`
and defines four reactions over two group-local intermediate species
`X_g`, `Y_g`, plus the shared target species **T**:

```
Px_g:    px_g R                          -> 1 X_g
Py_g:    py_g R                          -> 1 Y_g
Dir_g:   dx_g X_g                        -> dv_g T
Combo_g: cx_g X_g + cy_g Y_g + 1 Z       -> cw_g T
```

A reaction may only fire if the current stock of every reactant it needs
is sufficient; firing immediately and irreversibly subtracts the
reactants and adds the products. `Y_g` has **no** other use anywhere
except as a `Combo_g` reactant, and `Z` is consumed one unit per
`Combo_g` firing and produced by nothing.

## Input (stdin)

Line 1: `m N_R N_Z`. Then `m` lines, one per group, each
`px_g py_g dx_g dv_g cx_g cy_g cw_g` (all positive integers).
Constraints: `1 <= m <= 6`, `1 <= N_R <= 4000`, `1 <= N_Z <= 60`, all
group coefficients in `[1,9]`.

## Output (stdout)

Line 1: `K`, the number of firings. Line 2: `K` whitespace-separated
tokens, each of the form `<g><L>` where `g` is a group index and `L` is
one of `X`, `Y`, `D`, `C`, selecting which of that group's four reactions
fires next (e.g. `2C` fires `Combo_2`). The tokens are executed in the
given left-to-right order.

## Feasibility

Every token in the sequence must be affordable from the stock available
at that exact moment (raw counts, group intermediate counts, and the
catalyst all start at their initial values and only ever decrease/
increase as reactions fire). Any firing that would drive a reactant
negative makes the whole submission infeasible. `K` must equal the
number of tokens actually present, be non-negative, and every token must
parse as `<digits><X|Y|D|C>` with the group index in range.

## Objective and Scoring

Let `F` be the final count of `T` after executing the whole sequence
(0 if infeasible). The checker separately builds its own baseline `B`:
spend the entire `N_R` on group 0's `Dir_0` route alone (ignoring every
other group, `Y`, and `Z` entirely). The printed score is
`min(1000, 100*F/max(1e-9,B)) / 1000`. Feasible submissions that merely
match the baseline score `0.1`; smarter allocations score higher.

The scoring **shape** is fixed as above; the actual per-group rates that
determine which allocation is best are given only in the input, so you
must read and exploit them per instance, not the statement.

## Worked example (illustrative form only, not a planted test case)

`m=2, N_R=10, N_Z=3`. Group 0: `2 2 2 2 1 1 2`. Group 1: `1 1 1 2 1 1 5`.
Baseline `B`: group 0 direct only affords `10//(2*2)=2` firings of
`Dir_0` -> `B=4`. A submission that fires `Combo_1` three times (needs 3
`X_1` + 3 `Y_1`, costing `3+3=6` of `R`, using all `3` of `Z`, yielding
`3*5=15` T) and then spends the remaining `4` `R` on `Dir_1` (4 more
firings, `+8` T) reaches `F=23`, well above only chasing `Dir_1`'s
attractive per-step rate the whole time (which would give `F=20`). A
valid feasible token sequence realizing this: build all `1X`/`1Y` tokens
needed first (`3` copies each), then all `1C`/`1D` tokens (`3` and `4`
copies respectively) -- 13 tokens total, `K=13`.

## Notes

The single visibly-best per-firing reaction is not necessarily where the
raw material should go: a reaction group with an unglamorous or even
useless-looking side-product (`Y_g`) can unlock, together with the
scarce catalyst `Z`, a combination reaction whose true yield-per-resource
rate dominates every pure single-reactant route -- but only if enough of
that side-product and catalyst were reserved in advance rather than
spent (or ignored) reaction-by-reaction.
