# The Merchants' Guild: Screening Caravan Insurance

## Problem
The Caravan Merchants' Guild wants to sell raid insurance to `n` independent
expeditions. Expedition `i` is described by four numbers, all known to you:

- `p_i` -- the true probability its caravan is raided on the route,
- `a_i` -- its risk-aversion coefficient (how much it dislikes uncertainty),
- `L_i` -- the cargo value at risk,
- `u_i` -- the utility of its outside option (hiring private escorts instead
  of buying Guild insurance; usually negative).

You design a **menu** of at most 6 contracts. Contract `j` is a pair
`(P_j, c_j)`: a premium `P_j >= 0` paid regardless of outcome, and a coverage
fraction `c_j in [0,1]` -- if raided, the Guild reimburses `c_j * L_i` and the
expedition bears the rest itself. Every expedition independently looks at the
whole menu (plus the option to buy nothing) and picks whichever it likes best.
An expedition's utility from contract `(P,c)`, given its own `p_i,a_i,L_i`, is

```
U(P,c) = -( P + p_i*(1-c)*L_i ) - a_i * p_i*(1-p_i) * ((1-c)*L_i)^2
```

(expected out-of-pocket loss, plus a penalty for the residual risk it still
carries, scaled by its own risk aversion). It buys the contract maximizing
`U` if that beats its outside option `u_i`; otherwise it walks away and buys
nothing. Ties (equal utility) are broken in favor of walking away, and among
tied contracts the lowest-indexed one is preferred.

The Guild's **true** profit from an expedition that buys `(P,c)` is
`P - p_i * c * L_i` (premium minus the *actual* expected payout, using the
real `p_i`, not any claim). Expeditions that walk away contribute `0`. You
only choose the menu; every choice follows from self-interest.

## Input (stdin)
```
n
p_1 a_1 L_1 u_1
...
p_n a_n L_n u_n
```
`p_i in (0,1)`, `a_i >= 0`, `L_i` a positive integer, `u_i` a real number
(often large and negative). `200 <= n <= 2500`.

## Output (stdout)
```
k
P_1 c_1
...
P_k c_k
```
`0 <= k <= 6`. Each `P_j >= 0` finite, each `c_j` in `[0,1]` finite.

## Feasibility
Reject (score `0`) if: `k` is not an integer in `[0,6]`; the token count does
not match `1 + 2k`; any `P_j` or `c_j` is missing, non-numeric, non-finite,
negative (for `P_j`), or outside `[0,1]` (for `c_j`).

## Objective
Maximize `F`, the Guild's total true profit summed over all `n` expeditions
under their self-interested choices from your posted menu (as defined above).

## Scoring
The checker simulates every expedition's choice against your menu to get `F`.
It also computes two references of its own, purely from the input: `w_base`,
the profit of a single full-coverage contract priced at the raw (no-markup)
population-mean expected loss, and `w_ref`, the profit of the *best possible*
single full-coverage contract (found by an exact sweep over price
thresholds). Both ignore that different expeditions may need different
treatment. Score:
```
r = clamp( 0.10 + 0.70 * (F - w_base) / max(1e-9, w_ref - w_base), 0, 1 )
```
Matching `w_base` scores near `0.10`; matching the best single price scores
near `0.80`; a menu that serves different expeditions differently can score
higher still. Your score is the mean of `r` over 10 fixed instances.

## Constraints
Time limit 5s, memory 512m. Deterministic, seeded generator and checker.

## Example (illustrative only, not a real test case)
Two expeditions: #1 has `p=0.05, a=0.0002, L=1000, u=-70`; #2 has
`p=0.40, a=0.006, L=1500, u=-950`. Post the menu `k=2`: `(30, 0.5)` then
`(700, 1.0)`. For #1, contract 1 gives `U = -(30+0.05*500) -
0.0002*0.05*0.95*500^2 = -57.4`, better than `u=-70` and better than
contract 2's `U=-700` -- #1 buys contract 1. For #2, contract 2 gives
`U=-700`, better than `u=-950` and better than contract 1's `U=-1140`
(its own residual risk on a half-covered `1500`-value cargo is expensive) --
#2 buys contract 2. Guild profit: `(30 - 0.05*0.5*1000) + (700 -
0.40*1.0*1500) = 5 + 100 = 105`. A single blanket price could not have
captured both margins at once: cheap enough for #1's thin margin is far too
cheap for #2's true risk, and #2's fair price is far above what #1 will pay.
