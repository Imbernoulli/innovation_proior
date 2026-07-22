# Garden of Iterated Light: Seeding Portraits on a Ring

## Problem
A garden is a ring of `n` cells, each holding a shade in `F_p = {0,...,p-1}` (a fixed
prime `p`). You choose the **initial planting** `x_0[0..n-1]`, then every season the
whole ring updates simultaneously by a fixed **linear rule** of radius `r`, with integer
coefficients `c_{-r},...,c_r` (each in `[1,p-1]`, given in the input):

```
x_{t+1}[i] = ( sum_{d=-r}^{r} c_d * x_t[(i - d) mod n] )  mod p
```

Because this rule is *linear*, the shade of any cell at any future season is an exact
linear function of the initial planting `x_0` — computable directly, without simulating
season by season, via fast exponentiation of the update matrix (equivalently: repeated
squaring of the length-`n` coefficient vector under cyclic convolution mod `x^n-1`).
This matters because seasons range up to `t = 10^9`: literally stepping the ring that
many times is not an option.

Soil is scarce: your planting `x_0` may have **at most `s` nonzero cells** (all others
must be exactly `0`).

You are given `k` **target portraits**: tuples `(t_i, pos_i, val_i, w_i)` meaning "cell
`pos_i` should show exactly shade `val_i` at season `t_i`", each with an importance
weight `w_i`. Seasons `t_i` range from `0` (an instant portrait, no growth at all) up to
`10^9`.

## Input (stdin)
```
n p r s k
c_{-r} c_{-r+1} ... c_{r}
t_1 pos_1 val_1 w_1
...
t_k pos_k val_k w_k
```
All values are integers; `0 <= pos_i < n`, `0 <= val_i < p`, `1 <= c_d <= p-1`, `1 <= w_i <= 30`.

## Output (stdout)
Exactly `n` space/newline-separated integers: `x_0[0], x_0[1], ..., x_0[n-1]`.

## Feasibility
An output is valid iff **all** hold:
- exactly `n` integers are printed, each in `[0, p-1]`;
- at most `s` of them are nonzero.
Any violation scores `Ratio: 0.0`.

## Objective
Let `x_t[i]` denote the true state after `t` seasons of evolution from your `x_0`,
computed exactly mod `p`. Maximize
```
F = sum of w_i over all targets i with x_{t_i}[pos_i] == val_i  (mod p, exact)
```
i.e. the total importance of the portraits your garden reproduces perfectly on schedule.

## Scoring
The checker builds its own trivial reference: the single best one-shot seed. Over every
(target `i`, planting cell `j`) pair, it solves for the one value at cell `j` that
satisfies target `i` exactly in isolation, then scores that single seed by however many
targets (including possibly others, by luck or structure) it happens to also match. `B`
is the best score achieved this way over all `(i,j)` (always well-defined and positive:
the instant target at season `0` alone already gives one valid candidate). With `F` your
matched importance:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the checker's single-seed baseline scores `Ratio = 0.1`; doing `10x` better
caps at `1.0`.

## Constraints
- `8 <= n <= 40`, `p = 1000003` (fixed prime, same for every case), `1 <= r <= 2`,
  `1 <= s <= 8`, `2 <= k <= 14`.
- Time limit 5s, memory 512m.

## Example
Toy instance: `n=4, p=5, r=1`, coefficients `c_{-1}=1, c_0=2, c_1=1` — illustrative
FORM only, not the real test data. Targets: `(t=0,pos=2,val=3,w=8)`,
`(t=1,pos=0,val=2,w=10)`, `(t=1,pos=1,val=1,w=5)`.

Submit `x_0 = [1,0,0,0]` (budget allows). One season later:
`x_1[0] = 1*x_0[1] + 2*x_0[0] + 1*x_0[3] = 2`, `x_1[1] = 1*x_0[2] + 2*x_0[1] + 1*x_0[0] = 1`
— both evolved targets match exactly (`F = 10 + 5 = 15`); the instant target at `pos=2`
needs `x_0[2]=3` but got `0`, so it misses.

The checker's single-seed baseline: planting only `x_0[2]=3` satisfies the instant
target exactly (`B = 8`) and, checking, does not coincidentally also satisfy the other
two. So `sc = min(1000, 100*15/8) = 187.5`, `Ratio = 0.1875`.
