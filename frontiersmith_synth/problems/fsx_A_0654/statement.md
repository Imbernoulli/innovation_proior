# Greenhouse Ledger: Fewest-Multiplier Plan for a Banded Conversion Cascade

## Problem
A greenhouse tracks `S` linearly-arranged storage compartments (water, sugar,
starch, ...). Each day `t = 0..T-1` the whole vector `x_t` (compartment
levels) is updated by an affine "conversion" map: `x_{t+1} = A_t * x_t + b_t`,
all arithmetic taken **modulo the prime `P = 1,000,000,007`**.

There are `K` numbered *recipe blocks* `(A_0,b_0), ..., (A_{K-1},b_{K-1})`.
Each recipe matrix `A_k` is **banded**: `A_k[i][j] = 0` whenever `|i-j| > 1`
(only neighboring compartments interact). The greenhouse follows a fixed
**rotation** of period `p`: on day `t` it normally uses recipe
`pattern[t mod p]` — *except* on `m` explicitly listed **recalibration days**,
where a special one-off **dense** map `(G,h)` (no banding, arbitrary entries)
is used instead. Recalibration days never repeat and are not part of any
rotation.

You do not need to output `x_T`. Instead you submit a **PLAN**: a sequence of
actions from a fixed menu that together must reproduce `x_T` exactly, and
each action has a checker-computed multiplier cost. Minimize the total cost.

## Input (stdin)
```
S K T p m
x0[0] ... x0[S-1]
```
then, for each of the `K` recipe blocks: `S` rows of `S` integers (`A_k`),
then one row of `S` integers (`b_k`). Then one line with `p` integers
(`pattern[0..p-1]`, each in `[0,K)`). Then, for each of the `m` recalibration
days (given in strictly increasing day order): the day `t`, then `S` rows of
`S` integers (`G`), then one row of `S` integers (`h`).

## Output (stdout)
```
N
<N action lines>
```
Each action line is one of:
```
STEP P k     apply recipe k densely for one day        cost S*S
STEP O idx   apply recalibration idx (0-based, in the   cost S*S
             day order given in the input) for one day
BAND k       apply recipe k using only its nonzero       cost = (# nonzero
             (banded) entries for one day                  entries of A_k)
BUILD        cache the composed one-period map            cost p*S^3
PERIOD n     apply the cached period map n>=1 times       cost 2*bits(n)*S^3
             in a row (bits(n) = n's bit length)
```
Actions execute in order against a day counter starting at 0. `STEP O idx` is
only valid when the recalibration day at that index equals the current day
counter. `BAND` is only valid on a rotation day (never on a recalibration
day). `PERIOD n` requires a prior `BUILD`, requires the current day counter to
be a multiple of `p`, and requires no recalibration day inside the `n*p`-day
span it advances over.

## Feasibility
Replay the plan (exact arithmetic mod `P`) to get a final state and a final
day counter. The plan is feasible iff: every action is well-formed and valid
per the rules above, the day counter equals `T` exactly at the end, and the
replayed final state equals the true `x_T` (computed independently by the
checker). Any violation, parse error, non-integer token, or out-of-range
value scores 0.

## Objective
Minimize `F`, the sum of the charged costs above.

## Scoring
Let `B = T*S*S` (the cost of a plan using `STEP` for every single day, the
naive fully-dense simulation). `Ratio = min(1, 0.1 * B / F)`.

## Constraints
`1 <= S <= 8`, `1 <= K <= 4`, `5 <= T <= 3000`, `1 <= p <= 8`,
`0 <= m <= 6`, `1 <= N <= 200000`. Time limit 5s, memory 512MB.

## Example
Suppose `S=3,T=4,p=2,m=0,pattern=[0,1]`, and every `A_k` has 5 nonzero
entries. The dense baseline is `B = 4*9 = 36`. A plan using `BAND` every day
costs `4*5=20` (`Ratio=0.18`). A plan that first fits one full period
(`BUILD` cost `2*27=54`, then `PERIOD 2` cost `2*2*27=108` — worse here since
`T` is tiny) illustrates that `BUILD`/`PERIOD` only pay off once a rotation
run is long enough; this worked example is illustrative arithmetic only, not
a hint about which choice wins on the actual test data.
