# The Four-Hour Trajectory: Volatility-Scheduled Fragrance Blending

## Problem
You are formulating a fragrance from `K` candidate ingredients. Each ingredient `i` has a
**scent descriptor** `desc_i` (its intensity-per-unit-concentration on 4 fixed odor axes:
citrus, floral, woody, musk), an **evaporation rate** `k_i` (its volatility class, in 1/hour
-- large `k_i` = a fast-evaporating top note, small `k_i` = a slow-evaporating base note),
and an **IFRA concentration cap** `cap_i` (a hard regulatory ceiling on how much of that
ingredient the finished product may contain). A `K x K` **masking table** `mask[i][j]` says
how strongly the raw intensity of ingredient `j` perceptually suppresses ingredient `i` when
both are present at once (louder materials can drown out quieter ones).

You choose a blend: for a subset of ingredients, an initial concentration `c_i(0)`. As the
fragrance wears, each ingredient's raw intensity decays independently on its own volatility
clock: `raw_i(t) = c_i(0) * exp(-k_i * t)` (t in hours). What a person actually perceives is
reduced by masking from everything else still present:
`perceived_i(t) = raw_i(t) / (1 + sum_{j != i} mask[i][j] * raw_j(t))`.
The scent profile at time `t` is `profile_a(t) = sum_i desc_i[a] * perceived_i(t)`.

You are given a **target profile at 5 checkpoints spanning the whole four hours**
(t = 0, 1, 2, 3, 4h) -- typically a top-note emphasis at t=0 shifting through a heart-note
emphasis around t=2 to a base-note emphasis by t=4. A blend engineered to nail the target
only at t=0 will fit beautifully at the start and then drift badly once its fast ingredients
evaporate and nothing was scheduled to carry the later hours -- your score rewards the whole
trajectory, not the first impression.

## Input (stdin)
```
K D T
desc_1[1..D] k_1 cap_1
...
desc_K[1..D] k_K cap_K
mask[1][1..K]
...
mask[K][1..K]
t_1 ... t_T
target_1[1..D]
...
target_T[1..D]
```
`D=4` axes (citrus, floral, woody, musk), `T=5` checkpoints. `mask[i][i]` is always `0`.

## Output (stdout)
```
M
idx_1 c_1
...
idx_M c_M
```
`1 <= M <= K`; each `idx_j` is a distinct 1-based ingredient index; each `c_j` is its initial
(t=0) concentration. The total blend concentration may never exceed the fixed constant
`TOTAL_CAP = 1.0` (a fragrance is diluted in a carrier; you cannot use more than "all of it").

## Feasibility
Rejected (score `0`) if: token count != `1 + 2*M`; `M` not an integer in `[1, K]`; any
`idx_j` not an integer in `[1, K]`, or repeated; any `c_j` not a finite number; any
`c_j <= 0` or `c_j > cap_{idx_j}` (violates that ingredient's IFRA cap); or
`sum_j c_j > 1.0`.

## Objective and Scoring
The checker simulates the decay+masking model above at all 5 checkpoints and computes the
mean squared error `E` between your achieved profile and the target profile (averaged over
the 5 checkpoints and the 4 axes). Define `F = 1 / (E + 0.01)` (an inverse-error goodness
score -- lower error means higher `F`). The checker also builds an internal baseline: use
*only* the first provided ingredient, at its own IFRA cap, scored by the exact SAME
procedure -> baseline error `E_base` -> `B = 1 / (E_base + 0.01)`. Then
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the baseline scores `0.1`.

## Constraints
`4 <= K <= 8`, `0.05 <= k_i <= 4.0`, `0.018 <= cap_i <= 0.88`, `0 <= mask[i][j] <= 1.6`,
target axis values in `[0.11, 0.32]`.

## Example
Suppose `K=1` (a trivial illustration -- most instances give `K>=4`): a single ingredient
with `desc=(0.9,0.1,0.1,0.1)`, `k=0.5`, `cap=0.30`, targets roughly `(0.5,0.1,0.1,0.1)` at
every checkpoint (a scent that does not change). Output `1 / 1 0.30` reproduces the checker's
own baseline construction here, so `Ratio ~= 0.1`. (Illustrative FORM only -- real instances
have a genuinely time-varying target and reward spending budget on ingredients whose decay
rate matches WHEN their axis is needed, not just how well they match at t=0.)
