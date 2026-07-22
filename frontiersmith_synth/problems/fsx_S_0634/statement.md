# Gear Train Ratio Cutting

## Problem
You are cutting **G-stage gear trains**: G meshing pairs connected in series. Stage `i` is a
pair of integer tooth counts `(a_i, b_i)`, driver `a_i` meshing into driven `b_i`, both required
to satisfy `Tmin <= a_i, b_i <= Tmax`. A single train's transmission ratio is the **product** of
its stage ratios:
```
R = (a_1/b_1) * (a_2/b_2) * ... * (a_G/b_G)
```

You are given `K` **target** ratios to hit, each specified as an exact fraction `P/Q` (`P`, `Q`
are coprime positive integers -- `P/Q` is a high-precision rational approximation of some real
constant, e.g. pi, e, sqrt2, ...). For each target you must cut your **own separate** G-stage
train (fresh `a_i, b_i` choices) to approximate it.

Cutting big teeth costs metal. Each target also carries a cost weight `lambda >= 0` telling you
how much that train's total tooth count (the sum of all `a_i + b_i` across its `G` stages) is
penalized relative to how precisely it hits the target. Larger `lambda` means: sacrifice
precision to save teeth; near-zero `lambda` means: chase precision, cost barely matters.

## Input (stdin)
```
G Tmin Tmax K
P_1 Q_1 lambda_1
...
P_K Q_K lambda_K
```
All integers except `lambda_k` (a decimal). `1 <= Tmin < Tmax`. `G` is the number of stages,
the same for every target. Every `P_k, Q_k > 0` and coprime; every `lambda_k >= 0`.

## Output (stdout)
For `k = 1..K`, in order, print `G` lines `"a_i b_i"` -- the tooth counts of target `k`'s train,
stage 1 first. Total = `K*G` lines. Nothing else.

## Feasibility
Every printed `a_i, b_i` must be an integer with `Tmin <= value <= Tmax`. Exactly `K*G` pairs
must appear (a wrong token count, a non-integer token, or an out-of-range tooth count scores
`Ratio: 0.0`).

## Objective
For target `k` let `V_k` be the product of the `G` stage ratios you printed for it (computed
**exactly** as a rational number), and `cost_k` the sum of all `2G` printed teeth for that
target. Define
```
relerr_k  = |V_k - P_k/Q_k| / (P_k/Q_k)
contrib_k = relerr_k + lambda_k * cost_k / (G * (Tmin + Tmax))
```
Minimize `F = sum over k of contrib_k`.

## Scoring
The checker builds its own trivial train per target -- every stage set to `(mid, mid)` with
`mid = floor((Tmin+Tmax)/2)`, i.e. ratio `1` regardless of the target -- and sums its
contributions to get a baseline `B`. Then
```
Ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
printed via `Ratio: <float>`. Reproducing the baseline's strategy scores about `0.1`; a train
that is 10x better (in the combined error+cost sense) hits the `1.0` cap.

## Constraints
`2 <= G <= 4`, `4 <= Tmin < Tmax <= 60`, `1 <= K <= 8`, `0 <= lambda_k <= 1`. `Q_k` can be up to
about `2*10^6` -- far beyond what any bounded-teeth product can reduce to exactly, so you are
always approximating, never matching exactly. Time limit 5s, memory 512m, each input file is
under 5MB.

## Example (illustrative FORM only, single stage)
`G=1, Tmin=5, Tmax=8, K=1`, target `P=13, Q=8` (`= 1.625`), `lambda=0`. `mid = floor(13/2) = 6`,
so the baseline train is stage `(6,6)`: `V=1`, `relerr = |1-1.625|/1.625 = 0.384615`; since
`lambda=0` cost is irrelevant, so `contrib = B = 0.384615`.
- Printing `"6 8"`: `V = 6/8 = 0.75`, `relerr = |0.75-1.625|/1.625 = 0.538462` -- worse than the
  baseline, `Ratio < 0.1`.
- Printing `"8 5"`: `V = 8/5 = 1.6`, `relerr = |1.6-1.625|/1.625 = 0.015385`, about 25x smaller
  than the baseline's error -- `Ratio` hits the `1.0` cap.

With `G >= 2` the same idea applies per stage, but now the `G` stage ratios must be chosen
**jointly**: rounding each stage in isolation to look locally good can leave a remainder that the
other stages simply cannot reach, while a globally chosen combination -- and, when `lambda` is
large, a combination that also trades a little precision for much cheaper teeth -- does far
better.
