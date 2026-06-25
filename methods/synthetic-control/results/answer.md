# Synthetic Control

Synthetic control estimates the effect of an intervention on one treated aggregate unit by building its untreated counterfactual from a convex combination of untreated donor units.

Let units be `1, ..., J+1`. Unit `1` is treated only after period `T0`, and units `2, ..., J+1` are untreated donors. With no anticipation before `T0` and no interference from the treated unit to donors, the observed outcome is

`Y_it = Y_it^N + alpha_it D_it`,

where `D_it = 1` only when `i = 1` and `t > T0`. If outcomes can react before formal implementation, redefine `T0` as the first period in which a reaction is possible. The target for each post-treatment period is

`alpha_1t = Y_1t - Y_1t^N`,

where `Y_1t^N` is the treated unit's unobserved no-treatment outcome.

Choose donor weights

`W = (w_2, ..., w_{J+1})'`

with

`w_j >= 0` and `sum_{j=2}^{J+1} w_j = 1`.

For a given `W`, the no-treatment path is estimated by

`Yhat_1t^N(W) = sum_{j=2}^{J+1} w_j Y_jt`.

For pre-treatment predictors and selected pre-treatment outcomes, let `X1` be the treated-unit predictor vector and `X0` the donor predictor matrix. I allow a symmetric positive semidefinite distance matrix `V`; the canonical `Synth` implementation uses a normalized nonnegative diagonal `V`. For each `V`, choose donor weights by solving

`W^*(V) in argmin_W (X1 - X0 W)' V (X1 - X0 W)`

subject to `w_j >= 0` and `sum_{j=2}^{J+1} w_j = 1`.

The post-treatment effect estimate has the same sign as the estimand:

`alphahat_1t = Y_1t - sum_{j=2}^{J+1} w_j^* Y_jt`.

Thus a positive estimate means the treated outcome is above its estimated no-treatment path, and a negative estimate means it is below that path.

Implementation faithful to CRAN `Synth` 1.1-10:

- `dataprep()` builds `X1`, `X0`, `Z1`, `Z0`, `Y1plot`, and `Y0plot`; it requires one treated unit, at least two controls, a balanced panel, predictor periods, optimization periods, and plot periods.
- `X1` and `X0` are row-scaled before optimization by the sample standard deviation of each predictor row across donors and the treated unit.
- If `custom.v` is absent and there is more than one predictor, `synth()` searches over candidate predictor-weight vectors with `optimx()` and optionally `genoud`; if there is one predictor, it uses `V = 1`; if `custom.v` is supplied, it bypasses the outer search.
- In the optimized path, candidate predictor weights are evaluated as `abs(v) / sum(abs(v))`, and the selected optimizer parameter is stored as `solution.v = abs(v) / sum(abs(v))`; in the user-supplied path, `synth.R` stores `solution.v = abs(custom.v) / sum(custom.v)`.
- For each `V`, the donor-weight problem is solved by `ipop` with equality `sum_j w_j = 1` and bounds `0 <= w_j <= 1`. The quadratic program uses scaled predictors, with `H = X0' V X0` and linear term `c = -X1' V X0`; the dropped `X1' V X1` constant and positive rescaling do not change the minimizer.
- `loss.w` is the scaled predictor distance, and `loss.v = nrow(Z0)^{-1} (Z1 - Z0 W)'(Z1 - Z0 W)` is the pre-treatment outcome MSPE used to judge `V`.
- `gaps.plot()` computes `Y1plot - Y0plot %*% solution.w`, matching the sign of `alphahat_1t`.

Report the donor weights, predictor weights, predictor balance, pre-treatment outcome fit, post-treatment gaps, and placebo or permutation comparisons. Treat weak pre-treatment fit, a treated unit far from the donor convex hull, invalid donors, spillovers, anticipation, or sensitivity to one donor as threats to the design rather than as minor robustness details.
