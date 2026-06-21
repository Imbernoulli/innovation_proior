I would name this approach the synthetic control method. It is designed for settings where a single aggregate unit, such as a country, state, city, or firm, receives an intervention at a known point in time, and the goal is to estimate what would have happened to that unit in the absence of the intervention. The untreated version of the same unit is never observed, so the task is not to estimate a regression coefficient in the usual sense but to build a credible missing outcome path from the available untreated units.

The data structure is a balanced panel. Unit one is the treated unit, and it becomes treated only after period `T0`. Units two through `J+1` are the untreated donors. I assume no anticipation before `T0` and no spillover from the treated unit back into the donors; if anticipation is possible, `T0` should be moved to the first period in which a reaction can occur. The observed outcome can be written as `Y_it = Y_it^N + alpha_it D_it`, where `D_it` is one exactly when `i = 1` and `t > T0`. The object of interest is `alpha_1t = Y_1t - Y_1t^N`, the gap between the treated unit's observed outcome and its unobserved no-treatment outcome in each post-treatment period.

Because `Y_1t^N` is missing after `T0`, I construct it as a weighted combination of donor outcomes. The synthetic control is `Yhat_1t^N(W) = sum_{j=2}^{J+1} w_j Y_jt`. The weights are restricted to be nonnegative and sum to one, so the counterfactual always lies inside the convex hull of the donor pool. This restriction matters a great deal. It makes the comparison a mixture of real units rather than an extrapolation, and it lets me report exactly which donors enter and with what weights. A reader can then judge whether the synthetic unit is substantively plausible.

The weights are chosen using only pre-treatment information. I collect predictors and selected pre-treatment outcomes for the treated unit in a vector `X1`, and the corresponding donor quantities in a matrix `X0`. I also form a symmetric positive semidefinite matrix `V` that weights the different predictors. For a given `V`, I solve the constrained quadratic program `argmin_W (X1 - X0 W)' V (X1 - X0 W)` subject to `w_j >= 0` and `sum_j w_j = 1`. Expanding the objective shows a constant term in `X1`, a quadratic term `W' X0' V X0 W`, and a linear term `-2 X1' V X0 W`; only the quadratic and linear terms matter for the minimizer. In the canonical implementation, `V` is a normalized diagonal matrix whose entries are optimized to make the pre-treatment outcome fit as good as possible.

The quality of a candidate `V` is judged by a pre-treatment outcome criterion. Let `Z1` be the treated unit's pre-treatment outcomes and `Z0` the donor pre-treatment outcomes. I compute the weights `W^*(V)` from the predictor problem and then evaluate the mean squared prediction error `(Z1 - Z0 W^*(V))'(Z1 - Z0 W^*(V)) / nrow(Z0)`. The predictor weights that give the lowest pre-treatment outcome MSPE are selected. This links predictor importance directly to predictive performance on the very window whose fit determines the credibility of the design.

Once the weights are fixed, the post-treatment effect estimate is immediate: for each period after `T0`, `alphahat_1t = Y_1t - sum_{j=2}^{J+1} w_j^* Y_jt`. A positive value means the treated unit is above its estimated no-treatment path, and a negative value means it is below. The sign of the estimate matches the sign of the estimand by construction.

The pre-treatment fit is the central diagnostic. If the weighted donors cannot reproduce the treated unit's outcome history before the intervention, there is little reason to trust the synthetic path after the intervention. A poor fit is visible in the data rather than hidden behind a coefficient table. I also examine placebo exercises: I pretend in turn that each donor was treated at the same time and compute the gap that the synthetic control would have produced. These placebo gaps provide a reference distribution for judging whether the post-treatment gap for the treated unit is unusually large.

The method is not a black-box estimator. The donor weights, the predictor weights, the predictor balance, the pre-treatment outcome fit, the post-treatment gaps, and the placebo comparisons should all be reported. Threats to the design include weak pre-treatment fit, a treated unit that lies outside the convex hull of the donors, donor units that are themselves affected by the intervention, anticipation before the formal start date, and excessive dependence on a single donor. These are design issues, not minor robustness checks.

A key advantage of using a long pre-treatment outcome window is that observed covariates alone are rarely enough. Untreated outcomes may depend on unobserved unit-specific features whose effects evolve over time. If the weighted donors track the treated unit's outcome path for many periods, they are indirectly matching the latent determinants that drive that path, not just the measured predictors. The pre-treatment trajectory becomes evidence about hidden confounders, which is why the method is more credible than a simple parallel-trends comparison based on a single donor or an unweighted average.

In practice, the canonical implementation uses a `dataprep` step to build `X1`, `X0`, `Z1`, `Z0`, and the outcome series, followed by a `synth` optimization step. Predictors are scaled by their standard deviations before optimization. The inner donor-weight problem is solved as a quadratic program with an equality constraint and box constraints. The outer search over predictor weights uses standard optimization routines, and the selected predictor weights are normalized to be nonnegative and sum to one. The final gaps are computed by subtracting the weighted donor outcomes from the treated unit's outcomes in the plot period.

The synthetic control method is most appropriate when there is a single treated aggregate unit, a reasonably sized donor pool, a long pre-treatment period, and an intervention whose timing is well defined. It turns the counterfactual estimation problem into a transparent matching exercise in which the comparison object is a weighted average of real donors, selected to reproduce the treated unit before the event. The post-treatment gap is then interpreted as the effect of the intervention, conditional on the design being credible.

```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(0)

n_donors = 4
n_pre = 20
n_post = 10
common = np.linspace(0, 5, n_pre + n_post)

Y_donors = np.column_stack([
    common + np.random.randn(n_pre + n_post) * 0.3 for _ in range(n_donors)
])

true_w = np.array([0.4, 0.3, 0.2, 0.1])
Y1_N = Y_donors @ true_w + np.random.randn(n_pre + n_post) * 0.2

Y1 = Y1_N.copy()
Y1[n_pre:] += 1.5

X1 = np.array([Y1[:n_pre].mean(),
               np.polyfit(np.arange(n_pre), Y1[:n_pre], 1)[0]])
X0 = np.vstack([
    [Y_donors[:n_pre, j].mean(),
     np.polyfit(np.arange(n_pre), Y_donors[:n_pre, j], 1)[0]]
    for j in range(n_donors)
])

sd = np.std(np.vstack([X1, X0]), axis=0, ddof=1)
X1s, X0s = X1 / sd, X0 / sd

H = X0s @ X0s.T
c = -X0s @ X1s

def objective(w):
    return 0.5 * w @ H @ w + c @ w

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * n_donors
result = minimize(objective, x0=np.ones(n_donors) / n_donors,
                  method="SLSQP", bounds=bounds, constraints=constraints,
                  options={"ftol": 1e-12})
w_hat = result.x

gaps = Y1[n_pre:] - Y_donors[n_pre:] @ w_hat
print("Estimated donor weights:", np.round(w_hat, 3))
print("Mean post-treatment gap:", np.round(gaps.mean(), 3))
print("True treatment effect: 1.5")
```
