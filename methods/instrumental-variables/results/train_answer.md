I want the causal effect of a treatment, exposure, or regressor `D` on an outcome `Y`, and the obstacle is that `D` was never assigned as if by experiment. People select it, institutions target it, markets settle it jointly with `Y`, and measurement error distorts it. Written structurally as `Y = alpha + tau D + U`, the contamination is exactly `Cov(D, U) != 0`: the disturbance `U` still carries omitted ability, demand, severity, preference, or institutional information, so `D` moves together with `U` rather than independently of it. A regression of `Y` on `D` then uses the very movement in `D` that travels with `U`, and it cannot separate causal response from selection. Naive differences in means absorb all selection into the estimate; ordinary least squares with controls, matching, and propensity-score adjustment remove only the confounding that is observed and correctly modeled, leaving anything unobserved inside `U`. Intention-to-treat comparisons can be credible for an assignment itself but do not recover the effect of the treatment when compliance is partial. What is missing is a way to split `D` into a usable part and a contaminated part, and to say which movement in `D` is informative about the target equation.

I propose the instrumental-variables estimator. The idea is to stop treating all of `D` as evidence and instead find an external source `Z` — an instrument — that moves `D` but reaches `Y` through no path except that induced movement in `D`. The supply-demand geometry shows why this works: if weather or crop conditions shift supply while staying outside the demand equation, the induced movement in equilibrium price traces out the demand curve rather than the tangle of both curves. The instrument earns its keep not by being interesting in itself but by moving the suspect regressor from outside the target equation. To turn this into an estimator, take the covariance of `Z` with both sides of `Y = alpha + tau D + U`, which gives `Cov(Z, Y) = tau Cov(Z, D) + Cov(Z, U)`. The design is engineered to drive the last term to zero — that is the exclusion and independence claim — while keeping `Cov(Z, D)` away from zero — that is relevance. When it succeeds, the slope becomes visible as `tau = Cov(Z, Y) / Cov(Z, D)`. This ratio is not a mechanical trick. The numerator is the reduced-form movement in the outcome when the external source varies; the denominator is the first-stage movement in the contaminated regressor when that same source varies; dividing asks how much the outcome moves per unit of regressor movement that comes from the external source. For a binary instrument the same object reads as the Wald ratio, `{E[Y | Z = 1] - E[Y | Z = 0]} / {E[D | Z = 1] - E[D | Z = 0]}`, the outcome contrast by `Z` rescaled by the amount of treatment that `Z` actually changed.

What this ratio names depends on whether effects are constant. If every unit shares the same treatment effect, the ratio is the common causal slope. But constant effects are not the general case, and once effects differ I have to track whom the instrument moves. Partition units into always-takers, who receive treatment regardless of `Z`; never-takers, who never do; compliers, who move with `Z`; and defiers, who move against it. If I allow both compliers and defiers, the first-stage denominator can hide offsetting movements and the reduced form can combine effects with negative weights, so the ratio means nothing clean. The load-bearing fix is monotonicity: assume the instrument never pushes some units toward treatment and others away. Then the first stage counts exactly the people whose treatment status changes, the reduced form counts their average causal effect times that count, and the ratio identifies the local average treatment effect for the compliers — the units the instrument moves. This local reading is the honest estimand the design creates, not a fallback after the algebra fails. A draft lottery identifies the effect for those whose service status the lottery changes; a compulsory-attendance rule identifies the effect for those whose schooling it changes. I cannot silently rename that as the average effect for everyone without stronger assumptions.

The same source-of-movement logic extends to controls and to many instruments without changing its character. With predetermined covariates `W`, I residualize `Y`, `D`, and `Z` on `W` — removing the linear part each shares with `W` — and then divide reduced form by first stage on the residuals, `tau_hat = (z_r' y_r) / (z_r' d_r)`. This isolates residualized treatment movement; it does not pretend the whole treatment became clean. For a full linear model, collect the included exogenous controls and the endogenous treatment as `X = [W, D]`, and the controls plus excluded instruments as `Z = [W, Z_ex]`. The projection `P_Z = Z (Z'Z)^{-1} Z'` keeps the part of the regressors that lives in the exogenous instrument space, and `beta_hat = (X' P_Z X)^{-1} X' P_Z Y`. This is two-stage least squares, the standard computational form of the instrumental-variables estimator: project the endogenous regressor onto the controls and excluded instruments, then run the outcome equation on that instrument-induced component. But the projection is only the computational form; the discovery is that only externally induced movement in the contaminated regressor is permitted to carry the causal comparison.

Two failure modes therefore belong to the method, not outside it. Exclusion is a design claim the formula cannot prove — if `Z` touches `Y` directly or through another omitted cause, `Cov(Z, U)` is not zero and the ratio imports that direct path. And relevance must be strong in practice: a small denominator is not a minor inconvenience, because a weak first stage biases finite-sample instrumental-variables estimates toward the confounded ordinary-least-squares association and corrupts conventional inference. So the first stage, the reduced form, the exclusion argument, the complier population, and weak-identification diagnostics are all part of the estimator I report, not optional decoration.

```python
import numpy as np


def add_constant(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(x.shape[0]), x])


def residualize(v, controls=None):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if controls is None:
        out = v - v.mean(axis=0)
    else:
        c = add_constant(controls)
        out = v - c @ np.linalg.lstsq(c, v, rcond=None)[0]
    return out.squeeze()


def wald_iv(y, treatment, instrument, controls=None):
    y_r = residualize(y, controls)
    d_r = residualize(treatment, controls)
    z_r = residualize(instrument, controls)
    first_stage = float(z_r @ d_r)
    if np.isclose(first_stage, 0.0):
        raise ValueError("The instrument has no usable first stage.")
    return float((z_r @ y_r) / first_stage)


def two_stage_least_squares(y, endogenous, instruments, controls=None):
    y = np.asarray(y, dtype=float)
    d = np.asarray(endogenous, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    z_excluded = np.asarray(instruments, dtype=float)
    if z_excluded.ndim == 1:
        z_excluded = z_excluded[:, None]

    w = np.ones((y.shape[0], 1)) if controls is None else add_constant(controls)
    x = np.column_stack([w, d])
    z = np.column_stack([w, z_excluded])

    ztz_inv = np.linalg.pinv(z.T @ z)
    pz_x = z @ (ztz_inv @ (z.T @ x))
    beta = np.linalg.pinv(pz_x.T @ x) @ (pz_x.T @ y)
    return beta


if __name__ == "__main__":
    np.random.seed(0)
    n = 5000
    alpha, tau = 1.0, 2.0
    gamma, delta = 0.7, 0.7
    z = np.random.binomial(1, 0.5, size=n)
    u = np.random.normal(0, 1, size=n)
    d = delta * z + gamma * u + np.random.normal(0, 0.5, size=n)
    y = alpha + tau * d + u + np.random.normal(0, 0.5, size=n)

    X = add_constant(d)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0][1]
    tau_iv = wald_iv(y, d, z)
    beta_2sls = two_stage_least_squares(y, d, z)

    print(f"True tau: {tau:.3f}")
    print(f"OLS slope: {beta_ols:.3f}")
    print(f"Wald IV:   {tau_iv:.3f}")
    print(f"2SLS tau:  {beta_2sls[-1]:.3f}")
```
