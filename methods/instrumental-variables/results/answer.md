# Instrumental Variables

Instrumental variables estimate a causal effect when the treatment or regressor `D` is contaminated by omitted causes of the outcome `Y`. In the structural equation `Y = alpha + tau D + U`, ordinary least squares is not causal when `Cov(D, U) != 0`.

The method uses an excluded source of variation `Z` that changes `D` but has no direct path to `Y`. The identifying conditions are relevance, independence or as-if random assignment of `Z`, exclusion from the outcome equation, and, for binary-treatment local effects, monotonicity so the instrument does not move some units in opposite directions.

For one treatment and one instrument, the core estimand is:

```text
tau = Cov(Z, Y) / Cov(Z, D).
```

For a binary instrument this becomes the Wald ratio:

```text
tau =
  {E[Y | Z = 1] - E[Y | Z = 0]}
  / {E[D | Z = 1] - E[D | Z = 0]}.
```

The numerator is the reduced-form effect of the instrument on the outcome. The denominator is the first-stage effect of the instrument on treatment. With constant treatment effects, the ratio identifies the common causal slope. With heterogeneous effects and monotonic compliance, it identifies the local average treatment effect for compliers: the units whose treatment status changes because `Z` changes.

With controls `W`, residualize `Y`, `D`, and `Z` on `W` and use:

```text
tau_hat = (z_r' y_r) / (z_r' d_r).
```

For the general linear form with included exogenous regressors `W`, endogenous treatment columns `D`, and excluded instruments `Z_ex`:

```text
X = [W, D]
Z = [W, Z_ex]
P_Z = Z (Z'Z)^(-1) Z'
beta_hat = (X' P_Z X)^(-1) X' P_Z Y.
```

This is two-stage least squares: first project the endogenous regressor on the exogenous controls and excluded instruments, then estimate the outcome equation using that instrument-induced component.

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
```

Report the first stage, reduced form, uncertainty, instrument validity argument, and the population moved by the instrument. Weak first stages make the ratio unstable and can pull finite-sample IV estimates toward the confounded OLS association, so weak-identification diagnostics are part of the method rather than an optional add-on.
