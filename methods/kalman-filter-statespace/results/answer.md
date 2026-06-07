# The Kalman filter: recursive optimal linear estimation of a dynamic state

## Problem

A linear-Gaussian system evolves as
`x_{k+1} = F x_k + B u_k + w_k`, `z_k = H x_k + v_k`, with independent white noise
`w_k ~ N(0, Q)` and `v_k ~ N(0, R)`. The state `x` is hidden; only the noisy, possibly partial
measurements `z_k` are seen. Estimate `x_k` recursively (online, one measurement at a time, no
batch re-processing), with no stationarity assumption, while also reporting the estimate's own
error covariance. This is the estimation dual of the linear-quadratic regulator (both reduce to the
same Riccati recursion); here the unknown is the state, not the control.

## Key idea

Carry a Gaussian belief over the state — mean `x̂` and covariance `P`. Two moments are a sufficient
statistic, because a Gaussian stays Gaussian under a linear map plus independent Gaussian noise, so
the recursion is exact, not approximate. Alternate two operations:

- **Predict (time update):** roll the belief through the dynamics before seeing data. The mean goes
  through `F`; the covariance of the linear image is `F P F'`, plus the process-noise covariance `Q`.
  The fresh process noise `Q` only adds to the propagated covariance, never subtracts.
- **Update (measurement update):** fold in `z_k` by weighting the prediction against the measurement
  by their inverse covariances (precisions). The correction is the *innovation* `z_k − H x̂⁻` (the
  part of the measurement not already predicted) scaled by the Kalman gain; the covariance never
  increases (and strictly shrinks wherever the measurement is informative).

The gain `K = P⁻ H'(H P⁻ H' + R)⁻¹` is what minimizes the posterior error covariance. It gives the
orthogonal projection of the state onto the span of the observations (the error is left uncorrelated
with every measurement), which in the Gaussian case is exactly the conditional mean — hence the
globally minimum-mean-square-error estimate. Without Gaussianity it is still the best *linear* (linear-MMSE)
estimator. Limits make the weighting transparent: if the measured coordinates determine the state
(`H` square and nonsingular), `R→0` gives `K→H⁻¹` (trust the measurement); `P⁻→0` gives `K→0`
(trust the prediction).

## The algorithm

State: estimate `x̂` and error covariance `P`. Initialize with prior `x̂_0`, `P_0`.

**Predict**

    x̂⁻ = F x̂ (+ B u)
    P⁻ = F P F' + Q

**Update** (on measurement `z`)

    y = z − H x̂⁻                      # innovation / residual
    S = H P⁻ H' + R                   # innovation covariance
    K = P⁻ H' S⁻¹                     # Kalman gain
    x̂ = x̂⁻ + K y                     # corrected mean
    P = (I − K H) P⁻                  # corrected covariance (optimal-gain short form)

Eliminating `K` gives the Riccati-like covariance recursion in terms of the prior covariance:
`P_{k+1|k}=F { P_{k|k-1} − P_{k|k-1}H'(H P_{k|k-1}H' + R)⁻¹ H P_{k|k-1} } F' + Q`.
The expected squared error per step is `trace P`. For numerical robustness the covariance update is
implemented in **Joseph form**, `P = (I − K H) P⁻ (I − K H)' + K R K'`, a sum of two congruence terms
that is symmetric positive semidefinite for any `K` in exact arithmetic and resists drifting asymmetric
or indefinite under roundoff (it reduces to `(I − K H) P⁻` at the optimal gain).

## Code

This mirrors the local FilterPy predict/update equations: predict uses `x = F x + B u`, `P = FPF' + Q`;
update uses the innovation `z-Hx`, `S=HPH'+R`, `K=PH'S⁻¹`, and the Joseph-form covariance.

```python
import numpy as np
from numpy.linalg import inv

class LinearGaussianModel:
    """x_{k+1} = F x_k + B u_k + w,  z_k = H x_k + v,  w~N(0,Q), v~N(0,R)."""
    def __init__(self, F, H, Q, R, B=None):
        self.F, self.H, self.Q, self.R, self.B = F, H, Q, R, B

class StateBelief:
    """Gaussian belief summarized by mean x and covariance P."""
    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0

def time_update(belief, model, u=None):
    # Predict: x^- = F x (+ B u), P^- = F P F' + Q.
    belief.x = model.F @ belief.x
    if u is not None and model.B is not None:
        belief.x = belief.x + model.B @ u
    belief.P = model.F @ belief.P @ model.F.T + model.Q

def measurement_update(belief, model, z):
    # Update: residual, innovation covariance, gain, corrected mean, Joseph covariance.
    H, R = model.H, model.R
    y = z - H @ belief.x
    S = H @ belief.P @ H.T + R
    K = belief.P @ H.T @ inv(S)
    belief.x = belief.x + K @ y
    I_KH = np.eye(belief.P.shape[0]) - K @ H
    belief.P = I_KH @ belief.P @ I_KH.T + K @ R @ K.T

def run_filter(model, belief, measurements, controls=None):
    estimates = []
    for k, z in enumerate(measurements):
        time_update(belief, model, None if controls is None else controls[k])
        measurement_update(belief, model, z)
        estimates.append((belief.x.copy(), belief.P.copy()))
    return estimates
```

## Why it is optimal

In the linear-Gaussian model the carried belief `N(x̂_k, P_k)` is exactly the posterior
`p(x_k | z_{1:k})`: predict propagates a Gaussian through linear dynamics, update multiplies by the
Gaussian measurement likelihood, and the product of Gaussians is Gaussian. Since two moments fully
describe a Gaussian, no information is discarded — `x̂` is the conditional mean, the
minimum-mean-square-error estimate among *all* estimators. Drop Gaussianity but keep first and second
moments, and the same recursion computes the orthogonal projection of the state onto the observation
span — the best *linear* estimator. A nonlinear estimator can improve on it only when the process is
non-Gaussian *and* third-or-higher-order statistics are available. The gain weights the two
information sources (prediction and measurement) by their inverse covariances, so each is trusted in
proportion to its certainty; prediction adds the process-noise covariance `Q` while a measurement never
increases the covariance, the steady balance of the two giving the constant-gain limit when `F, H, Q, R` are time-invariant.
