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
  Prediction always inflates uncertainty.
- **Update (measurement update):** fold in `z_k` by weighting the prediction against the measurement
  by their inverse covariances (precisions). The correction is the *innovation* `z_k − H x̂⁻` (the
  part of the measurement not already predicted) scaled by the Kalman gain; the covariance shrinks.

The gain `K = P⁻ H'(H P⁻ H' + R)⁻¹` is what minimizes the posterior error covariance. It equals the
orthogonal projection of the state onto the span of the observations (the error is left uncorrelated
with every measurement), which in the Gaussian case is exactly the conditional mean — hence the
globally minimum-mean-square-error estimate. Without Gaussianity it is still the best *linear* (linear-MMSE)
estimator. Limits make the weighting transparent: `R→0` gives `K→H⁻¹` (trust the measurement);
`P⁻→0` gives `K→0` (trust the prediction).

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

The error covariance satisfies the Riccati-like recursion obtained by eliminating `K`:
`P⁻_{k+1} = F { P − P H'(H P H' + R)⁻¹ H P } F' + Q`. The expected squared error per step is
`trace P`. For numerical robustness the covariance update is implemented in **Joseph form**,
`P = (I − K H) P⁻ (I − K H)' + K R K'`, which stays symmetric positive semidefinite even for a
non-optimal `K` and under roundoff (it reduces to `(I − K H) P⁻` at the optimal gain).

## Code

Faithful to the `filterpy` `KalmanFilter` (predict / update with Joseph-form covariance).

```python
import numpy as np

class KalmanFilter:
    """Recursive linear-Gaussian state estimator.
    Model: x_{k+1} = F x_k + B u_k + w,  z_k = H x_k + v,
           w ~ N(0, Q),  v ~ N(0, R)."""

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F, self.H, self.Q, self.R = F, H, Q, R   # dynamics, measurement, noise covariances
        self.B = B                                    # optional control matrix
        self.x = x0                                   # state estimate (belief mean)
        self.P = P0                                   # error covariance (belief spread)
        self.I = np.eye(F.shape[0])

    def predict(self, u=None):
        # Time update: x̂⁻ = F x̂ (+ B u),  P⁻ = F P F' + Q
        if self.B is not None and u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Measurement update.
        y = z - self.H @ self.x                       # innovation z - H x̂⁻
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance H P⁻ H' + R
        K = self.P @ self.H.T @ np.linalg.inv(S)      # gain P⁻ H' S⁻¹
        self.x = self.x + K @ y                       # x̂ = x̂⁻ + K y
        I_KH = self.I - K @ self.H                    # Joseph-form covariance update
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


# Example: constant-velocity tracking. State [position, velocity]; measure position only.
if __name__ == "__main__":
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])      # x_{k+1} = position + velocity*dt; velocity constant
    H = np.array([[1.0, 0.0]])           # measure position only
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    R = np.array([[1.0]])                # measurement-noise variance
    x0 = np.array([[0.0], [1.0]])        # initial guess: at 0, moving at 1
    P0 = np.eye(2) * 10.0                # large initial uncertainty

    kf = KalmanFilter(F, H, Q, R, x0, P0)
    rng = np.random.default_rng(0)
    true_x = np.array([[0.0], [1.0]])
    for _ in range(20):
        true_x = F @ true_x                                  # true state evolves
        z = H @ true_x + rng.normal(0, 1.0, size=(1, 1))     # noisy position measurement
        kf.predict()
        kf.update(z)
        print(f"est pos {kf.x[0,0]:6.2f}  vel {kf.x[1,0]:5.2f}  "
              f"trace(P) {np.trace(kf.P):6.3f}")
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
proportion to its certainty; prediction inflates the covariance by `Q`, and each measurement shrinks
it, the steady balance of the two giving the constant-gain limit when `F, H, Q, R` are time-invariant.
