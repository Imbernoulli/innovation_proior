# The Kalman Filter

## Problem it solves

Estimate, in real time, the unobserved state of a linear dynamic system driven by white noise, from a stream of noisy and possibly partial measurements — optimally with respect to mean-squared error — handling time-varying and nonstationary dynamics with a single method, and cheaply enough to run online on a computer. It is the time-domain, recursive, state-space replacement for the Wiener filter, which is stationary, frequency-domain, specified as an impulse response, and poorly suited to machine computation.

## Key idea

Model the world in state-space:
- dynamics: `x(t+1) = Φ(t+1;t) x(t) + u(t)`, with `u` white noise, zero mean, covariance `Q(t)`;
- observation in the projection derivation: `y(t) = M(t) x(t)`, with `M` possibly singular; direct measurement noise can be represented by augmenting the state or, in the predict/update implementation, by keeping a separate covariance `R`.

For quadratic loss the optimal estimate is the conditional expectation; for the linear/Gaussian case this equals the **orthogonal projection** of `x` onto the span `Y(t)` of all observations, under the inner product `⟨u,v⟩ = E[uv]`. It depends only on first and second moments.

The new measurement `y(t)` adds to `Y(t-1)` only its **innovation** `ỹ(t|t-1) = y(t) − M(t)x*(t|t-1)` — the part orthogonal to the past — so `Y(t) = Y(t-1) ⊕ Z(t)` is an orthogonal sum and the projection updates by *adding* a correction in that one direction. This makes the optimal estimate **recursive**: carry only the current estimate and its error covariance, discard the history. The gain comes from requiring the new error to be orthogonal to the innovation; it balances trust in the model against trust in the measurement. The error covariance propagates by a matrix **Riccati** difference equation — the time-domain analogue of the Wiener–Hopf equation — computable forward from the model alone, and `trace P` is the filter's own expected loss. The same Riccati equation is, under transpose-and-time-reversal, the optimal-regulator (LQR) equation: estimation and control are **dual** (observation ↔ control, `M ↔ M̂`).

## The algorithm (one-step predictor form)

Optimal predictor, written first in innovation form:

- innovation `ỹ(t|t-1) = y(t) − M(t)x*(t|t-1)`
- estimate `x*(t+1|t) = Φ(t+1;t)x*(t|t-1) + ∆*(t)ỹ(t|t-1)`
- equivalent dynamic-system form `x*(t+1|t) = Φ*(t+1;t)x*(t|t-1) + ∆*(t)y(t)`

- gain `∆*(t) = Φ(t+1;t) P*(t) M'(t) [M(t) P*(t) M'(t)]⁻¹`
- estimator transition `Φ*(t+1;t) = Φ(t+1;t) − ∆*(t) M(t)`
- error covariance `P*(t+1) = Φ(t+1;t){ P*(t) − P*(t)M'(t)[M(t)P*(t)M'(t)]⁻¹ M(t)P*(t) }Φ'(t+1;t) + Q(t)`
- initialize `P*(t0) = E[x(t0)x'(t0)]`; expected quadratic loss `= trace P*(t)`.

Equivalently, for a numerical predict/update implementation, split each cycle into a **time update** (predict) and a **measurement update** (correct), carrying state `x` and covariance `P`:

**Predict (time update):**
- `x ← F x`
- `P ← F P F' + Q`

**Update (measurement update), with measurement `z`, measurement matrix `H`, measurement-noise covariance `R`:**
- innovation `y = z − H x`
- innovation covariance `S = H P H' + R`
- gain `K = P H' S⁻¹ = P H'(H P H' + R)⁻¹`
- correct `x ← x + K y`
- covariance `P ← (I − K H) P (I − K H)' + K R K'` (Joseph form; symmetric, PSD, robust to non-optimal `K`)

Here `F = Φ`, `H = M`, and the `P` in the update equations is the covariance after prediction and before the measurement correction. With the measurement noise folded into the state, the filtered-form gain relates to the one-step predictor gain by `∆* = ΦK`; with direct measurement noise kept separate, the same balance appears as the extra `R` term in `S`.

## Working code

```python
import numpy as np
from copy import deepcopy
from math import log
import sys
from numpy import dot, zeros, eye, isscalar

def reshape_z(z, dim_z, ndim):
    z = np.asarray(z)
    if z.ndim == 0:
        z = z.reshape(1, 1)
    return z.reshape(dim_z, 1) if ndim == 2 else z.reshape(dim_z)

class KalmanFilter(object):
    """Recursive state estimator for a linear dynamic system."""

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))           # state
        self.P = eye(dim_x)                  # uncertainty covariance
        self.Q = eye(dim_x)                  # process uncertainty
        self.B = None                        # control transition matrix
        self.F = eye(dim_x)                  # state transition matrix
        self.H = zeros((dim_z, dim_x))       # measurement function
        self.R = eye(dim_z)                  # measurement uncertainty
        self._alpha_sq = 1.0                 # fading memory control
        self.M = np.zeros((dim_x, dim_z))    # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        self.K = np.zeros((dim_x, dim_z))    # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty
        self._I = eye(dim_x)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None
        self.inv = np.linalg.inv

    def predict(self, u=None, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
```

## Why it matters

One state-space projection recursion subsumes filtering, prediction, and smoothing; stationary and nonstationary statistics; growing/infinite memory; and partial observation — the cases the Wiener theory handled only by separate, opaque derivations. Forward prediction is immediate from the transition matrix; present-time filtering is recovered from the one-step predictor when the transition is invertible; smoothing earlier states uses the same projection framework with extra coordinates for the unobserved pieces. The estimator needs only first and second moments, runs in fixed memory with fixed matrix work per step, and carries a running estimate of its own accuracy.
