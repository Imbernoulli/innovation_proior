It is past eleven and the train has been sitting outside Baltimore for the better part of an hour. I have a headache and nothing to do but think about the filtering problem, which has been bothering me for weeks. Let me think it through from the pain, because the pain is sharp and specific.

The accepted way to do optimal estimation is Wiener's. Signal and noise are random processes, I observe their sum, and I want the minimum-mean-square estimate of the signal. Wiener showed this reduces to the Wiener–Hopf integral equation, and for stationary processes with rational spectra you solve it by spectral factorization and read off the optimal filter. It is beautiful and it is, for my purposes, almost useless. The optimal filter comes out as an impulse response, and turning an impulse response into an actual realizable filter is not a simple task. Computing that impulse response is involved and gets rapidly worse as the problem grows — it is poorly suited to a machine, and a machine is exactly what I want to use. Every time I change the setting — finite memory, growing memory, nonstationary statistics — I have to do a whole new derivation, each one of considerable difficulty. And the derivations are opaque: the assumptions and their consequences are buried in the spectral manipulations. Four separate complaints, but they have one root. The whole apparatus is built in the frequency domain, for stationary processes. A stationary process has one spectrum, valid for all time. The moment the system is time-varying — and every real system I care about is time-varying — the machinery has nothing to grab onto.

So the question is not "how do I solve the Wiener–Hopf equation better." The question is: what would I have to change so that nonstationarity, time-variation, partial observation, and machine computation all stopped being special cases requiring new derivations, and became automatic? I keep coming back to the same thing I have been chewing on for years, ever since Ragazzini's sampled-data paper: there is no fundamental difference between continuous and discrete linear systems, and the natural way to describe either is by its *state*. Not by a transfer function, not by a spectrum — by the state, the least information about the past you need to predict the future, evolving by first-order equations. In control I describe a plant as `x(t+1) = Φ(t+1;t) x(t) + …`, `y(t) = M(t) x(t)`, and `Φ` is allowed to depend on `t` without any fuss at all. Time-variation is free in state-space. It is only a curse in the frequency domain.

Why not apply the notion of state variables to the Wiener filtering problem?

That is the thought. Let me see whether it actually buys anything or whether I am just renaming the difficulty.

First I need a state-space description of the random signal, not a spectrum. Here Bode and Shannon already did the essential thing, even if they wrapped it in stationary language: any random signal, up to its second-order statistics, can be represented as the output of a linear dynamic system driven by white noise. White noise — uncorrelated from instant to instant, zero mean — is the only random input I need, and the dynamics shape it into the correlated signal I observe. If direct measurement noise is present, I can either include the noise-producing coordinates in the state or keep their covariance as a separate matrix in the numerical recursion. The clean projection derivation is simplest with the first choice. So I will model the world as
`x(t+1) = Φ(t+1;t) x(t) + u(t)`, with `u` white, zero mean, covariance `Q(t)`,
and `y(t) = M(t) x(t)` for the observation. `M` can be singular — I only see some coordinates of `x`, or combinations that include noise coordinates — which is good, because that *is* the real problem: reconstruct the whole state, including the parts I never measure, from noisy partial observations. Notice I have written everything in discrete time. That is deliberate. The measurements come at discrete instants anyway, and if I keep it discrete I can keep the mathematics rigorous and elementary instead of fighting Hilbert-space subtleties; the continuous case can be approached after the discrete case is clear.

Now, what is the optimal estimate? Strip away the spectra and ask the bare probability question. I have observed `y(t0), …, y(t)`. I want `x(t1)`. Everything the data tell me about `x(t1)` is in the conditional distribution of `x(t1)` given the observations. If I penalize error quadratically, the estimate that minimizes the conditional risk is the conditional expectation `E[x(t1) | y(t0), …, y(t)]`; no Gaussian assumption is needed for that squared-loss statement. For broader symmetric loss functions the same conditional mean is still optimal when the conditional distribution has the needed symmetry and convexity, and Gaussian conditional distributions have those properties automatically. So the optimal estimate is the conditional mean in the cases I can defend. Fine, but conditional expectations are not directly computable from the limited information I want to assume — they need the full distribution, and I refuse to assume I know more than first and second moments of anything.

Here is where the headache lifts for a moment. I have been reading Loève on the train — probability theory — and there is this fact sitting there that the engineers never use. Consider the observed random variables `y(t0), …, y(t)` and all their linear combinations `∑ aᵢ y(i)`. That set is a vector space — call it `Y(t)` — and I can put an inner product on it: two random variables `u, v` are "orthogonal" if `E[uv] = 0`. With that inner product the random variables become a geometry. And the conditional expectation, in the Gaussian case, is nothing but the **orthogonal projection** of `x(t1)` onto `Y(t)`.

Let me make sure I believe that, not just recite it. Decompose `x(t1)` into the part lying in `Y(t)` and the part orthogonal to it: `x(t1) = x̂ + x̃`, where `x̂ ∈ Y(t)` and `E[x̃ · u] = 0` for every `u ∈ Y(t)`. The projection `x̂` is the linear combination of the observations closest to `x(t1)` in mean square, because for any other `w ∈ Y(t)`,
`E[(x(t1) − w)²] = E[(x̃ + (x̂ − w))²] = E[x̃²] + E[(x̂ − w)²] ≥ E[x̃²]`,
the cross term vanishing precisely because `x̃` is orthogonal to `x̂ − w ∈ Y(t)`. So among *linear* estimates the projection minimizes the squared error, always — no Gaussian assumption needed for that. And if the processes *are* Gaussian, then the orthogonal part `x̃`, having zero mean and being uncorrelated with `Y(t)`, is actually *independent* of `Y(t)`, so `E[x(t1) | Y(t)] = x̂` exactly — the projection is the full conditional mean, and no nonlinear estimate can do better. This is why I only ever need first and second moments: the whole thing runs on covariances and the projection geometry, never on the shape of the distribution. Wiener's opaque fourth complaint just evaporated — the assumptions are right out in the open: linearity, or Gaussianness, plus quadratic loss.

So the optimal estimate is `x*(t1|t) = Ê[x(t1) | Y(t)]`, the projection of `x(t1)` onto the span of all observations up to `t`. Write `Ê` for this projection-expectation. Now I have a clean object. The danger is that computing the projection looks exactly as expensive as Wiener: as `t` grows, `Y(t)` grows, and projecting onto an ever-larger space, re-solving as each measurement arrives, is just the growing-memory problem in new clothes. If state-space is going to earn its keep, the projection has to be computable *recursively* — I must be able to get `x*(t+1|t)` from `x*(t|t−1)` and the single new measurement `y(t)`, without dragging the whole past along. Otherwise I have gained nothing.

Let me look hard at how `Y(t)` grows. `Y(t)` is `Y(t−1)` plus whatever the new measurement `y(t)` adds. But `y(t)` is not entirely new — I could already partly predict it from the past. The part I *could* predict is its projection onto `Y(t−1)`, which is `M(t) x*(t|t−1)` (the predicted observation). The genuinely new information is the leftover:
`ỹ(t|t−1) = y(t) − M(t) x*(t|t−1)`,
the component of `y(t)` orthogonal to `Y(t−1)`. This residual — the discrepancy between what I measured and what I expected to measure — is the only new direction the measurement contributes. Call the space it generates `Z(t)`. And by construction `Z(t)` is orthogonal to `Y(t−1)`, so
`Y(t) = Y(t−1) ⊕ Z(t)` as an *orthogonal* sum.

That orthogonality is the whole game, and I should pause on why it matters so much. Projection onto an orthogonal direct sum splits into a sum of projections:
`Ê[x(t1) | Y(t)] = Ê[x(t1) | Y(t−1)] + Ê[x(t1) | Z(t)]`.
I do not have to re-project onto the enlarged space from scratch. I take the projection I already had onto the old space, and I *add* a correction that lives entirely in the new direction `ỹ(t|t−1)`. That is a recursion. The past collapses into the single previous estimate; the present enters only through the residual. The growing-memory problem is gone — not by approximation, but because the geometry of orthogonal increments lets the optimal estimate update additively.

Let me assemble the two terms for the prediction case `t1 = t+1`. The first term, `Ê[x(t+1) | Y(t−1)]`: from the dynamics `x(t+1) = Φ(t+1;t) x(t) + u(t)`, and since the process noise `u(t)` is white — orthogonal to everything in the past `Y(t−1)` — its projection drops out, leaving `Φ(t+1;t) x*(t|t−1)`. (More carefully, `Ê[x(t+1)|Y(t−1)] = Φ Ê[x(t)|Y(t−1)] + Ê[u(t)|Y(t−1)]`, and the last term is zero because `u(t)` is independent of the past.) That is just the model carrying the old estimate forward — pure prediction, no measurement. The second term, `Ê[x(t+1) | Z(t)]`, is a projection onto the one-dimensional-per-component space spanned by the residual, so it must be a linear map applied to that residual:
`Ê[x(t+1) | Z(t)] = ∆*(t) · ỹ(t|t−1)`
for some matrix `∆*(t)`. Putting it together,
`x*(t+1|t) = Φ(t+1;t) x*(t|t−1) + ∆*(t) [y(t) − M(t) x*(t|t−1)]`.

Stare at that. It is *exactly* the recursive-least-squares shape Gauss used: new estimate = (propagated old estimate) + (a gain) × (measured minus predicted). I had half-expected something like this, because the deterministic least-squares update has the same skeleton — correct the old answer by a gain times the new residual, never redo the whole fit. But Gauss's version was for a *static* unknown with a fixed measurement model; there was no state evolving in time, no process noise, no propagation of the unknown between observations. What state-space has given me is the dynamic, stochastic version of the same recursion: the `Φ x*` term is the part Gauss never had — the unknown *moves* between measurements, and I propagate both it and my uncertainty about it through the dynamics, then correct with the residual. The estimator is itself a linear dynamic system, of the same form as the plant, driven by the measurements.

Everything now hinges on the gain `∆*(t)`. The defining property is optimality: `x*(t+1|t)` is the projection, so the error `x(t+1) − x*(t+1|t)` must be orthogonal to all of `Y(t)`, and in particular orthogonal to the residual `ỹ(t|t−1)` that I just used. The part of `x(t+1)` already projected from `Y(t−1)` is orthogonal to this residual, so the normal equation for the new direction can be written just for the correction. The correction `∆*(t) ỹ(t|t−1)` must absorb exactly the part of `x(t+1)` correlated with the residual and no more:
`E{[x(t+1) − ∆*(t) ỹ(t|t−1)] ỹ'(t|t−1)} = 0`.
Expand:
`E[x(t+1) ỹ'(t|t−1)] = ∆*(t) E[ỹ(t|t−1) ỹ'(t|t−1)]`.
This is the projection's normal equation. To evaluate the two expectations I need the covariance of the current prediction error. Define
`P*(t) = E[x̃(t|t−1) x̃'(t|t−1)]`, where `x̃(t|t−1) = x(t) − x*(t|t−1)`.
The residual is `ỹ(t|t−1) = y(t) − M(t)x*(t|t−1) = M(t)[x(t) − x*(t|t−1)] = M(t) x̃(t|t−1)` (using `y = Mx` for the noise-in-the-state model). So its covariance is
`E[ỹ ỹ'] = M(t) P*(t) M'(t)`,
and the cross-covariance, using `x(t+1) = Φ x(t) + u(t)` with `u(t)` independent of the past error,
`E[x(t+1) ỹ'] = Φ(t+1;t) E[x̃(t|t−1) x̃'(t|t−1)] M'(t) = Φ(t+1;t) P*(t) M'(t)`.
Therefore
`∆*(t) = Φ(t+1;t) P*(t) M'(t) [M(t) P*(t) M'(t)]⁻¹`.
The inverse exists whenever `P*(t)` is positive definite and no observed component is a linear combination of the others — i.e., the measurements are not redundant.

Look at what that gain *is*, physically. `P*(t)` is how badly I trust my prediction, `M P* M'` is that prediction uncertainty projected into measurement space, and `P* M'` is how the state error shows up in the measured coordinates. In this formulation any observation noise has been folded into the state model, so the residual covariance is exactly `M P* M'`; when measurement noise is kept as a separate direct term, the same balance becomes `H P H' + R`. When my prediction is shaky relative to the measurement, the gain is large and I move the estimate hard toward what I just measured; when my prediction is sharp and the measurement is noisy, the gain is small and I mostly ignore the new data. It is an automatic, time-varying balance between trusting the model and trusting the measurement — and it falls straight out of demanding that the error be orthogonal to the residual. I did not have to posit it; the geometry chose it.

But the gain needs `P*(t)`, so I need to know how the error covariance itself evolves. This is the part that worried me — if I had to estimate `P*` from data online, I would be back in the swamp. I do not. `P*` propagates by its own deterministic recursion, computable in advance from the model alone, because the error is *itself* governed by a linear dynamic system. Subtract the estimate recursion from the true dynamics:
`x̃(t+1|t) = x(t+1) − x*(t+1|t) = Φ x(t) + u(t) − Φ* x*(t|t−1) − ∆* M x(t)`,
where I have written `Φ*(t+1;t) = Φ(t+1;t) − ∆*(t) M(t)`. Grouping terms,
`x̃(t+1|t) = Φ*(t+1;t) x̃(t|t−1) + u(t)`.
So `Φ*` is the transition matrix of the error too. If I write the covariance straight from that error equation I get `Φ* P*(t) Φ*' + Q(t)`, with the cross terms vanishing because `u(t)` is independent of the past error. The optimal-gain normal equation says `ΦP*(t)M'(t) = ∆*M(t)P*(t)M'(t)`, hence `(Φ − ∆*M)P*(t)M'(t) = Φ*P*(t)M'(t) = 0`. Transposing gives `M(t)P*(t)Φ*' = 0`, so `Φ*P*(t)Φ*' = Φ*P*(t)(Φ' − M'∆*') = Φ*P*(t)Φ'`. Thus the displayed covariance recursion is
`P*(t+1) = Φ*(t+1;t) P*(t) Φ'(t+1;t) + Q(t)`.
Substituting `Φ* = Φ − ∆* M` and the formula for `∆*`, this collapses to a single nonlinear recursion in `P*` alone:
`P*(t+1) = Φ(t+1;t) { P*(t) − P*(t) M'(t) [M(t) P*(t) M'(t)]⁻¹ M(t) P*(t) } Φ'(t+1;t) + Q(t)`.
I start it with `P*(t0) = E[x(t0) x'(t0)]`, my prior uncertainty about the initial state, and crank it forward. If `Q(t)` is positive definite, and the measured rows remain nonredundant, the iterates stay positive definite and the gain is well defined.

This equation is doing for me exactly what the Wiener–Hopf equation did for Wiener — it is the object whose solution specifies the optimal filter — except that it is an ordinary nonlinear *difference* equation in a finite matrix, which I can integrate forward step by step on a machine, instead of an integral equation I must solve by spectral factorization. Solving it is incomparably simpler. And the bracketed term is recognizable: it is the prediction covariance minus the reduction the measurement buys; the structure is a matrix Riccati recursion. (Bucy, down the hall at RIAS, keeps telling me the continuous-time Wiener–Hopf equation, under a finite-dimensional state model, is equivalent to the matrix Riccati equation — and here it is, the discrete Riccati, sitting at the heart of estimation. The estimate's expected quadratic loss is just `trace P*(t)`: the filter reports its own accuracy as it runs.)

So I have the whole thing, and it is recursive, time-domain, handles time-varying `Φ, M, Q`, handles partial observation through a singular `M`, needs only first and second moments, and runs forward on a computer. Prediction beyond the next instant is just additional forward propagation by `Φ`; filtering at the present is recovered from the one-step predictor when `Φ` is invertible; smoothing earlier states is the same projection problem but needs extra coordinates for the unobserved pieces between the measurement times. The single state-space derivation covers the cases Wiener had to treat separately, without pretending the backward case is the same one-line propagation as forward prediction. Wiener's four complaints are all gone: no impulse response to synthesize (I have a running dynamic system), the computation is a forward matrix recursion, nonstationary and growing-memory cases are automatic, and the assumptions are transparent.

One more thing nags at me, and it is too pretty to be coincidence. The gain recursion, the `Φ* = Φ − ∆* M` feedback, the Riccati equation — I have *seen* this before. When I solved the noise-free optimal regulator, the optimal control was a time-varying linear feedback `u*(t) = −∆̂*(t) x(t)`, with the feedback gain and the cost matrix generated by a matrix Riccati recursion of precisely this shape. Let me line them up. In the regulator, `x` is the state to be driven, `u` the control, `M̂` how control enters the state, running backward from a terminal time `T`. In the filter, `x` is the state to be estimated, `y` the observation, `M` how the state enters the observation, running forward from the first observation `t0`. If I take the filter recursions and replace each matrix `X(t0 + τ)` by `X̂'(T − τ)` — transpose and reverse time — the estimation recursions turn into the regulator recursions, and back again. The optimized estimation-error covariance `P*` maps to the regulator's cost matrix; the estimator gain `∆*` maps to the control gain `∆̂*`; the observation matrix `M` maps to the control matrix `M̂`. They are the *same problem*, dual under transpose-and-time-reversal. Observation and control are dual quantities. Both reduce to the same Riccati-like equation. I had suspected the two theories were related, but I had never been able to make it explicit — and now it is forced on me by the algebra. It means the regulator's Riccati arguments for convergence and closed-loop stability can be carried across under the corresponding observability and controllability hypotheses, instead of being reproved from nothing.

Now let me get this into a form I would actually run on a machine. The predictor equation folds prediction and correction into one step, but the operations are separable. Between measurements the dynamics carry the estimate and its uncertainty forward: `F` acts on `x`, and `Q` is injected into `P`. When a measurement arrives, I form the residual, compute the gain, correct the estimate, and shrink the covariance. Splitting `x*(t+1|t)` into a measurement update that produces the filtered estimate `x(t|t)` and then a time update `Φ x(t|t)` is the same recursion, reorganized so each half is a self-contained matrix operation. So: carry the state estimate `x` and the covariance `P`. Time update: `x ← F x`, `P ← F P F' + Q` (here `F = Φ`). Measurement update: residual `y = z − H x` (here `H = M`), innovation covariance `S = H P H' + R` (with direct measurement noise kept as the separate covariance `R`), gain `K = P H' S⁻¹ = P H'(H P H' + R)⁻¹`, correct `x ← x + K y`, and shrink `P`. The compact covariance shrink `P ← (I − K H) P` is correct for the exact optimal gain in exact arithmetic, but the symmetric identity `P ← (I − K H) P (I − K H)' + K R K'` is the form I want in code because it preserves symmetry and positive semidefiniteness when arithmetic is finite and `K` is not exactly optimal. The `P` in this update is the covariance after prediction and before the measurement correction; my one-step predictor gain `∆*` is just `Φ` times this measurement-update gain — the same quantity, with the propagation pulled out into the separate time-update step.

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

        # x = F x + B u
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = F P F' + Q
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

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable and works for non-optimal K.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
```
