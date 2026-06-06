OK, let me think this through from scratch. I have a physical system that changes over time — say something moving — and I can't see its internal variables directly. All I get is a stream of measurements, one at a time, each noisy, and usually each measurement only sees part of the system, not all of it. At every instant I want the best guess of where the system actually is, and I want it to keep improving as more data arrive. That's the whole job: turn a running stream of dirty observations into a running estimate of a hidden state.

The accepted tool for this is the Wiener filter, so let me start there and figure out exactly why it bothers me. Wiener models signal and noise as stationary random processes with known power spectra, and the optimal mean-squared-error linear filter falls out of the Wiener–Hopf integral equation, which you solve by spectral factorization in the frequency domain. Fine — it's the right answer when it applies. But look at what it costs me. The answer comes out as an impulse response, and turning an impulse response into an actual realizable filter is a separate chore. Computing that impulse response is numerically nasty and gets worse fast as the problem grows; it is not something I'd want to hand a computer to iterate. Every time I change the situation — finite data instead of an infinite record, nonstationary statistics, a growing memory — I have to go back and do a brand-new derivation, often a hard one. And the whole thing is opaque about what it's actually assuming. The stationarity, the rational spectra, the "use the entire record" — these are buried.

So the pain isn't that Wiener is wrong. It's that it's *batch*, *frequency-domain*, *stationary*, and *one-off per problem*. What I want is the opposite of every one of those: something recursive that eats one measurement at a time, lives in the time domain, doesn't care whether anything is stationary, and is a single procedure a computer iterates regardless of the specific problem. And while I'm at it, I want it to tell me how uncertain it is, and to reconstruct the hidden coordinates from partial observations.

Let me get precise about "best." I have an unknown `x(t1)` and observations `y(t0), …, y(t)`. Everything those observations tell me about `x(t1)` is in the conditional distribution `Pr[x(t1) ≤ ξ | y(t0), …, y(t)]`. Any estimate I form is some function of the observations, and I'll grade it by a loss `L(ε)` on the error `ε = x(t1) − x̂`. The natural loss is `L(0)=0`, even, non-decreasing in `|ε|`. I want to minimize the expected loss. Now, the expected loss splits: `E{L} = E[ E{L | y(t0),…,y(t)} ]`, and the outer expectation doesn't depend on my choice of `x̂` — only on the observations. So minimizing risk is the same as, for each realized set of observations, minimizing the *conditional* expected loss. Good, that localizes the problem.

Take the squared-error loss `L(ε)=ε²` first, since it's the clean one. The conditional expected loss is `E[x² | ·] − 2 x̂ E[x | ·] + x̂²`; differentiate with respect to `x̂` and set to zero: `x̂ = E[x(t1) | y(t0),…,y(t)]`. The optimal squared-error estimate is the conditional mean. And actually the same point extends beyond squared loss: if the conditional distribution is symmetric about its mean and has the convex lower-side shape used in Sherman's theorem, then any even loss that grows with `|ε|` is minimized at that same mean. Gaussian conditional distributions have exactly that symmetry and shape. So under broad conditions the thing I want is the conditional expectation. That's clean, but it's also useless as stated: in general I have no way to compute a conditional expectation as an explicit function of the data. I need structure.

Suppose I restrict myself to estimates that are *linear* functions of the observations, and keep squared loss. Consider the set of all linear combinations `∑ a_i y(i)` of the observed variables — that's a vector space, call it `Y(t)`, and I can put an inner product on it via `⟨u,v⟩ = E[u v]`. Two observations are "orthogonal" if `E[u v]=0`. Now my estimate `x̂` is supposed to be the element of `Y(t)` closest to `x` in mean square. Decompose `x` into its projection onto `Y(t)` plus a remainder: `x = x̂ + x̃`, where `x̂ ∈ Y(t)` and `x̃ ⊥ Y(t)`. For any other candidate `w ∈ Y(t)`,

  `E(x − w)² = E[(x̃) + (x̂ − w)]² = E x̃² + E(x̂ − w)² ≥ E x̃²`,

because `x̃` is orthogonal to everything in `Y(t)`, in particular to `x̂ − w`, so the cross term vanishes. Equality only when `w = x̂`. So the best linear estimate is exactly the **orthogonal projection** of `x` onto the span of the observations, and it is characterized by the *error being uncorrelated with every observation*: `E[(x − x̂) y(i)] = 0` for all `i`. That orthogonality condition is going to be my workhorse — it's a concrete equation I can actually solve, unlike "compute a conditional expectation."

But wait — projection gives me the best *linear* estimate, and a moment ago I wanted the conditional expectation, which is the best estimate among *all* functions. When do those coincide? When the process is Gaussian. If `x` and the `y`'s are jointly Gaussian, the conditional expectation is already linear in the conditioning variables, and the orthogonal part `x̃` — being orthogonal and Gaussian and zero-mean — is actually *independent* of the observations, so `E[x̃ | y] = 0` and `E[x | y] = x̂`, the projection. So in the Gaussian world the projection isn't an approximation; it *is* the conditional mean, the globally optimal estimate. And outside the Gaussian world, the projection is still the best linear estimator, and if I only know first and second moments anyway, it's the best I can justify. So I'll commit to: carry a Gaussian belief, compute the projection. The cost of restricting to linear estimators is zero when things are Gaussian, and these noise sources usually are roughly Gaussian — macroscopic noise is a superposition of many tiny effects, so the central-limit machinery pushes it toward Gaussian.

That settles *what* to compute. Now I need *how*, recursively, and for that I need a model of how `x` evolves. This is where I think the real reframing has to happen, because Wiener describes the signal by its spectrum, and a spectrum is an inherently stationary, whole-record object — it's the source of all my complaints. Let me describe the system instead by its **state**: the minimal information about the past needed to predict the future. For a linear system the state evolves by a first-order recursion,

  `x(t+1) = Φ(t+1;t) x(t) + (driving noise)`,  `y(t) = M(t) x(t)`,

where `Φ` is the transition matrix and `M` reads out the observation — possibly only some coordinates of `x`, since I said measurements are partial. The beauty of first order is that it makes no distinction between stationary and time-varying: `Φ` and `M` can depend on `t` and nothing in the machinery changes. That kills complaint three (every generalization needs a new derivation) at the root.

But how do I get a *random* process into this state form? Observed signals are correlated across time — almost nothing physical is white. The trick, due to Bode and Shannon, is to think of that correlation as produced by a linear system sitting between a primary white source and me. Independent macroscopic noise sources are roughly Gaussian and white; pass white noise through a linear dynamic system and out comes a correlated Gaussian process with whatever second-order statistics I want. So I model the process as

  `x(t+1) = Φ(t+1;t) x(t) + u(t)`,

with `{u(t)}` independent, zero-mean, Gaussian, `E u(t)u'(s) = 0` for `t≠s`, and `E u(t)u'(t) = Q(t)`. Only first and second moments enter; `Q` is the covariance of the excitation. And by the Gaussian-stays-Gaussian fact, `x(t)` is then a zero-mean Gaussian process too. This is a *model*, and a clean one: I'll take it as given and treat the separate problem of identifying `Φ, M, Q` from data as somebody else's job.

Now the recursion. Let me write `x*(t|t-1)` for my estimate of `x(t)` given observations up to `t-1`, and `x*(t|t)` once I've also seen `y(t)`. By the projection result, `x*(t1|t) = Ê[x(t1) | Y(t)]`, the projection onto the span of all observations through time `t`. I want to compute `x*(t+1|t)` from `x*(t|t-1)` and the single new measurement `y(t)`, cheaply.

The new observation `y(t)` is partly redundant — its predictable part is already in `Y(t-1)`. Split off the genuinely new piece: let

  `ỹ(t|t-1) = y(t) − M(t) x*(t|t-1)`

be the component of `y(t)` orthogonal to everything I knew before. Call it the *innovation*. Then `Y(t)` decomposes as `Y(t-1)` plus the new orthogonal direction `Z(t)` spanned by `ỹ(t|t-1)`. The point of forcing orthogonality is that projection is additive over orthogonal subspaces — projecting onto `Y(t)` is projecting onto `Y(t-1)` plus projecting onto `Z(t)`, with no cross-talk. So

  `x*(t+1|t) = Ê[x(t+1) | Y(t-1)] + Ê[x(t+1) | Z(t)]`.

Look at the first term. `x(t+1) = Φ(t+1;t) x(t) + u(t)`, and `u(t)` is orthogonal to the entire past `Y(t-1)` — it's brand-new noise, uncorrelated with everything up to `t-1`. So `Ê[x(t+1) | Y(t-1)] = Φ(t+1;t) Ê[x(t) | Y(t-1)] = Φ(t+1;t) x*(t|t-1)`. That's the prediction: just push the previous estimate through the dynamics. The second term is a linear function of the one new direction, so write it as `Ê[x(t+1) | Z(t)] = ∆*(t) ỹ(t|t-1)` for some matrix `∆*(t)` I have to find. Putting it together,

  `x*(t+1|t) = Φ(t+1;t) x*(t|t-1) + ∆*(t) ỹ(t|t-1)`.

So the estimator is itself a linear dynamic system: previous estimate in, push through `Φ`, plus a correction equal to a gain `∆*` times the innovation. That's already the shape I wanted — recursive, one measurement at a time, time-domain. Now I just need `∆*`.

Use orthogonality. The error `x(t+1) − ∆* ỹ` must be orthogonal to the new direction `ỹ`, i.e. `E[(x(t+1) − ∆*(t) ỹ(t|t-1)) ỹ'(t|t-1)] = 0`. Expand: `E[x(t+1) ỹ'] = ∆*(t) E[ỹ ỹ']`. I need both pieces. The innovation is `ỹ = y(t) − M x*(t|t-1) = M x(t) − M x*(t|t-1) = M (x(t) − x*(t|t-1)) = M x̃(t|t-1)`, where `x̃(t|t-1)` is the prediction error. Let `P*(t) = E[x̃(t|t-1) x̃'(t|t-1)]` be the prediction-error covariance. Then `E[ỹ ỹ'] = M P*(t) M'`. And `E[x(t+1) ỹ'] = E[(Φ x(t) + u(t)) (M x̃(t|t-1))'] = Φ E[x(t) x̃'(t|t-1)] M'`, since `u(t)` is orthogonal to the past error. Now `E[x(t) x̃'] = E[(x̂ + x̃) x̃'] = E[x̃ x̃'] = P*(t)`, because `x̂ = x*(t|t-1)` lies in `Y(t-1)` and is orthogonal to the error `x̃`. So `E[x(t+1) ỹ'] = Φ P*(t) M'`. Plug both in:

  `Φ P*(t) M' = ∆*(t) M P*(t) M'`  ⇒  `∆*(t) = Φ(t+1;t) P*(t) M' [M P*(t) M']⁻¹`.

That inverse exists as long as `M P* M'` is positive definite, which holds whenever `P*` is positive definite and no measured coordinate is a linear combination of the others (no redundant rows of `M`). So the gain is the prediction-error covariance, pushed into measurement space by `M`, normalized by the innovation covariance `M P* M'`. The structure is: I weight the correction by *how uncertain my prediction is* (the `P* M'` in the numerator) relative to *how much noise the innovation carries* (the `M P* M'` in the denominator). Big prediction uncertainty, big correction; noisy innovation, small correction. The gain is doing inverse-covariance weighting.

Now I need the covariance to recur too, otherwise I can't compute the next gain. The error transition: `x̃(t+1|t) = x(t+1) − x*(t+1|t)`. Substitute both lines:

  `x̃(t+1|t) = Φ x(t) + u(t) − [Φ x*(t|t-1) + ∆* M x(t)]`
       `= (Φ − ∆* M)(x(t) − x*(t|t-1)) + u(t) − ... `

let me be careful. `x*(t+1|t) = Φ x*(t|t-1) + ∆* ỹ = Φ x*(t|t-1) + ∆* M x̃(t|t-1) = Φ x*(t|t-1) + ∆* M (x(t) − x*(t|t-1))`. So

  `x̃(t+1|t) = Φ x(t) + u(t) − Φ x*(t|t-1) − ∆* M (x(t) − x*(t|t-1)) = (Φ − ∆* M) x̃(t|t-1) + u(t)`.

Define `Φ*(t+1;t) = Φ(t+1;t) − ∆*(t) M(t)` — the transition matrix of the *error*. Since `u(t)` is independent of the past error,

  `P*(t+1) = E[x̃(t+1|t) x̃'(t+1|t)] = Φ* E[x̃(t|t-1) x̃'(t|t-1)] Φ*' + E[u u']`.

The expression simplifies because the new error is orthogonal to the innovation. In symbols,
`Φ* P* M' = (Φ − ∆* M)P*M' = ΦP*M' − ∆*MP*M' = 0`, using the gain equation above. Therefore
`Φ*P*Φ*' = Φ*P*(Φ − ∆*M)' = Φ*P*Φ'`, and

  `P*(t+1) = Φ*(t+1;t) P*(t) Φ'(t+1;t) + Q(t)`,  `Q(t) = E[u(t)u'(t)]`.

There's my covariance recursion. And the expected quadratic loss at any time is just `trace P*(t)` — the sum of the per-coordinate mean-squared errors. The filter rates itself.

Let me make sure I can start it and that it closes the loop. At `t0`, before any data, `x̃(t0|t0-1) = x(t0)`, so I initialize `P*(t0) = E[x(t0) x'(t0)]`, the prior covariance. Then: equation for `∆*` gives `∆*(t0)`; `Φ* = Φ − ∆* M` gives the error transition; the recursion gives `P*(t0+1)`; and around again. If `Q` is positive definite, `P*` stays positive definite at every step, so the inverse in `∆*` is always well-defined. The whole thing is a self-starting iteration a computer just turns the crank on. Complaint two — numerically awful — gone. Complaint one — impulse response — gone; I never compute an impulse response, I run a recursion.

Now I can eliminate `∆*` and `Φ*` and see the covariance evolve on its own. Substitute `Φ* = Φ − ∆* M` and `∆* = Φ P* M' [M P* M']⁻¹` into the recursion. The bracket `Φ* P* Φ'` becomes `(Φ − ∆* M) P* Φ' = Φ P* Φ' − Φ P* M'[M P* M']⁻¹ M P* Φ'`, so

  `P*(t+1) = Φ(t+1;t) { P*(t) − P*(t) M'[M P*(t) M']⁻¹ M P*(t) } Φ'(t+1;t) + Q(t)`.

That's a single nonlinear difference equation for the error covariance — quadratic in `P*`. It plays exactly the role the Wiener–Hopf equation played, but it's a recursion I iterate forward in time rather than an integral equation I solve once over the whole record. And in this no-explicit-measurement-noise state model, if `M` were invertible (I measure everything), the bracket collapses and `P*(t+1) = Q(t)` — trivial, because then I observe the full state and the only residual uncertainty is the fresh excitation. The interesting case is partial observation, `p < n`, where this equation does real work reconstructing the hidden coordinates.

Let me sanity-check the gain against intuition in the scalar case, because matrices can hide sign errors. Take one state, `M = 1`, predict-error variance `P⁻` (writing `P⁻` for the prediction covariance and folding the dynamics aside), measurement variance `R`. There are really two pieces of information here: a prior `N(x̂⁻, P⁻)` from the prediction and a likelihood `N(z, R)` from the measurement. Combining two independent Gaussian pieces of information about the same quantity multiplies their densities, and the product of two Gaussians is again Gaussian with

  mean `= (P⁻ z + R x̂⁻)/(P⁻ + R)`,  variance `= P⁻ R/(P⁻ + R)`.

Rewrite the mean: `x̂⁻ + [P⁻/(P⁻ + R)](z − x̂⁻)`. So the posterior is the prediction plus a gain `K = P⁻/(P⁻ + R)` times the residual `z − x̂⁻`, and the posterior variance is `(1 − K) P⁻`. Read off the limits: if the sensor is perfect, `R→0`, then `K→1` — believe the measurement entirely. If the prediction is already perfect, `P⁻→0`, then `K→0` — ignore the measurement. In between, the weights are the *inverse variances* (precisions): each source is trusted in proportion to how certain it is. That's exactly the precision-weighting I saw in the matrix gain, and it tells me the matrix gain must be the multivariate version of `P⁻/(P⁻+R)` with `M` projecting into measurement space — which is `P⁻ M' (M P⁻ M' + R)⁻¹`. Notice the `+R`: in the clean state-space derivation above I folded the measurement noise into the model as a noise state, but if I keep an explicit measurement noise `v ~ N(0,R)` with `z = M x + v`, the innovation covariance is `M P⁻ M' + R` instead of `M P⁻ M'`. That `R` is the floor on how much the innovation can ever shrink uncertainty.

This last point is worth pinning down, because the cleaner engineering form splits the recursion into two named half-steps and keeps `R` explicit. Let me carry the belief as mean `x̂` and covariance `P`, and write the model as `x_{k+1} = F x_k + w_k`, `z_k = H x_k + v_k`, `w ~ N(0,Q)`, `v ~ N(0,R)` (renaming `Φ→F`, `M→H` to match how I'll code it).

I push the belief through the dynamics before seeing the measurement. The mean goes through `F`: `x̂⁻ = F x̂`, or `x̂⁻ = F x̂ + B u` if a known control input is present. For the covariance, the covariance of a linear image `F x` is `F P F'` — that's just `E[(Fx̃)(Fx̃)'] = F E[x̃ x̃'] F'`. Then the independent process noise adds its own covariance: `P⁻ = F P F' + Q`. The key qualitative fact is that prediction makes me *more* uncertain. `Q` is added, never subtracted; rolling the dynamics forward without new data can only inflate the spread. That's the right direction — extrapolation degrades knowledge.

Then `z_k` arrives. The residual (innovation) is `y = z_k − H x̂⁻`: I subtract the part of the measurement I already predicted, because only the orthogonal remainder carries new information. Its covariance is `S = H P⁻ H' + R`. The gain is `K = P⁻ H' S⁻¹ = P⁻ H' (H P⁻ H' + R)⁻¹`. The state correction is `x̂ = x̂⁻ + K y`, and the covariance shrinks: `P = (I − K H) P⁻`.

Let me actually *derive* the gain a second way, by minimizing the posterior error covariance directly, to be sure `(I−KH)P⁻` and the gain are mutually consistent — I don't want to just assert the trace-minimizing `K`. Posit the update form `x̂ = x̂⁻ + K(z − H x̂⁻)` with `K` unknown, and ask which `K` minimizes the posterior error spread. The posterior error is

  `e = x − x̂ = x − x̂⁻ − K(H x + v − H x̂⁻) = (x − x̂⁻) − K H(x − x̂⁻) − K v = (I − K H) e⁻ − K v`,

where `e⁻ = x − x̂⁻` is the prediction error with covariance `P⁻`, independent of the measurement noise `v` (covariance `R`). Since `e⁻ ⊥ v`, the posterior covariance is

  `P = E[e e'] = (I − K H) P⁻ (I − K H)' + K R K'`.

This is the *Joseph form*, and notice it's a valid covariance for *any* `K`, not only the optimal one — it's manifestly a sum of two `A X A'` terms, so it stays symmetric positive semidefinite even with rounding error or a deliberately suboptimal gain. Now minimize `trace P` over `K`. Expand:

  `trace P = trace P⁻ − 2 trace(K H P⁻) + trace(K(H P⁻ H' + R)K')`,

using `trace(K H P⁻) = trace((K H P⁻)') = trace(P⁻ H' K')` and `P⁻ = P⁻'`. Differentiate with respect to `K` (matrix calculus: `d trace(K A)/dK = A'`, `d trace(K B K')/dK = 2 K B` for symmetric `B`):

  `d/dK trace P = −2 (H P⁻)' + 2 K (H P⁻ H' + R) = −2 P⁻ H' + 2 K S`.

Set to zero: `K S = P⁻ H'`, so `K = P⁻ H' S⁻¹ = P⁻ H'(H P⁻ H' + R)⁻¹`. Same gain as the orthogonality route — good, the two derivations agree. And it's a minimum, since the second derivative `2 S` is positive definite. Now substitute the optimal `K` back into the Joseph form and watch it collapse. `(I − K H) P⁻ (I − K H)' + K R K' = (I − K H)P⁻ − (I − K H)P⁻ H' K' + K R K'`. The last two terms are `−P⁻ H' K' + K H P⁻ H' K' + K R K' = −P⁻ H' K' + K (H P⁻ H' + R) K' = −P⁻ H' K' + K S K'`. But `K S = P⁻ H'`, so `K S K' = P⁻ H' K'`, and the two cancel. Left with `P = (I − K H) P⁻`. So the optimal-gain covariance is the short form `(I − K H) P⁻`, and the Joseph form is its numerically safe generalization. For computation I'll keep the Joseph form `P = (I − K H) P⁻ (I − K H)' + K R K'`, precisely because it tolerates a non-optimal `K` and roundoff without losing symmetry or positivity — the short form `(I−KH)P⁻` can drift indefinite when `KH` is computed imperfectly.

Step back and check optimality cleanly. In the linear-Gaussian case, the belief I carry is genuinely the conditional distribution: `p(x_k | z_{1:k}) = N(x̂_k, P_k)`. Predict and update are exactly the two operations of Bayesian recursion on Gaussians — propagate through the linear dynamics (a Gaussian stays Gaussian, mean through `F`, covariance `FPF'+Q`), then multiply by the measurement likelihood `N(z; Hx, R)` (product of Gaussians, which is the update). Because two moments are a *sufficient statistic* for a Gaussian, nothing is lost by carrying only `x̂` and `P`; the recursion is exact, not an approximation. So the estimate is the conditional mean — the globally minimum-mean-square-error estimate, the best of *all* estimators, linear or not. If the noises aren't Gaussian but I only know their first and second moments, the very same recursion computes the orthogonal projection, which is the best *linear* estimator (best linear unbiased / linear-MMSE) — and you can only beat it with a nonlinear estimator if the process is nongaussian *and* you bring in third-or-higher-order statistics, which by assumption I don't have. Either way, given what's knowable, this is optimal.

One more structural thing nags at me, and it's a pleasant surprise rather than a tool I need. The covariance recursion `P*(t+1) = Φ{P* − P* M'[M P* M']⁻¹ M P*}Φ' + Q` — I've seen that quadratic difference equation before, in the noise-free optimal regulator: choose controls `u(t)` to minimize a quadratic cost `∑ x'Qx`, and the cost-to-go matrix obeys a Riccati recursion of identically the same shape, just with the time index running backward and the matrices transposed and roles swapped (observation matrix `M` ↔ control input matrix, transition ↔ transition'). Estimation and control are duals: replace each matrix `X(t0+τ)` by `X̂'(T−τ)` and the estimation recursion becomes the regulator recursion. The mathematical backbone of both is this one Riccati equation. I won't chase the control side here — it's a separate problem with its own derivation — but it's striking that filtering a noisy signal and steering a plant optimally are the same equation read in two directions. The duality also explains *why* the filter has the feedback structure it does: the innovation `ỹ` is a white process (it's orthogonal across time by construction), so the optimal filter is realizable as a feedback system driven by white residuals, mirroring how the optimal regulator is a feedback law.

Now land it in code, filling the empty slots I sketched earlier. The belief is just the pair `(x, P)`; `time_update` is the predict half-step — push both through `F`, inflate by `Q`; `measurement_update` is the update half-step — residual `y = z − Hx`, innovation covariance `S = HPH' + R`, gain `K = PH'S⁻¹`, correct the mean by `Ky`, shrink the covariance with the Joseph form.

```python
import numpy as np
from numpy.linalg import inv

class LinearGaussianModel:
    """ x_{k+1} = F x_k + B u_k + w,  z_k = H x_k + v,  w~N(0,Q), v~N(0,R) """
    def __init__(self, F, H, Q, R, B=None):
        self.F, self.H, self.Q, self.R, self.B = F, H, Q, R, B

class StateBelief:
    """Two moments are a sufficient summary of a Gaussian belief."""
    def __init__(self, x0, P0):
        self.x = x0      # mean: current best estimate
        self.P = P0      # covariance: our uncertainty about x

def time_update(belief, model, u=None):
    # Predict: roll the belief forward through the dynamics, before the measurement.
    # Mean through F (+ optional control); covariance of a linear image F x is F P F'.
    belief.x = model.F @ belief.x
    if u is not None and model.B is not None:
        belief.x = belief.x + model.B @ u
    # Add process noise Q: extrapolation can only inflate uncertainty.
    belief.P = model.F @ belief.P @ model.F.T + model.Q

def measurement_update(belief, model, z):
    # Update: fold in z, weighting prediction vs measurement by inverse covariance.
    H, R = model.H, model.R
    y = z - H @ belief.x                      # innovation: part of z not already predicted
    S = H @ belief.P @ H.T + R                # innovation covariance (prediction spread + R)
    K = belief.P @ H.T @ inv(S)               # gain: P H' S^-1, precision-weights the residual
    belief.x = belief.x + K @ y              # correct the mean by the scaled residual
    # Joseph form: valid (symmetric PSD) even for non-optimal K and under roundoff.
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

So the chain is: I want a recursive, time-domain, nonstationary estimate of a hidden state, which Wiener can't give cheaply. Squared-error optimality says estimate by the conditional mean; for Gaussian (or for linear estimators in general) that's the orthogonal projection of the state onto the span of the observations, characterized by error-uncorrelated-with-data. Modeling the process in state-space as white noise shaped by linear dynamics lets me split the new measurement into a predictable part and an orthogonal innovation, so the projection becomes a prediction through `F` plus a correction along the innovation. Orthogonality (equivalently, minimizing the trace of the posterior covariance) pins the correction's gain to `K = P⁻H'(HP⁻H'+R)⁻¹`, inverse-covariance weighting between prediction and measurement; the error covariance then satisfies a Riccati-like recursion `(I−KH)P⁻` per step, inflating by `Q` on predict and shrinking on update. Carry just the mean and covariance — a sufficient statistic for the Gaussian belief — and the whole thing is two short matrix half-steps, predict and update, iterated forever.
