## Research question

A physical system evolves in time — the position and velocity of a tracked object, the
concentration in a reactor, the orientation of an airframe. We cannot see its internal state `x`
directly; we only get a stream of measurements `y(t0), y(t1), …, y(t)`, each corrupted by noise,
and often measuring only some combination of the state coordinates rather than the state itself.
The goal is a rule that, at each instant, produces the best estimate of the current (or a future,
or a past) value of the state given everything observed so far, together with an honest statement
of how uncertain that estimate is.

Three flavors of the same problem go under one name, *estimation*: if we want `x(t1)` for `t1 < t`
it is smoothing (interpolation); for `t1 = t`, filtering; for `t1 > t`, prediction. "Best" must be
made precise — penalize the estimation error `ε = x(t1) − x̂(t1)` by a loss `L(ε)` and minimize the
expected loss (the risk). What is wanted is a rule that:

1. is **recursive / online** — it folds in each new measurement as it arrives, in bounded work,
   without re-processing the entire past record;
2. does **not assume stationarity** — the system, the noise statistics, and the available data length
   may all vary with time;
3. is **well suited to machine computation** — a procedure a computer can iterate, not an integral
   equation to be solved analytically per problem;
4. comes with the **covariance of its own error**, so the estimate is self-rating;
5. handles the case where **only part of the state is measured**, reconstructing the hidden
   coordinates from noisy observations of the rest.

The prevailing tool meets essentially none of these cleanly. That gap is the problem.

## Background

**The probabilistic setting.** Signal, noise, and their combination are random processes. Given
measured values `η(t0), …, η(t)` of the observed variable, all the information they convey about an
unknown `x(t1)` is contained in the conditional distribution `Pr[x(t1) ≤ ξ | y(t0)=η(t0), …, y(t)=η(t)]`.
Any estimate is a (non-random) function of the observations. Penalizing error by a loss `L(ε)` that is
zero at zero, even, and non-decreasing in `|ε|`, the quantity to minimize is the conditional expected
loss `E{L[x(t1) − x̂(t1)] | y(t0), …, y(t)}`.

**Conditional expectation as the optimal estimate.** Under a squared-error loss `L(ε)=ε²`, the
estimate minimizing expected loss is the conditional mean `x̂ = E[x(t1) | y(t0), …, y(t)]` — a
classical fact in probability theory (Doob, *Stochastic Processes*, pp. 77–78). More generally, if the
conditional distribution is symmetric about its mean and convex on the lower side in the sense used by
Sherman's loss theorem, that same conditional mean is optimal for any even loss that grows with
`|ε|`; Gaussian conditional distributions satisfy these requirements.

**The orthogonality principle.** Among all *linear* functions of the observations, the one minimizing
mean-squared error is the **orthogonal projection** of the unknown onto the linear span `Y(t)` of the
observed random variables: the estimate `x̂` is the vector in `Y(t)` for which the error `x − x̂` is
orthogonal (uncorrelated, `E[(x−x̂)y(i)]=0`) to every observation. Equivalently `x̂` is the closest
point in `Y(t)` to `x` in mean-square. This is standard in the probability literature (Doob pp. 75–78,
148–155; Loève pp. 455–464) but had not been used much in engineering.

**Gaussian conditioning.** For a Gaussian process the orthogonal projection is *exactly* the
conditional expectation, not merely the best linear approximation to it. The supporting facts: linear
functions (hence conditional expectations) of a Gaussian process are themselves Gaussian; orthogonal
zero-mean Gaussian variables are independent; and for any process with given means and covariances
there is a unique Gaussian process matching them. Consequences: (a) a Gaussian belief stays Gaussian
under a linear map plus independent Gaussian noise, so it is fully described by its mean and
covariance — two moments are a sufficient statistic; (b) in the Gaussian case the best linear estimator
*is* the globally optimal estimator, so restricting to linear estimators costs nothing.

**State and state transition.** A dynamic system is described by its *state* — the least information
about the past needed to predict the future — evolving by a first-order recursion. A linear
discrete-time system is `x(t+1) = Φ(t+1;t) x(t) + (driving term)`, `y(t) = M(t) x(t)`, with `Φ` the
transition matrix (`Φ(t;t)=I`, `Φ(t;s)Φ(s;r)=Φ(t;r)`) and `M` reading out the (possibly partial)
observation. This first-order viewpoint covers time-varying systems with no change of machinery.

**Random processes as shaped white noise.** A correlated random signal can be modeled as the output of
a linear dynamic system driven by independent ("white") noise (Bode–Shannon 1950): primary macroscopic
noise sources are well approximated as independent Gaussian, and the correlations we observe come from a
linear system sitting between source and observer. So a process with given second-order statistics is
represented as `x(t+1) = Φ x(t) + u(t)`, `{u(t)}` independent, zero-mean, Gaussian, with
`E u(t)u'(s)=0` for `t≠s` and `E u(t)u'(t)=Q(t)`. Only first- and second-order averages enter; no
higher statistics are needed.

**The empirical fact that motivates the model.** Observed random phenomena are generally *not*
describable by independent random variables — the statistical dependence between a signal sampled at
different times is the rule, not the exception (the thermal-noise voltage in a resistor is white, but
almost nothing downstream of a physical system is). That correlation is exactly what a dynamic system
between source and observer produces, which is why the shaped-white-noise model is the natural one.

## Baselines

**The Wiener filter (Wiener 1949) and the Wiener–Hopf equation.** The reference solution to filtering
and prediction. Modeling signal and noise as stationary random processes with known (rational) power
spectra, Wiener showed the optimal linear mean-squared-error filter is characterized by the
Wiener–Hopf integral equation, solved by *spectral factorization* in the frequency domain; the result
is the optimal filter's impulse response. It is the correct answer for stationary signal-plus-noise and
is the yardstick. Its open limitations:

- The optimum is delivered as an **impulse response**; synthesizing a physical/realizable filter from
  that data is itself a non-trivial task.
- **Numerical determination** of the optimal impulse response is involved and "poorly suited to machine
  computation," and degrades rapidly as the problem grows in dimension.
- Each **generalization** — growing-memory filters, finite data records, nonstationary statistics —
  requires a fresh and frequently difficult derivation; there is no single machine that covers them.
- The derivations **obscure the assumptions**; what is actually being assumed (stationarity, rational
  spectra, infinite/whole-record data) is not transparent, and the method is intrinsically *batch* and
  frequency-domain.

**Bode–Shannon (1950), "A Simplified Derivation of Linear Least-Squares Smoothing and Prediction."**
Re-derives the Wiener result by *whitening*: model the signal as white noise passed through a shaping
filter, so the problem reduces to handling white noise. This supplies the shaping-filter / state-space
representation of a random process, but the estimation itself is still posed and solved in the
stationary, whole-record frequency domain.

**Finite-memory and nonstationary extensions (Zadeh–Ragazzini 1950; Booton 1952; Blum 1958).**
Patches onto the Wiener program: Zadeh–Ragazzini handle finite memory, Booton the nonstationary
Wiener–Hopf equation, Blum gives recursion formulas for growing-memory digital filters. Blum's is the
only prior *explicit recursion* for the growing-memory case — but it is special-cased and algebraically
heavy, "much more complicated" than a single uniform recursion would be.

**Least squares and its recursive form (Gauss; recursive least squares).** Choose the estimate
minimizing the sum of squared residuals between predicted and observed quantities; the solution is the
normal equations. In recursive form, each new observation updates the previous estimate by a correction
proportional to its residual, with a gain that shrinks as data accumulate, so no batch re-inversion is
needed. This is the ancestor of the online, "correct-by-a-weighted-residual" structure, but classical
least squares carries no dynamic model of how the unknown itself evolves between observations.

## Evaluation settings

The natural testbeds are nonstationary prediction problems with known linear-Gaussian models, where the
solution can sometimes be obtained in closed form for cross-checking against earlier Wiener-theory
treatments:

- **Scalar signal-plus-noise prediction.** Signal `x1(t+1)=φ11 x1(t)+u1(t)`, independent noise
  `x2(t+1)=u2(t)`, observation `y=x1+x2`, with excitation variances `a²` (signal) and `b²` (noise);
  predict one step ahead and study how the answer depends on the noise-to-signal ratio `b²/a²`.
- **Tracking a particle with random initial velocity.** Position `x1(t+1)=x1+x2`, constant unknown
  velocity `x2(t+1)=x2`, plus correlated measurement noise `x3` with `x3(t+1)=φ33 x3 + u3`, observing
  `y=x1+x3`; estimate position and velocity at the time of the last measurement. Cross-check the
  large-`t` limit against straight-line least-squares fitting and against the continuous-data Wiener
  results of Shinbrot (1958) and Solodovnikov (1956).

Metrics: the mean-squared estimation error per coordinate, and its sum `trace P*(t)`, the expected
quadratic loss carried by the error-covariance recursion itself. The protocol is to specify the model
matrices and the initial covariance, iterate the recursion, and read off the error covariance at each
step. Sensor noise covariance is something one can sample offline by exercising the sensor; the process
(excitation) covariance is harder, since the process being estimated is not directly observable.

## Code framework

What already exists: linear-algebra primitives (matrix multiply, transpose, inverse) and a sampled
linear-system model. Given the model matrices and noise covariances, we need a recursive estimator
that carries some summary of the hidden state, advances that summary through the dynamics, and folds
in each measurement.

```python
import numpy as np

class LinearGaussianModel:
    """A sampled linear system  x_{k+1} = F x_k + B u_k + w,  z_k = H x_k + v,
    with  w ~ N(0, Q),  v ~ N(0, R)  independent and white."""
    def __init__(self, F, H, Q, R, B=None):
        self.F, self.H, self.Q, self.R, self.B = F, H, Q, R, B

class StateBelief:
    """Whatever we carry forward to summarize what is known about x_k.
    TODO: decide what statistics are sufficient to represent the belief."""
    def __init__(self, x0, P0):
        self.x = x0    # current best estimate
        self.P = P0    # TODO: representation of uncertainty about x

def time_update(belief, model, u=None):
    """Advance the belief from step k-1 to step k using only the dynamics,
    before seeing the new measurement.
    TODO: propagate the summary through F, optional B u, and the noise w."""
    pass

def measurement_update(belief, model, z):
    """Fold a new noisy measurement z_k into the belief.
    TODO: combine the predicted belief with z, weighting each by how
    trustworthy it is, and reduce the uncertainty accordingly."""
    pass

def run_filter(model, belief, measurements, controls=None):
    """Recursively estimate the state over a stream of measurements."""
    estimates = []
    for k, z in enumerate(measurements):
        time_update(belief, model, None if controls is None else controls[k])
        measurement_update(belief, model, z)
        estimates.append((belief.x.copy(), belief.P.copy()))
    return estimates
```
