A physical system evolves in time — a moving object, a reactor, an airframe — and its internal state $x$ is hidden from me. All I receive is a stream of measurements $y(t_0), \dots, y(t)$, each corrupted by noise and often reading out only some combination of the state coordinates rather than the state itself. I want a rule that at each instant produces the best estimate of the current (or a future, or a past) state given everything seen so far, together with an honest statement of how uncertain that estimate is. The rule has to be recursive — folding in each new measurement in bounded work, without re-touching the whole record; it has to tolerate nonstationarity in the system, the noise, and the data length; it has to be something a computer can iterate rather than an integral equation re-solved per problem; it should report the covariance of its own error so the estimate is self-rating; and it must reconstruct hidden coordinates from noisy observations of only part of the state.

The accepted tool, the Wiener filter, meets essentially none of these cleanly, and it is worth being precise about why. Wiener models signal and noise as stationary random processes with known rational power spectra and obtains the optimal mean-squared-error linear filter from the Wiener–Hopf integral equation, solved by spectral factorization in the frequency domain. It is the correct answer when it applies, but it delivers the optimum as an impulse response, and synthesizing a realizable filter from that is a separate chore; computing the impulse response is numerically awkward and degrades fast as the dimension grows; every change of situation — finite records, nonstationary statistics, growing memory — demands a fresh and often hard derivation; and the assumptions (stationarity, rational spectra, whole-record data) are buried rather than transparent. The pain is not that Wiener is wrong; it is that it is batch, frequency-domain, stationary, and one-off per problem. I want the opposite of every one of those.

I propose the Kalman filter: a recursive, time-domain, exactly-optimal linear estimator of a hidden dynamic state. Let me first fix what "best" means and then build the recursion. Everything the observations convey about an unknown $x(t_1)$ lives in the conditional distribution $\Pr[x(t_1)\le\xi \mid y(t_0),\dots,y(t)]$, and any estimate is a function of the observations graded by a loss $L(\varepsilon)$ on the error $\varepsilon = x(t_1)-\hat x$. Because the risk factors as $E\{L\} = E[\,E\{L\mid y(t_0),\dots,y(t)\}\,]$ and the outer expectation is fixed by the data, minimizing risk is the same as minimizing the conditional expected loss for each realized record. Under squared-error loss the conditional expected loss is $E[x^2\mid\cdot] - 2\hat x\,E[x\mid\cdot] + \hat x^2$, whose minimizer is the conditional mean $\hat x = E[x(t_1)\mid y(t_0),\dots,y(t)]$; and by Sherman's theorem the same conditional mean stays optimal for any even, $|\varepsilon|$-increasing loss whenever the conditional distribution is symmetric and convex on its lower side, which Gaussians are. So I want the conditional expectation — but as stated that is uncomputable without structure.

The structure comes from restricting to linear estimates. The linear combinations $\sum a_i\,y(i)$ form a vector space $Y(t)$ with inner product $\langle u,v\rangle = E[uv]$, and the best linear estimate is the one closest to $x$ in mean square. Decomposing $x = \hat x + \tilde x$ with $\hat x\in Y(t)$ and $\tilde x\perp Y(t)$, for any other $w\in Y(t)$ I get $E(x-w)^2 = E\tilde x^2 + E(\hat x-w)^2 \ge E\tilde x^2$, since the cross term vanishes by orthogonality. Hence the best linear estimate is the orthogonal projection of $x$ onto the span of the observations, characterized by the error being uncorrelated with every observation, $E[(x-\hat x)\,y(i)] = 0$. That orthogonality condition is a concrete equation I can solve, unlike "compute a conditional expectation." And in the Gaussian world the cost of restricting to linear estimators is zero: a zero-mean Gaussian $\tilde x$ orthogonal to the data is actually independent of it, so $E[x\mid y] = \hat x$ — the projection *is* the conditional mean.

To make the projection recursive I describe the system by its state rather than its spectrum, killing the stationarity assumption at the root. A linear process is modeled, following Bode and Shannon, as white noise shaped by linear dynamics, $x(t+1) = \Phi(t+1;t)\,x(t) + u(t)$ with $\{u(t)\}$ independent zero-mean Gaussian, $E\,u(t)u'(s)=0$ for $t\ne s$ and $E\,u(t)u'(t)=Q(t)$, observed through $y(t)=M(t)\,x(t)$. The transition $\Phi$ and readout $M$ may depend on time with no change in machinery. Now write $x^*(t\mid t-1)$ for the projection of $x(t)$ onto the observations through $t-1$. The new measurement $y(t)$ is partly redundant; its genuinely new part is the innovation $\tilde y(t\mid t-1) = y(t) - M(t)\,x^*(t\mid t-1)$, orthogonal to all of $Y(t-1)$. Because projection is additive over orthogonal subspaces, $x^*(t+1\mid t) = \hat E[x(t+1)\mid Y(t-1)] + \hat E[x(t+1)\mid Z(t)]$, where $Z(t)$ is spanned by the innovation. The fresh noise $u(t)$ is orthogonal to the entire past, so the first term collapses to $\Phi(t+1;t)\,x^*(t\mid t-1)$ — just push the previous estimate through the dynamics — and the second is a linear function of the one new direction, giving

$$x^*(t+1\mid t) = \Phi(t+1;t)\,x^*(t\mid t-1) + \Delta^*(t)\,\tilde y(t\mid t-1).$$

The estimator is itself a linear dynamic system: predict through $\Phi$, then correct by a gain $\Delta^*$ times the innovation. The gain is pinned by orthogonality: requiring $E[(x(t+1)-\Delta^*\tilde y)\tilde y'] = 0$ gives $E[x(t+1)\tilde y'] = \Delta^* E[\tilde y\tilde y']$. With $\tilde y = M\tilde x(t\mid t-1)$ and prediction-error covariance $P^*(t) = E[\tilde x\tilde x']$, the innovation covariance is $M P^* M'$, and $E[x(t+1)\tilde y'] = \Phi E[x(t)\tilde x']M' = \Phi P^* M'$ since $E[x(t)\tilde x'] = E[\tilde x\tilde x'] = P^*$ (the estimate is orthogonal to its own error). Hence

$$\Delta^*(t) = \Phi(t+1;t)\,P^*(t)\,M'\,[\,M\,P^*(t)\,M'\,]^{-1},$$

the prediction uncertainty $P^*M'$ in the numerator weighed against the innovation noise $MP^*M'$ in the denominator: big prediction uncertainty means big correction, noisy innovation means small correction. The error then transitions as $\tilde x(t+1\mid t) = (\Phi - \Delta^* M)\tilde x(t\mid t-1) + u(t)$, and with $\Phi^* = \Phi - \Delta^* M$ the new error is orthogonal to the innovation, so the cross terms drop and the covariance recurs as $P^*(t+1) = \Phi^* P^* \Phi^{*\prime} + Q = \Phi^* P^* \Phi' + Q$. Eliminating the gain yields the single quadratic difference equation

$$P^*(t+1) = \Phi\,\bigl\{\,P^* - P^* M'[\,M P^* M'\,]^{-1} M P^*\,\bigr\}\,\Phi' + Q,$$

a Riccati-like recursion that plays the role of the Wiener–Hopf equation but is iterated forward in time rather than solved once over the whole record. Started from the prior covariance $P^*(t_0) = E[x(t_0)x'(t_0)]$ and kept positive definite by a positive-definite $Q$ and full-row-rank $M$, the whole thing is a self-starting crank a computer just turns. The expected squared error per step is $\mathrm{trace}\,P^*$, so the filter rates itself.

The cleaner engineering form keeps the measurement noise explicit and splits the step into two named half-steps. Carry the belief as mean $\hat x$ and covariance $P$ with model $x_{k+1}=Fx_k+w_k$, $z_k=Hx_k+v_k$, $w\sim N(0,Q)$, $v\sim N(0,R)$. The predict step rolls the belief through the dynamics: $\hat x^- = F\hat x\,(+\,Bu)$ and $P^- = F P F' + Q$, where $Q$ enters as an addition — extrapolation alone can never sharpen knowledge. The update step folds in $z_k$ through the residual $y = z_k - H\hat x^-$ with innovation covariance $S = H P^- H' + R$; that extra $R$ is the floor on how much the innovation can shrink uncertainty. The gain is $K = P^- H' S^{-1} = P^- H'(H P^- H' + R)^{-1}$, the mean correction is $\hat x = \hat x^- + Ky$, and the covariance contracts. In the scalar case this is exactly the product of two Gaussians, prior $N(\hat x^-, P^-)$ times likelihood $N(z, R)$, giving posterior mean $\hat x^- + [P^-/(P^-+R)](z-\hat x^-)$ and variance $(1-K)P^-$: perfect sensor $R\to0$ forces $K\to1$, perfect prediction $P^-\to0$ forces $K\to0$, and in between each source is trusted in proportion to its precision.

I can derive the gain a second way to confirm consistency — minimize the posterior error covariance directly. Positing $\hat x = \hat x^- + K(z - H\hat x^-)$, the posterior error is $e = (I-KH)e^- - Kv$ with $e^-\perp v$, so

$$P = (I - K H)\,P^-\,(I - K H)' + K R K',$$

the Joseph form, which is a sum of two congruence terms and therefore symmetric positive semidefinite for *any* $K$, optimal or not — which is exactly why it resists drifting asymmetric or indefinite under roundoff. Minimizing $\mathrm{trace}\,P$ over $K$ gives, via $d\,\mathrm{trace}(KA)/dK = A'$ and $d\,\mathrm{trace}(KBK')/dK = 2KB$, the stationarity condition $KS = P^- H'$, hence $K = P^- H'(HP^-H'+R)^{-1}$ — the same gain as the orthogonality route, and a genuine minimum since the second derivative $2S$ is positive definite. Substituting the optimal $K$ back collapses the Joseph form to the short form $P = (I-KH)P^-$, because $KSK' = P^-H'K'$ cancels the cross term. For computation I keep the Joseph form precisely because it tolerates a non-optimal $K$ and roundoff, where the short form can drift indefinite.

What makes this exact rather than approximate is that two moments are a sufficient statistic for a Gaussian: the carried belief $N(\hat x_k, P_k)$ *is* the posterior $p(x_k\mid z_{1:k})$. Predict propagates a Gaussian through linear dynamics (Gaussian stays Gaussian, mean through $F$, covariance $FPF'+Q$); update multiplies by the Gaussian measurement likelihood; the product of Gaussians is Gaussian. So $\hat x$ is the conditional mean, the minimum-mean-square-error estimate among all estimators, linear or not. Drop Gaussianity but keep first and second moments and the identical recursion computes the orthogonal projection — the best linear estimator — beatable only when the process is non-Gaussian and third-or-higher-order statistics are on hand, which by assumption they are not. The quadratic covariance recursion is, pleasingly, the same Riccati equation that governs the linear-quadratic regulator read with time reversed and roles swapped: estimation and control are duals, and the duality also explains the feedback structure, since the innovation is a white process and the optimal filter is realizable as feedback driven by white residuals. The whole estimator reduces to two short matrix half-steps, predict and update, iterated forever, carrying only the mean and covariance.

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
