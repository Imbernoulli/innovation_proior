We have an unknown nondecreasing mean response curve $M(x) = \mathbb{E}[Y(x)]$ and a target level $\alpha$, and we want the unique $\theta$ solving $M(\theta) = \alpha$. The catch is that we never see $M(x)$; we can only run an experiment at a chosen level $x$ and receive a random response $Y(x)$ whose conditional mean is $M(x)$. Each level may depend on all the levels and responses that came before. The clean instance is quantile estimation from binary data: a subject has an unobserved threshold, at dose $x$ we see only $y \in \{0,1\}$ with $P(y=1 \mid x) = F(x)$ for an unknown distribution function $F$, and we want the dose $\theta$ where $F(\theta) = \alpha$. We want a rule that is distribution-free — it should not require knowing $F$, its derivative, or the noise law — and that drives the chosen levels toward $\theta$ from an arbitrary start.

The two obvious tools both fail in opposite ways. Deterministic root finding — Newton, false position, or a fixed-gain step against the residual $M(x)-\alpha$ — works because the residual changes sign across the root, so it always points us the right way. But it assumes we can evaluate $M$. If I simply substitute one noisy response $y_n$ for the unknown residual and keep a fixed gain, the mean motion still contracts toward $\theta$, but each new response injects a fresh, fixed-size random kick that never fades. The iterate cannot converge to a point; at best it forms a cloud around it. The opposite repair is to estimate first and move later: at each $x_n$, repeat the experiment many times, average the responses into an accurate estimate of $M(x_n)$, then take a deterministic step. Repeated sampling does shrink the uncertainty of a mean, but this decouples estimation from search and spends a large sampling budget pinning down $M$ at levels that may be far from the target — observations whose only payoff is certifying that the current point was wrong. The real difficulty is to make a single response do two jobs at once: move the level in the correct average direction, and have its random part average away over time rather than remain permanently visible.

I propose the Robbins–Monro stochastic approximation method. The sign of the response fixes the shape of the update before any proof: when $x_n$ is too high, $M(x_n) - \alpha > 0$, so the increment should be negative; when $x_n$ is too low, the increment should be positive. With a positive gain $a_n$ the natural rule is therefore
$$x_{n+1} = x_n + a_n\,(\alpha - y_n).$$
There is no loss function, no analytic gradient, and no differentiability requirement here — only a mean response whose sign around $\theta$ is informative. The entire content of the method is then the schedule of gains, and the right schedule is forced by looking at the squared error, which separates drift from noise cleanly. Let $V_n = (x_n - \theta)^2$. Conditioning on the past $\mathcal{F}_n$,
$$\mathbb{E}[V_{n+1} \mid \mathcal{F}_n] = V_n - 2a_n\,(x_n - \theta)(M(x_n) - \alpha) + a_n^2\,\mathbb{E}[(y_n - \alpha)^2 \mid \mathcal{F}_n].$$
This is the decisive split. The middle term is the useful drift: because $M$ is monotone, $(x_n - \theta)$ and $(M(x_n) - \alpha)$ share a sign, so $(x_n - \theta)(M(x_n) - \alpha) \ge 0$ and the term genuinely pulls the squared error down. The last term is the pure cost of using a noisy response in place of the exact residual; with bounded second moments it is on the order of $a_n^2$. So the two conditions on the gains are not decorative — each does one of the two jobs. Square summability, $\sum_n a_n^2 < \infty$, makes the total injected variance finite, so the squared error has a chance to settle. But a settled process can settle at the wrong place if it runs out of motion before reaching $\theta$, so the deterministic pull must remain inexhaustible: $\sum_n a_n = \infty$. The slope of $M$ can only be controlled on the region the iterate actually reaches, so the lower bound on the drift is carried as a sequence, $d_n \ge k_n b_n$ with $b_n = \mathbb{E}[(x_n - \theta)^2]$; under the monotonicity and local-slope assumptions the accumulated weighted lower bound diverges for $1/n$-type gains, while the squared-error recursion already forces the total drift to be finite — and a nonnegative series that both diverges in its lower bound and sums to something finite can only do so if $b_n \to 0$. The limiting squared error is therefore zero. A gain like $a_n = c/n$ is the simplest compromise that satisfies both conditions at once: the kicks shrink fast enough that their variances add to a finite number, yet the total deterministic travel is still unbounded.

In the modern martingale language the same argument is one statement: $V_n$ is an almost-supermartingale, its conditional expectation bounded by the current value plus a summable noise allowance minus a nonnegative progress term. The almost-supermartingale theorem gives a finite limit for the error and a finite total progress term, and the nonsummable gains together with the sign/slope condition rule out any positive limiting error — yielding almost-sure convergence under standard strengthened hypotheses, where Robbins and Monro's original 1951 argument gives convergence in quadratic mean and hence in probability. The method is an averaging process disguised as search: it never stops sampling, never separately estimates the whole curve, and never lets old noise keep full influence. For the quantile instance the update is unchanged — $x_{n+1} = x_n + a_n(\alpha - y_n)$ with $y_n \in \{0,1\}$ and $P(y_n = 1 \mid x_n) = F(x_n)$ — so the dose decreases after too many responses and increases after too few, with the gain shrinking over time. SGD falls out as the special case where the noisy response happens to be an unbiased gradient or residual of an expected objective; stochastic approximation is the broader thing, noisy root or optimum finding by sequential experiments with diminishing gains.

The maximization version follows the same pattern once I stop treating it as a different problem. The maximizer is a root of the derivative $M'$, but $M'$ is not observed, so I manufacture a noisy derivative by probing two nearby points and taking a finite difference — the Kiefer–Wolfowitz adaptation — giving $z_{n+1} = z_n + a_n\,(y_+ - y_-)/(2c_n)$ with $y_\pm \sim M(z_n \pm c_n) + \text{noise}$, again a noisy root-finding step now aimed at $M'(\theta) = 0$. This adds a second knob with competing demands: the probe width $c_n$ must go to zero so the finite difference becomes a true derivative, but dividing by $c_n$ amplifies the observation noise. So beyond $\sum_n a_n = \infty$ I need the accumulated finite-difference bias, roughly $\sum_n a_n c_n$, to be finite, and the amplified variance, roughly $\sum_n a_n^2 / c_n^2$, to be finite. With $a_n = 1/n$ and $c_n = n^{-1/3}$ both tails are summable while the gains themselves still add up forever.

```python
import random
import math


def robbins_monro_step(x, gradient, n, c=1.0):
    """One Robbins–Monro/SGD step with diminishing gain c / n."""
    gain = c / n
    return x - gain * gradient


def sgd_robbins_monro(observe, alpha, x0, n_steps, c=1.0):
    """Find a root of M(x) = alpha from noisy observations y ~ M(x)."""
    x = x0
    for n in range(1, n_steps + 1):
        y = observe(x)
        residual = alpha - y          # noisy direction toward the root
        x = robbins_monro_step(x, -residual, n, c)
    return x


# Example: estimate the median of a logistic distribution.
def observe_logistic(x):
    p = 1.0 / (1.0 + math.exp(-x))
    return 1 if random.random() < p else 0


if __name__ == "__main__":
    random.seed(0)
    estimate = sgd_robbins_monro(
        observe_logistic, alpha=0.5, x0=-2.0, n_steps=5000, c=2.0
    )
    print(f"estimated root: {estimate:.4f}")
```
