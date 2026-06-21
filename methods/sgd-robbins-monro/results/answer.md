# Robbins-Monro Stochastic Approximation

## Root-Finding Method

Let `M(x)=E[Y(x)]` be an unknown nondecreasing mean response, and suppose the equation

```text
M(theta) = alpha
```

has a unique solution. At step `n`, choose level `x_n`, observe one random response `y_n` with conditional mean `M(x_n)`, and update

```text
x_{n+1} = x_n + a_n (alpha - y_n).
```

Use positive gains satisfying

```text
sum_n a_n = infinity,
sum_n a_n^2 < infinity.
```

The standard example is `a_n = c/n`.

The construction joins two older ideas without reducing to either one: deterministic equation solving supplies the "move by the residual" shape, while repeated-sample mean estimation supplies the need to average away observation error. The recursion makes the sequential levels themselves perform that averaging.

## Why It Converges

With `V_n=(x_n-theta)^2`,

```text
E[V_{n+1} | F_n]
  = V_n
    - 2 a_n (x_n-theta)(M(x_n)-alpha)
    + a_n^2 E[(y_n-alpha)^2 | F_n].
```

Monotonicity makes `(x_n-theta)(M(x_n)-alpha) >= 0`, so the middle term pulls toward the root. Bounded second moments make the final term summable when `sum a_n^2 < infinity`. The condition `sum a_n = infinity`, together with the monotone/slope assumptions in the 1951 theorem, prevents the iteration from exhausting its motion before reaching the root. Robbins and Monro prove convergence in quadratic mean and hence in probability; the later almost-supermartingale view gives the same noise-damping/reaching split for almost-sure convergence under standard strengthened hypotheses.

## Quantile Form

For response/nonresponse data,

```text
y_n in {0, 1},
P(y_n = 1 | x_n) = F(x_n),
F(theta) = alpha.
```

The same update becomes

```text
x_{n+1} = x_n + a_n (alpha - y_n),
```

so the dose decreases after too many responses and increases after too few responses, with the gain shrinking over time.

## Noisy Maximum Variant

Kiefer and Wolfowitz adapt the idea to maximizing an unknown regression function. Probe near the current point:

```text
y_+ ~ M(z_n + c_n) + noise,
y_- ~ M(z_n - c_n) + noise,
z_{n+1} = z_n + a_n (y_+ - y_-) / (2 c_n).
```

Use

```text
c_n -> 0,
sum_n a_n = infinity,
sum_n a_n c_n < infinity,
sum_n a_n^2 / c_n^2 < infinity.
```

The `c_n` terms handle finite-difference bias and noise amplification.

## Code Artifact

See `code/robbins_monro.py` for a minimal implementation of the root recursion and the finite-difference maximum recursion.

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

SGD is a special case of this framework when the noisy response is an unbiased gradient or residual of an expected objective. The original method is broader: stochastic approximation is noisy root or optimum finding by sequential experiments with diminishing gains.
