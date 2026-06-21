# Context: Finding a Level From Noisy Responses

## Problem

There is an unknown mean response curve `M(x)`. For a given target level `alpha`, the task is to find the unique value `theta` such that

```text
M(theta) = alpha.
```

The experimenter does not know the formula for `M`, and at a chosen level `x` cannot observe `M(x)` directly. The only available operation is to run an experiment at `x` and receive a random response `Y(x)` whose conditional mean is `M(x)`. The next level may depend on all previous levels and responses. The objective is a sequential rule that drives the chosen levels toward `theta`, even though every observation remains noisy.

This captures response/nonresponse experiments. A subject has an unobserved threshold; at dose `x`, the experimenter sees only a binary response. The response probability is the unknown distribution function `F(x)`, and the desired dose is the quantile where `F(theta)=alpha`.

## Known Tools

Deterministic root finding assumes exact function information. If `M(x)` can be evaluated, one may use Newton's method, false position, or a simpler fixed-gain iteration based on the residual `M(x)-alpha`. These methods move in the correct direction because the residual has the opposite sign on the two sides of the root.

Statistical estimation supplies a different tool. If the goal is to estimate the mean response at one fixed level, repeated observations and a sample average reduce the noise. The standard error of a mean shrinks with repeated sampling, and small-sample theory warns that very short series can leave substantial uncertainty.

Sequential experimental design adds a constraint: the sampling locations are themselves decisions. A useful procedure should spend observations where they help locate the level, not merely describe the whole response curve.

## Failure Modes

A deterministic iteration is fragile under direct noisy substitution. If a single noisy response is used where the exact residual should have been used, a fixed gain keeps injecting fresh noise into the level. The expected motion may point toward the root, but the realized sequence keeps jittering.

The opposite repair is also unsatisfactory. One can average many observations at each proposed level, estimate `M(x)` accurately there, and then take a deterministic root-finding step. But this pays a large sampling cost at levels that may later prove far from the target. Estimation and search are decoupled, so many observations only certify that the current point was not the one wanted.

The core difficulty is to make one response serve both purposes: it should move the current level in the right average direction, while its random part should not remain permanently visible.

## Evaluation Setting

The clean testing case is quantile estimation from binary responses. There is an unknown nondecreasing distribution function `F`, a target probability `alpha`, and an unknown quantile `theta` with `F(theta)=alpha`. At each chosen level `x_n`, the observation is

```text
y_n = 1  with probability F(x_n),
y_n = 0  otherwise.
```

The rule should be distribution-free: it should not require knowing `F`, its derivative, or the noise law. It should work from an arbitrary starting value under natural monotonicity and local slope conditions. A related optimization setting asks for the maximizer of an unknown mean response when only noisy function values, not derivatives, can be observed.

## Starting Code

```python
def observe(x):
    """Return one noisy response with conditional mean M(x)."""
    raise NotImplementedError


def root_find(observe, alpha, x0, n_steps):
    """Choose levels sequentially so the levels approach M(x)=alpha."""
    x = x0
    for n in range(1, n_steps + 1):
        y = observe(x)
        # TODO: update x using this one noisy response.
        pass
    return x


def maximize(observe, x0, n_steps):
    """Locate a maximizer of an unknown mean response from noisy values."""
    x = x0
    for n in range(1, n_steps + 1):
        # TODO: infer a direction without an analytic derivative.
        pass
    return x
```
