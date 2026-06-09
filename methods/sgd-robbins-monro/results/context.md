# Context: finding the root of a function you can only measure with noise

## Research question

A function $M(x)$ is defined as the expected value, at level $x$, of the response to an experiment. We do not know its form. We are told only that it is monotone and that the equation
$$M(x) = \alpha$$
has a unique root $x = \theta$, where $\alpha$ is a given constant; we want to locate $\theta$. The catch is the only thing we can do is run an experiment at a level $x$ of our choosing and read off a random response $Y(x)$, whose distribution $H(y\mid x)$ is also unknown, with
$$M(x) = \int_{-\infty}^{\infty} y\, dH(y\mid x) = \mathbb{E}[Y \mid x].$$
So at each $x$ we see $M(x)$ only through noise. We must choose a sequence of levels $x_1, x_2, \dots$ sequentially — $x_{n+1}$ may depend on everything seen so far — so that $x_n \to \theta$.

Why it matters: this is the shape of an enormous class of problems. In a bioassay we set a dose $x$ and observe only whether each subject responds (a 0/1 outcome); the response *probability* $F(x)$ is the unknown monotone curve and we want the dose $\theta$ at which it equals a target $\alpha$. In sensitivity testing, sequential design of experiments, and any tuning problem where each evaluation is expensive and noisy, the same structure appears: a monotone mean response, observable only through a random draw, whose level-set we need. A good method must (i) reach $\theta$ from an arbitrary start, with no knowledge of the form of $M$ or of the noise; (ii) be computable as a simple recursion on the running level; and (iii) drive the error to zero despite the noise never going away. The hard part is the third: every observation carries irreducible noise, so any method that simply trusts its measurements will jitter forever.

## Background

The deterministic version of this problem is classical. If $M(x)$ is *known* exactly, "find the root of $M(x)=\alpha$" is solved by successive approximation. Newton's method, $x_{n+1} = x_n - (M(x_n)-\alpha)/M'(x_n)$, converges quadratically when started near $\theta$ and $M' \neq 0$. Even a crude fixed-gain iteration $x_{n+1} = x_n + a\,(\alpha - M(x_n))$ converges geometrically for small enough constant $a>0$ when $M$ is increasing, because the deterministic map is a contraction near $\theta$. The defining feature of all of these is that they read an *exact* value $M(x_n)$ at each step and take a step that is a deterministic function of it. The error contracts because the residual $M(x_n)-\alpha$ contracts.

The statistical machinery for "measure a mean through noise" is equally classical and points the opposite way. To estimate $M(x)$ at a single fixed $x$, the law of large numbers says: draw $m$ independent responses $Y^{(1)}(x),\dots,Y^{(m)}(x)$, average them, and the sample mean $\bar Y_m(x)$ concentrates around $M(x)$ with error of order $\sigma/\sqrt{m}$ (the central limit theorem sharpens this to a $1/\sqrt m$ Gaussian width). So pinning $M(x)$ down to within $\varepsilon$ costs on the order of $\sigma^2/\varepsilon^2$ experiments — *at that one level $x$*. This is the standard, trusted way to fight noise: repeat and average.

Two further pieces of background are load-bearing. First, the theory of martingales: Doob's convergence theorem says a nonnegative supermartingale (a sequence whose conditional expectation never increases, $\mathbb{E}[Z_{n+1}\mid \mathcal F_n] \le Z_n$) converges almost surely to a finite limit. This is the natural tool whenever a nonnegative "energy" can be shown to drift downward in expectation. Second, and more elementary, the bias–variance character of a step size: a large gain moves fast but injects a lot of the observation's noise into the iterate, while a small gain injects little noise but moves slowly.

There is also a known phenomenon about the obvious composite strategy. Suppose we estimate $M$ on a grid by averaging at each grid point, then root-find on the estimated curve. The averaging cost $\sigma^2/\varepsilon^2$ is paid *at every grid point*, including points far from $\theta$ where the value of $M$ is irrelevant to locating the root. The sampling budget is spent establishing the whole curve to high precision when all we wanted was one level-crossing. This is the diagnostic pain that any efficient method must avoid: estimation and search are decoupled, and the estimation half does work that the search half throws away.

A prior published note (Anderson, McCarthy & Tukey, Naval Ordnance Report 65-46, 1946) had already raised the possibility that a *convergent sequential* process could be used in this quantile-estimation setting — i.e. that one need not fix the levels in advance — but without a procedure carrying a convergence guarantee.

## Baselines

**Deterministic successive approximation (Newton / fixed-gain iteration).** Core idea: treat $g(x)=M(x)-\alpha$ as a known residual and iterate $x_{n+1} = x_n - g(x_n)/M'(x_n)$ (Newton) or $x_{n+1}=x_n + a\,(\alpha - M(x_n))$ (fixed gain). Math: near $\theta$, $M(x)-\alpha \approx M'(\theta)(x-\theta)$, so the fixed-gain map has multiplier $1 - aM'(\theta)$, contracting for $0<a<2/M'(\theta)$; Newton self-scales and converges quadratically. Gap it leaves: it requires the *exact* value $M(x_n)$ (and, for Newton, $M'$). Feed it a single noisy draw $y_n$ in place of $M(x_n)$ and, with the gain $a$ fixed, the noise enters every step at full strength; the multiplier still contracts the *mean* error but the injected variance is constant, so $x_n$ converges to a noisy ball around $\theta$, not to $\theta$. There is no mechanism in a fixed-gain rule to damp observation noise over time.

**Sample-mean estimation then root-find (law of large numbers).** Core idea: use the trusted noise-fighting tool — average. At each candidate level $x$, draw many responses and form $\bar Y_m(x) \to M(x)$; once you have an accurate curve, apply a deterministic root-finder. Math: $\mathrm{Var}(\bar Y_m) = \sigma^2/m$, so $m \sim \sigma^2/\varepsilon^2$ buys accuracy $\varepsilon$ per level. Gap it leaves: it decouples estimation from search. The $\sigma^2/\varepsilon^2$ cost is paid per level, and most levels visited are not the root, so a large share of the sampling budget produces information that is discarded. It is also awkward sequentially — you must decide $m$ before you know how close to $\theta$ you are. Nothing forces the sampling effort to concentrate where it matters.

**Doob martingale convergence (the analysis tool, as it stands).** Core idea: if a nonnegative quantity $Z_n$ (e.g. a squared error) satisfies $\mathbb{E}[Z_{n+1}\mid \mathcal F_n] \le Z_n$, it converges a.s. Gap it leaves *as stated*: a noisy update's squared error does not obey a clean supermartingale inequality — each step adds a nonnegative noise term as well as subtracting a drift term, so the conditional expectation is $\mathbb{E}[Z_{n+1}\mid\mathcal F_n] \le Z_n + (\text{nonneg noise drift}) - (\text{drift toward root})$. Doob's theorem in its bare form does not cover a process with an extra summable upward push; one needs the almost-supermartingale generalization (a nonnegative process with $\mathbb{E}[Z_{n+1}\mid\mathcal F_n]\le (1+\eta_n)Z_n + \gamma_n - \psi_n$, summable $\eta_n,\gamma_n$) to conclude both that $Z_n$ converges and that $\sum \psi_n <\infty$.

## Evaluation settings

The natural proving ground is sequential estimation of a quantile from response/non-response data. There is an unknown distribution function $F(x)$ with $F(\theta)=\alpha$ for a target level $\alpha\in(0,1)$ and $F'(\theta)>0$; independent subjects have threshold distribution $F$. At a chosen level $x_n$ we observe only
$$y_n = \begin{cases} 1 & \text{(``response'')} \\ 0 & \text{(``non-response'')}\end{cases}, \qquad \Pr[y_n=1\mid x_n] = F(x_n),$$
so the mean response is $M(x)=F(x)$ and the noise is Bernoulli. The yardstick is whether the chosen levels $x_n$ converge to the quantile $\theta$ — convergence in quadratic mean and in probability — for an *arbitrary* unknown $F$ satisfying the monotonicity and slope conditions, starting from an arbitrary initial guess $x_1$, and using a sampling rule that is *distribution-free* (it must not depend on knowing $F$). A secondary setting groups $r$ observations at each level before moving, using the group mean $\bar y_n$ in place of a single $y_n$ — a knob between the single-step and the average-then-step extremes. The general regression problem ($Y$ associated with $x$, $M(x)$ the regression of $Y$ on $x$, of unknown form) is the same setting without the 0/1 restriction. For the maximization variant, the yardstick is reaching the maximizer $\theta$ of a regression function $M$ whose peak location is unknown, again from noisy values only.

## Code framework

The available machinery is just a noisy oracle, an empty search routine, and a step-size schedule.

```python
import numpy as np

# --- the noisy oracle: all we are allowed to touch ---
def observe(x):
    """Run one experiment at level x; return a single noisy response y
    with E[y | x] = M(x). The form of M and of the noise is unknown to us."""
    # e.g. bioassay: return 1.0 if subject_threshold(x) else 0.0   (Bernoulli, mean F(x))
    raise NotImplementedError

# --- step-size schedule: a sequence of positive gains a_1, a_2, ... ---
def step_size(n):
    # TODO: what sequence of gains makes a noisy iteration converge?
    pass

# --- the search routine we must design ---
def root_find(observe, alpha, x0, n_steps):
    """Choose levels x_1, x_2, ... sequentially from noisy responses so that
    the running level converges to the root theta of M(x) = alpha.
    Return the final level estimate."""
    x = x0
    for n in range(1, n_steps + 1):
        y = observe(x)          # one noisy measurement at the current level
        # TODO: how to update x from a single noisy y so that x -> theta?
        pass
    return x

# --- maximization variant (peak of M, gradient unobservable) ---
def maximize(observe, x0, n_steps):
    """Reach the maximizer of M using only noisy values of M (no analytic gradient)."""
    # TODO
    pass
```
