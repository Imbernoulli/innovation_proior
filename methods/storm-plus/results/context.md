## Research question

We want to minimize a smooth but non-convex objective that we can only see through noise,

    min_x  F(x) = E_{xi ~ D}[ f(x; xi) ],

where `D` is an unknown distribution we can draw i.i.d. samples from and `f(., xi)` is one sampled
loss (a single example or a minibatch). Because `F` is non-convex, finding a global minimum is in
general intractable, so the goal is relaxed to finding an *approximate stationary point*: after `T`
oracle calls, output a point `xbar` with `E||grad F(xbar)||` as small as possible. We have access
only to first-order stochastic gradients `grad f(x; xi)`, no Hessians.

How can we choose the step size and momentum coefficient each iteration, from observed gradient
feedback alone, for the class of objectives that are an expectation over smooth losses?

## Background

By this time, stochastic gradient descent is the workhorse for training non-convex models — deep
networks, but also phase retrieval (Candes et al. 2015), non-negative matrix factorization
(Hoyer 2004), matrix completion (Ge et al. 2016). The relevant theory:

- **The two rate regimes.** For general smooth non-convex `F`, Ghadimi & Lan (2013) show SGD with a
  suitably tuned step size drives the gradient norm down at rate `O(1/T^{1/4})`, and this is tight
  (Arjevani et al. 2019 give a matching lower bound). But for the narrower, very common class where
  the objective is an *expectation over smooth losses* — i.e. each sampled `f(.,xi)` is itself
  `L`-smooth, sometimes called mean-squared smoothness — a better rate of `O(1/T^{1/3})` is
  achievable (Fang et al. 2018; Zhou et al. 2018), and this too was shown to be tight
  (Arjevani et al. 2019). Closing the gap from `1/4` to `1/3` is what *variance reduction* buys in
  the non-convex world.

- **What "variance reduction" means and why it helps.** Plain SGD's update direction `g_t =
  grad f(x_t; xi_t)` is an unbiased estimate of `grad F(x_t)` whose mean-squared error is the noise
  floor `sigma^2 = E||grad f(x;xi) - grad F(x)||^2`; that floor never decreases, which is what caps
  SGD at `1/T^{1/4}`. Variance-reduced methods replace `g_t` by a lower-variance estimate whose error
  *shrinks over the run*, so steps stay informative even near a stationary point.

- **The standing assumptions for this regime.** The analyses here lean on: bounded variance
  (`E||grad f(x;xi) - grad F(x)||^2 <= sigma^2`); per-sample `L`-smoothness
  (`||grad f(x;xi) - grad f(y;xi)|| <= L||x-y||`), which implies `F` is `L`-smooth and gives the
  descent inequality `F(y) <= F(x) + grad F(x)^T(y-x) + (L/2)||y-x||^2`; often a bound on the
  gradient norm (`||grad f(x;xi)||^2 <= G^2`); and sometimes a bound on the range of function values
  (`max_{x,y}|F(x)-F(y)| <= B`).

- **The adaptive-step-size frame.** A separate line — AdaGrad (Duchi, Hazan & Singer 2011), and the
  popular heuristics RMSProp (Tieleman & Hinton 2012) and Adam (Kingma & Ba 2014) — sets the step
  size from the *observed* gradients rather than from problem constants, e.g. AdaGrad scales the step
  inversely by `(sum_{i<=t} ||g_i||^2)^{1/2}`. In the non-convex setting these adaptive schemes were
  shown to recover SGD's rate without knowing `L` or `sigma` (Li & Orabona 2019; Ward et al. 2019;
  Reddi et al. 2018). This is the template for "let the data set the step size."

- **The role of momentum.** A widely used heuristic is momentum — step along a weighted average of
  past gradients, `d_t = (1-a) d_{t-1} + a g_t` (Sutskever et al. 2013). It is well documented that
  in the stochastic setting plain momentum does not provably improve the *rate* over SGD: the noise
  can nullify its theoretical benefit, and there is no theorem establishing that adding momentum to
  stochastic SGD helps. So momentum's empirical success in non-convex training was, at this time, a
  standing puzzle rather than a proven accelerator.

A useful algebraic tool, recurring across these analyses, is the "AdaGrad telescoping" inequality:
for non-negative `b_1, b_2, ...` and `p in (0,1)`,
`sum_i b_i / (sum_{j<=i} b_j)^p <= (1/(1-p)) (sum_i b_i)^{1-p}` (McMahan & Streeter 2010). With
`p = 1/2` it collapses a sum of `g^2 / sqrt(sum g^2)` terms to `2 sqrt(sum g^2)`; the same identity
at other powers `p` is what makes data-dependent step sizes summable.

## Baselines

The prior methods a new optimizer would be measured against, and react to.

**SVRG (Johnson & Zhang 2013; concurrently Zhang, Mahdavi & Jin 2013, Wang et al. 2013).** For
finite sums `F = (1/n) sum_i f_i`, keep a periodically refreshed snapshot `xtilde` and its **full
gradient** `mu = (1/n) sum_i grad f_i(xtilde)`, then use the control variate
`v_t = grad f_i(x_t) - grad f_i(xtilde) + mu`, which is unbiased for `grad F(x_t)` and whose variance
collapses as `x_t, xtilde -> x*`. Gives linear convergence for strongly convex finite sums.

**SARAH (Nguyen, Liu, Scheinberg & Takac 2017).** Replace SVRG's control variate by a *recursive*
estimate that reuses the same sample at consecutive iterates,
`v_t = grad f_i(x_t) - grad f_i(x_{t-1}) + v_{t-1}`, initialized each outer loop at the full gradient
`v_0 = grad F(x_0)`. The recursion threads a within-sample gradient *difference* through the estimate.

**SPIDER / SpiderBoost / PAGE (Fang et al. 2018; Li et al. 2021).** Push the SARAH-style recursive
estimator to optimal complexity: periodically compute a **mega-batch** checkpoint gradient on
`N` samples (`N` as large as `O(T)`, typically no smaller than `O(T^{2/3})`), then run many cheap
recursive steps between checkpoints, with a step size capped so that `||x_t - x_{t-1}||` stays below
a tolerance tied to `L`. This reaches the optimal `O(1/T^{1/3})` for the expectation-over-smooth-
losses setting.

**STORM (Cutkosky & Orabona 2019).** The first method to reach the optimal `O(1/T^{1/3})` for this
setting **without any checkpoint or mega-batch**. It keeps a *corrected momentum* estimate, computed
with two gradient calls on the *same* fresh sample `xi_t` at the current and previous iterate,

    d_t = grad f(x_t; xi_t) + (1 - a_t) ( d_{t-1} - grad f(x_{t-1}; xi_t) )
        = a_t grad f(x_t; xi_t) + (1 - a_t) d_{t-1}
          + (1 - a_t) ( grad f(x_t; xi_t) - grad f(x_{t-1}; xi_t) ),
    x_{t+1} = x_t - eta_t d_t.

The bracketed extra term over plain momentum is a within-sample gradient difference; tracking the
estimate error `eps_t = d_t - grad F(x_t)` gives the recursion
`eps_t = (1-a_t) eps_{t-1} + a_t (g_t - grad F(x_t)) + (1-a_t) Z_t`, with
`Z_t = (grad f(x_t;xi_t) - grad f(x_{t-1};xi_t)) - (grad F(x_t) - grad F(x_{t-1}))`. Smoothness bounds
`||Z_t|| <= 2L ||x_t - x_{t-1}|| = 2L eta_{t-1} ||d_{t-1}||`, so the error is squeezed by making the
step and the momentum small. STORM sets, AdaGrad-style,

    eta_t = k / ( w + sum_{i<=t} ||g_i||^2 )^{1/3} ,    a_{t+1} = c eta_t^2 ,

and proves the rate via a Lyapunov potential `Phi_t = F(x_t) + z_t ||eps_t||^2` with a
**time-varying** weight `z_t proportional to 1/eta_{t-1}` (a constant weight appears to force at
least one checkpoint). The constants are `k = b G^{2/3}/L`, `c = 28 L^2 + ...`,
`w = max((4Lk)^3, 2G^2, ...)`, so the step size and momentum are set using the smoothness `L` and a
gradient-norm bound `G`. Its variance-adaptive rate also carries a `(log T)^{3/4}` factor on the
leading `1/sqrt(T)` term. (Concurrently, Tran-Dinh et al. 2019 obtained the same rate with a similar
update, still using one checkpoint and knowledge of `L` and `sigma`.)

## Evaluation settings

The natural yardsticks for a method in this regime, all pre-existing:

- **Image classification on CIFAR-10** (Krizhevsky 2009) with a residual network (He et al. 2016) —
  the standard non-convex deep-learning testbed; train/test split as shipped in the standard package,
  minibatches of about 100, constant step-size schedule with no extra heuristics.
- **Metrics:** the theoretical quantity is `E||grad F(xbar)||` (stationarity) for a randomly chosen
  iterate; the empirical proxies are training loss (cross-entropy), and train/test accuracy, plotted
  against the number of passes over the data (epochs) — noting that a method making two oracle calls
  per step pays roughly twice the per-iteration forward/backward cost.
- **Protocol:** to compare optimizers fairly, fix everything to default values except the initial
  step size, sweep the step size over a common logarithmic grid, and read off each method at its best
  setting; identical initialization across optimizers. Natural comparison points are AdaGrad and Adam
  (one tuned step size each) and SGD with momentum.

## Code framework

The optimizer plugs into the corrected-momentum template that the no-checkpoint baseline already
established: maintain a gradient *estimate* `d`, take SGD-style steps along it, and refresh `d` each
iteration from two gradient calls on the same fresh sample at the current and previous iterate. That
recursion and the two-oracle-calls-per-step structure are fixed and known. The substrate below is the
generic harness with the step size `eta_t` and momentum coefficient `a_t` schedule choices left as
empty slots.

```python
import torch


def grad(model, loss_fn, x_params, batch):
    """One stochastic gradient grad f(x; xi), flattened to a single vector."""
    # forward/backward on `batch` at parameters `x_params`; return flat grad vector
    ...


class CorrectedMomentumOptimizer:
    """No-checkpoint variance-reduction template. Holds a gradient estimate d_t,
    steps x_{t+1} = x_t - eta_t d_t, and refreshes d via the corrected-momentum
    recursion using two gradient calls on the same fresh sample. The per-iteration
    step size eta_t and momentum a_t are NOT decided here -- that schedule rule is
    what we will design. No full/large-batch checkpoint is ever computed."""

    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.x = flat_params(model)      # current iterate x_t
        self.x_prev = None               # previous iterate x_{t-1}
        self.g = None                    # current stochastic gradient g_t
        self.d = None                    # gradient estimate d_t
        self.t = 0
        self.state = {}                  # any running statistics the schedule needs

    def _schedule(self, g_t, d_t):
        # TODO: the per-iteration step size and momentum we will design.
        #       From the gradient feedback seen so far (the current sample
        #       gradient g_t, the current estimate d_t, and any running
        #       statistics kept in self.state), return (eta_t, a_t).
        #       Must use only observed quantities.
        raise NotImplementedError

    def step(self, batch=None):
        if self.d is None:                                  # warm start: d_1 = g_1
            if batch is None:
                batch = draw_sample()
            self.g = grad(self.model, self.loss_fn, self.x, batch)
            self.d = self.g.clone()
            self.x_prev = self.x.clone()

        self.t += 1
        g_t = self.g                                         # grad f(x_t; xi_t)
        eta_t, a_t = self._schedule(g_t, self.d)             # <-- the slot to fill

        self.x_prev, x_old = self.x.clone(), self.x.clone()
        with torch.no_grad():
            self.x = self.x - eta_t * self.d                 # x_{t+1} = x_t - eta_t d_t

        # refresh estimate on a FRESH sample, two calls at x_{t+1} and x_t (no checkpoint)
        next_batch = draw_sample()
        g_next = grad(self.model, self.loss_fn, self.x, next_batch)     # grad f(x_{t+1}; xi_{t+1})
        g_tilde = grad(self.model, self.loss_fn, x_old, next_batch)     # grad f(x_t;     xi_{t+1})
        self.d = g_next + (1.0 - a_t) * (self.d - g_tilde)             # corrected momentum
        self.g = g_next                                                # carry g_{t+1} into next step
```

The corrected-momentum recursion, the two-calls-per-step refresh, and the SGD-style update are fixed;
the single empty slot is `_schedule` — the rule that turns observed gradient feedback into `eta_t` and
`a_t`.
