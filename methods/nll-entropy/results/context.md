# Context: training stochastic policies / output distributions that keep exploring (early 1990s connectionist RL)

## Research question

A learning agent is a network of stochastic units: it does not emit one action, it emits a
*distribution* over actions, and an output is drawn from that distribution. We want to tune
the parameters so the agent does well on a scalar performance signal — expected reward in a
function-optimization or immediate-reinforcement task, or, in a supervised setting, the
likelihood the model assigns to the right answer. The natural recipe is to follow the
gradient of that performance signal. The question is how to construct the training objective
for such stochastic policies and output distributions.

## Background

The setting is connectionist reinforcement learning. A unit's output `y` is sampled from a
distribution `g(y; θ, x)` controlled by the unit's parameters `θ` and input `x`. Two
canonical units fix ideas. A **Bernoulli unit** outputs `0/1` with `Pr{y=1}=p`, `p=f(s)`,
`s=wᵀx`, `f` a logistic squashing function. A **Gaussian unit** outputs a real number with
density `g(y; μ, σ) = (1/√(2π) σ) · exp(−(y−μ)²/(2σ²))`, with the mean `μ` and standard
deviation `σ` as the adaptable parameters (any further dependence of `μ, σ` on weights and
input is handled by the chain rule).

The performance measure for an immediate-reinforcement task is the **expected reinforcement**
`E{r | θ}`, a deterministic-but-unknown function of the parameters; the objective is to find
θ maximizing it. Gradient-based learning on this is well understood. The key analytic object
is the **characteristic eligibility** of a parameter, `e = ∂ ln g / ∂θ`, the score function
of the output distribution. For a Gaussian unit these are clean:
`∂ ln g / ∂μ = (y − μ)/σ²` and `∂ ln g / ∂σ = ((y − μ)² − σ²)/σ³`; for a Bernoulli-logistic
unit, `∂ ln g / ∂wⱼ = (y − p) xⱼ`. The reason these matter is the identity
`d/dθ E{r|θ} = E{ r · ∂ ln g/∂θ }` (the score-function / likelihood-ratio trick): you can
estimate the gradient of expected reward from sampled `(y, r)` pairs without differentiating
the reward or the environment.

A second, older idea on the table is **entropy** as a measure of how spread-out a
distribution is. For a discrete distribution `p`, `H(p) = −Σ_y p(y) log p(y)`; for a density,
the differential entropy `H = −∫ g(y) log g(y) dy = E[−log g(y)]`. Entropy is maximal for the
uniform/most-spread distribution and falls to its minimum as the distribution concentrates on
a single outcome (for a Gaussian, `H = ½ log(2π e σ²)`, increasing with `σ`; for a Bernoulli,
`H = −p log p − (1−p) log(1−p)`, maximal at `p=½` and zero at `p∈{0,1}`). The **maximum
entropy principle** (Jaynes, 1957) says: among all distributions consistent with what you
actually know, prefer the one of greatest entropy — the least-committal one, which assumes no
structure beyond what the evidence forces.

Networks of stochastic units trained by following the expected-reward gradient reliably
converge to local optima, as one expects of any gradient-follower. Tasks with hierarchical
structure, where good solutions require coordinating several units simultaneously, are a
particularly demanding test of exploration-based learning.

## Baselines

**REINFORCE (Williams, 1992).** A general class of algorithms for networks of stochastic
units that climb the expected-reward gradient *without explicitly forming a gradient
estimate*. At the end of each trial each parameter is incremented by
`Δθ = α (r − b) · ∂ ln g/∂θ`, i.e. **R**eward **I**ncrement = **N**onnegative **F**actor
(`α`) × **O**ffset **R**einforcement (`r − b`, the reward minus a baseline `b`) ×
**C**haracteristic **E**ligibility (`∂ ln g/∂θ`). The central result (Theorem 1): for any
such algorithm the expected update lies along `∇_θ E{r|θ}` — `(r−b)·∂ ln g/∂θ` is an
*unbiased* estimate of `∂ E{r|θ}/∂θ` for any baseline `b` conditionally independent of `y`.
The baseline `b` (e.g. a running average of recent reward, "reinforcement comparison")
reduces variance without biasing the direction. Specialized to a Gaussian unit, the updates
move `μ` toward sampled actions that beat the baseline and adjust `σ` by whether the squared
error `(y−μ)²` ran above or below `σ²`.

**Associative reward–penalty, A_{R-P} (Barto and colleagues, 1980s).** For Bernoulli-logistic
units, `Δwᵢⱼ = α [ r (yᵢ − pᵢ) + λ(1−r)(1−yᵢ − pᵢ) ] xⱼ`, with a reward term and a separate
penalty term (weight `λ`) that, on failure, pushes the unit toward the *opposite* of what it
did. With `λ=0` it reduces to a REINFORCE algorithm; with `λ>0` the penalty term provides a
form of persistent pressure that empirically helps it find solutions where plain
reward-following gets stuck on the harder reinforcement shapings.

**Undirected stochastic exploration heuristics (ε-greedy and fixed noise injection).** Keep
the policy mostly greedy but, with small probability or via a fixed-magnitude noise floor,
act randomly so the agent does not get trapped.

## Evaluation settings

- **Nonassociative function optimization with networks of Bernoulli units**: optimize a
  scalar function of the binary outputs of a network of stochastic units; metric is the value
  attained / probability of reaching the global optimum, over training trials. Tasks with
  deliberate **hierarchical structure** (where good solutions require coordinating several
  units and keeping options open during the search) are the stress case for premature
  collapse.
- **Single-Gaussian-unit and Bernoulli-unit immediate-reinforcement tasks** (in the style of
  Sutton, 1984): a single stochastic unit facing associative and nonassociative reward; track
  how the mean and spread evolve and whether the unit settles on the rewarding action.
- **Supervised / imitation-style tasks recast as distribution fitting**: a model emits a
  parametric distribution over the target and is scored by the log-likelihood it assigns to
  the correct target (equivalently, negative log-likelihood as a loss). The natural yardstick
  is held-out task performance of the resulting (possibly stochastic) policy.
- Protocol: fixed network architecture and initialization; a small positive learning rate;
  compare the unmodified gradient-following policy with any candidate variant under the same
  optimizer and data budget.

## Code framework

The loss plugs into a standard training loop where a model produces, per input, a parametric
**output distribution** `dist` (a `torch.distributions` object — e.g. a Gaussian or a mixture
of Gaussians over the action), and a `Loss` module turns that distribution together with the
observed target into a scalar to minimize. Everything below already exists; the one thing not
settled is what, if anything, the loss should encode about the output distribution beyond
fitting the target.

```python
import torch
import torch.nn as nn


class Loss(nn.Module):
    """Scores a predicted output distribution against an observed target.

    `dist` is a torch.distributions distribution over the target space (it exposes
    .log_prob(value) and .sample()); `target` is the observed/expert value.
    Returns a scalar to minimize.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: any coefficients the loss we design will need.
        pass

    def forward(self, dist, target):
        # Fitting term that already exists: negative log-likelihood of the target
        # under the predicted distribution (maximize the likelihood of the data).
        nll = -dist.log_prob(target).mean()
        # TODO: the term we will design and add here.
        return nll


# existing training loop the loss plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        dist = model(inputs)              # model emits a distribution over the target
        loss = loss_fn(dist, targets)     # score it against the observed target
        loss.backward()
        optimizer.step()
```

The training loop hands `forward` a distribution and a target; `forward` returns the scalar
the optimizer descends. The fitting term is fixed; the open slot is whatever else the loss
should encode about the *shape* of the distribution it is training.
