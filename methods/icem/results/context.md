## Research question

Model-based control on high-dimensional, contact-rich systems increasingly leans on
*zeroth-order*, population-based trajectory optimizers run inside a model-predictive-control
(MPC) loop: at every environment step, replan an `h`-step sequence of actions by sampling
candidate sequences, rolling each one through a (ground-truth or learned) dynamics model,
scoring them by a cost, and executing the first action of the chosen sequence. These methods
need no gradients of the model or the cost, tolerate black-box and non-differentiable
objectives, are relatively insensitive to hyperparameters, and are less prone to getting stuck
in poor local optima than gradient-based shooting. They have been shown to plan well even on
*learned* models and on *sparse-reward* manipulation tasks, sometimes matching or beating
model-free RL.

Each candidate sequence requires a full model rollout over the horizon, and these optimizers
draw many candidates per replanning step to move the sampling distribution onto good actions —
on the order of thousands of rollouts per environment step in published settings (one standard
configuration evaluates and then discards roughly fifty-five thousand individual planned action
entries *per step*). At a control rate of tens of milliseconds per step, that sample count
sets the optimizer's wall-clock per step. The question is how to allocate a sampling-based MPC
optimizer's per-step rollout budget — how to draw, select, and carry candidate action
sequences across the inner iterations and across environment steps — and how task performance
behaves as that budget is varied.

## Background

**Sampling-based planning and the MPC loop.** In trajectory optimization for MPC the decision
variable is an action sequence of length `h` (the planning horizon), one `d`-dimensional
action per timestep, so the search space is `R^{d×h}`. A *trajectory optimizer* proposes
sequences, a *forward model* unrolls each into a predicted state trajectory, and a *cost*
(or negative reward) scores it. MPC executes only the *first* action of the optimized
sequence, advances one step in the real environment, and re-solves the now-shifted `h`-step
problem. Two pieces of MPC folklore matter here: because consecutive replanning problems
overlap by `h-1` timesteps, the previous solution is a good *warm start* — the standard
"shift initialization" carries the previous step's optimized mean forward by one timestep and
fills the freed last slot with a default — and because the optimizer is re-run every step, it
only has to produce a good *first action*, not a fully converged whole-horizon plan.

**Power spectra and temporal correlation of an action sequence.** Read as a time series, an
action sequence has a power spectral density (PSD), the squared magnitude of its Fourier
transform, which says how much energy sits at each temporal frequency. A sequence of
independent Gaussian draws has, in expectation, a *flat* PSD — this is *white noise*. Noise
whose PSD falls off as a power law `PSD(f) ∝ 1/f^β` is called *colored*: `β = 0` is white,
`β = 1` "pink", `β = 2` "Brownian"/"red"; larger `β` suppresses high frequencies relative to
low ones, i.e. successive samples become positively correlated in time. A standard,
FFT-based way to draw such a sequence (Timmer & Koenig, 1995) works directly in frequency
space: draw complex Gaussian Fourier coefficients, scale each frequency bin by `f^{-β/2}` so
that the squared magnitude follows `1/f^β`, force the DC (and, for an even-length series, the
Nyquist) bin to be real, and inverse-FFT back to the time domain. Two implementation details
in the canonical routine are load-bearing: the zero-frequency bin must be handled specially
(its scale would otherwise diverge), and because zeroing the imaginary part at DC (and Nyquist)
removes half the variance there, the corresponding real parts are multiplied by `√2` so the
realized variance matches the target; the whole series is then divided by a normalizer
`σ = 2·√(Σ wₖ²)/T` (with the Nyquist weight halved for even `T`) to give unit variance.

**Temporal correlation of *actions* and the spread of trajectories.** Consider the simplest
continuous-time link between actions and states, `dx/dt = a(t)` — the state is the running
integral of the action. With white-noise (uncorrelated) actions, `x(t)` is a Brownian random
walk: independent increments cancel, so a fixed amount of action "energy" produces only a small
net displacement, and `x(t)` stays close to its start on average. Studies of natural foragers
searching sparse environments describe movement not by Brownian diffusion but by long,
temporally correlated "Lévy-walk" excursions that cover more ground for the same energy. So the
*correlation structure* of the perturbations, not just their variance, sets how far a sampled
trajectory ranges from where it started.

**Elite statistics from few samples.** A population-based optimizer that refits a
`d×h`-dimensional mean and (diagonal) standard deviation from a handful of top "elite" samples
is estimating a high-dimensional distribution from little data; the fit varies step to step.
Smoothing the refit across iterations (a momentum/EMA on the distribution parameters) is the
standard remedy. As such an optimizer's sampling distribution narrows toward an optimum, it
concentrates over successive iterations.

## Baselines

These are the prior optimizers a new sampling-based MPC method would be measured against and
react to.

**The Cross-Entropy Method, CEM (Rubinstein & Davidson 1999; de Boer, Kroese, Mannor &
Rubinstein 2005 tutorial).** Born as an adaptive importance-sampling scheme for estimating
rare-event probabilities via the cross-entropy (KL) measure, CEM was repurposed as a
derivative-free optimizer of a cost `f(x), f: R^n → R`. Maintain a sampling distribution,
canonically a Gaussian with mean `μ` and *diagonal* covariance `diag(σ²)`, `μ, σ ∈ R^n`. Per
iteration: draw `N` samples, evaluate `f`, sort, keep the best `K` ("elite set"; the `(1-ρ)`
cost quantile), and refit `μ, σ` to the elites. Iterating concentrates the distribution onto
low-cost `x`. For trajectory optimization `n = d×h` and each `x` is a whole action sequence;
because samples are drawn *independently per timestep*, the covariance is diagonal and the
per-timestep perturbations are uncorrelated — i.e. white noise along the horizon.

**CEM for MPC, with standard modifications (Chua et al. 2018, PETS; Wang & Ba 2020).** The
common way CEM is run inside MPC adds two pieces. *Shift initialization:* the next step's mean
is the previous step's optimized mean shifted by one timestep (warm start), rather than a
constant. *Momentum in the refit:* the update is smoothed,
`μ^{i+1} = α·μ^i + (1-α)·μ_elite` with `α ∈ [0,1]` over inner CEM-iterations `i` (and likewise
for `σ`). Actions are kept in range with *truncated* normal distributions whose bounds are
adapted to the action limits.

**The PETS sampling variant (Chua et al. 2018, documented only in source).** A particular
truncation rule: truncate always at `2σ`, and cap `σ` to be no larger than half the minimum
distance to the action bounds.

**Model Predictive Path Integral control, MPPI (Williams, Aldrich & Theodorou 2015; Williams
et al. 2016).** A sampling-based MPC derived from path-integral stochastic optimal control.
Rather than fit a Gaussian to elites, MPPI perturbs a nominal control sequence with Gaussian
noise `δu`, rolls out, and updates each control by an *exponentially cost-weighted average* of
the perturbations,
`u_i ← u_i + Σ_k exp(-S(τ_k)/λ)·δu_{i,k} / Σ_k exp(-S(τ_k)/λ)`,
where `S(τ_k)` is the cost-to-go of rollout `k` and `λ` is a temperature; it warm-starts by
shifting the control sequence each step and samples rollouts on a GPU to reach real-time rates.
The perturbations `δu` are temporally uncorrelated (white) Gaussian noise.

## Evaluation settings

The natural yardsticks for a sampling-based MPC optimizer, all available at the time:

- **Continuous-control MuJoCo / OpenAI Gym tasks** spanning locomotion and manipulation, with
  observation dimensionality from ~18 up to ~376 and action spaces up to ~30 dimensions:
  a half-cheetah running task (with a penalty discouraging the degenerate rolling gait), a
  humanoid stand-up task (rise from lying without falling), and a sparse-reward Fetch
  Pick&Place (reward is only the negative box-to-target distance, so no reward until the box
  moves).
- **Dexterous-hand manipulation** from the DAPG project on a simulated 24-DoF ShadowHand: a
  Door-opening task (push the handle to release a latch, then pull), its sparse-reward variant
  (reward only once the door opens), and a Relocate task (lift a ball to a target).
- **Planning on learned latent-dynamics models** trained from pixels with a PlaNet-style model
  on DeepMind Control Suite tasks (Cheetah Run, Walker Walk, Cup Catch, Reacher Easy, Finger
  Spin, Cartpole Swingup), where the optimizer plans in the model's latent space; horizon 12
  to match that model.
- **Protocol.** Use the ground-truth MuJoCo dynamics for the algorithm-only study (so model
  error does not confound the optimizer comparison) and learned models separately. Sweep the
  *per-step sample budget* (total rollouts per environment step) over a wide range, on a
  log scale, to read off how performance changes as the budget shrinks. Metrics are
  cumulative reward (locomotion/stand-up) or task success rate (manipulation), and the
  *number of rollouts per step* needed to reach a target performance level (sample
  efficiency), plus wall-clock seconds per step. Same hyperparameters across tasks where
  possible, varied deliberately per task only where a clear task property demands it.

## Code framework

The optimizer is one object plugged into an MPC loop that already exists. The loop owns the
forward model and the cost, calls the planner once per environment step to get an action
sequence, executes the first action, and advances. The substrate here is only the generic
sampling-based-MPC machinery: a Gaussian-style mean/std over the `d×h` action sequence, a
routine to draw candidate perturbations and clip actions to the action bounds, the supplied
model-rollout-and-cost evaluation, and the outer execute-and-shift loop.

```python
import torch


def sample_noise(horizon, num_samples, action_dim):
    """Draw perturbation sequences of shape [horizon, num_samples, action_dim].
    The sampling rule is the open design choice."""
    # TODO: the sampling rule we will design.
    raise NotImplementedError


class Planner:
    """Sampling-based MPC trajectory optimizer.

    Owns a sampling distribution over the d x h action sequence (mean, std).
    unroll() forward-simulates a batch of action sequences through the provided
    model; cost_function() scores them (lower is better). plan() is called once
    per environment step."""

    def __init__(self, unroll, action_dim, plan_length,
                 num_samples, n_iters, **kwargs):
        self.unroll = unroll
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples          # per-step sample budget knob
        self.n_iters = n_iters                  # inner refinement iterations
        # lazily-initialized per-step state, if the method keeps any across calls
        self._state = {}

    def cost_function(self, actions, obs_init):
        """unroll the batch of action sequences and score them (lower = better)."""
        return self.objective(self.unroll(obs_init, actions))

    def plan(self, obs_init, steps_left=None, t0=False):
        horizon = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # initialize / carry the sampling distribution over the d x h sequence
        mean = torch.zeros(horizon, self.action_dim)
        std = torch.ones(horizon, self.action_dim)

        for i in range(self.n_iters):
            noise = sample_noise(horizon, self.num_samples, self.action_dim)
            actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            # clip to the permitted action range
            cost = self.cost_function(actions, obs_init)   # [num_samples]
            # TODO: select candidates and refit/update the distribution.
            pass

        # TODO: which action sequence to return / execute.
        raise NotImplementedError
```

The open slot is the sampling, selection, update, and return rule that the `plan()` loop fills
in.
