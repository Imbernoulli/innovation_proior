# iCEM (improved Cross-Entropy Method), distilled

iCEM is a sample-efficient drop-in upgrade to the Cross-Entropy Method used as a
sampling-based MPC trajectory optimizer. It keeps CEM's gradient-free
sample-evaluate-refit loop but reclaims the rollouts CEM wastes and replaces white-noise
action sampling with temporally correlated *colored* noise, aiming for the same control quality
at a fraction of the per-step rollout budget, so the optimizer can be used for real-time planning.

## Problem it solves

Sampling-based MPC optimizers (CEM, MPPI) plan by drawing many candidate action sequences,
rolling each through a (ground-truth or learned) dynamics model, scoring, and executing the
first action of the best. They work well even on learned models and sparse rewards, but need
thousands of model rollouts per environment step, which makes real-time control impossible.
Goal: comparable control with tens-to-hundreds of rollouts per step.

## Key ideas

Two structural wastes in CEM/MPPI are fixed.

1. **Colored-noise action sampling (the central component).** CEM perturbs each
   timestep with independent (white) Gaussian noise; for `dx/dt = a(t)` white actions
   integrate to a Brownian random walk, which stays near its start, so on sparse reward most
   rollouts never reach informative states. Instead sample action noise with a power-law
   spectrum `PSD_a(f) ∝ 1/f^β`. Then the state spectrum is `PSD_x(f) = PSD_a(f)/(4π²f²) ∝
   1/f^{β+2}` (using `F[dx/dt] = i2πf·F[x]`), so larger `β` puts more energy at low
   frequencies → smoother, far-ranging trajectories at the *same* action variance. `β=0` is
   white; `β≈2–4` for smooth/low-frequency control (manipulation), small `β` for
   high-frequency control (running). The colored sequence is drawn in the rFFT domain
   (Timmer & Koenig 1995).
2. **Memory — reuse evaluated rollouts instead of discarding them.**
   - *Keep elites (across inner CEM-iterations):* re-add a fraction (`ξ=0.3`) of each
     iteration's elite set to the next iteration's candidate pool.
   - *Shift elites (across environment steps):* carry a fraction of the last step's elite set
     forward, dropping the executed first timestep and appending a fresh last action.
   - Only a *fraction* in both cases: full (low-variance) elite sets would dominate the first
     iteration and collapse `σ`.

Plus four smaller fixes: **sample decay** `N_i = max(N·γ^{-i}, 2K)` (`γ=1.25`), exploiting
CEM's own narrowing to free up extra iterations within a fixed budget; **execute the best
trajectory in the final scored buffer** (not the never-evaluated mean), with the **mean added
as a sample at the last iteration only** so it can win without over-narrowing the variance;
**clip** to action bounds instead of truncated normals so boundary actions get sampled; and
the standard CEM-MPC **shift-initialized mean** and **refit momentum** `α=0.1`.

## Final algorithm

```
for each env step t:
  if t == 0: μ ← const else μ ← shift(μ) (repeat last);   σ ← σ_init  (everywhere)
  if t > 0:  shifted_elites ← shift(last elites): drop first step, append fresh last action
  N_i ← N
  for inner iteration i = 0 .. iters-1:
    if i > 0: N_i ← max(2K, ⌊N_i / γ⌋)                        # population decay
    action_sequences ← clip(μ + σ ⊙ C^β(d,h)), N_i of them     # colored noise + clip
    if i == last: action_sequences[0] ← μ                      # add mean, last iter only
    if i == 0: action_sequences ← action_sequences ∪ frac(shifted_elites)
    simulated_paths ← rollout(action_sequences)
    if i > 0: simulated_paths ← simulated_paths ∪ frac(kept_elites)
    costs ← cost_fn(simulated_paths)
    best_idx ← argmin(costs)
    elites ← best K rollouts from simulated_paths
    μ, σ ← (1-α)·fit(elites) + α·(μ, σ)                       # refit with momentum
    kept_elites ← elites
  execute first action of simulated_paths[best_idx]            # best-action, not mean
  shift μ for next step, reset σ to σ_init, save kept_elites
```

Default hyperparameters (fixed across tasks except `β`): `K=10`, `σ_init=0.5`, `α=0.1`,
`γ=1.25`, `ξ=0.3`, horizon `30` (ground-truth model) / `12` (learned latent model); `β` per
task (e.g. 0.25 running, 2.0 humanoid stand-up, 2.5 door, 3.0–3.5 grasping/relocate).

## Colored-noise sampler (Timmer & Koenig, rFFT domain)

```python
from typing import Iterable, Optional, Union
from numpy import integer, newaxis, sqrt
from numpy import sum as npsum
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState


def powerlaw_psd_gaussian(
    exponent: float,
    size: Union[int, Iterable[int]],
    fmin: float = 0.0,
    random_state: Optional[Union[int, Generator, RandomState]] = None,
):
    """Gaussian noise, PSD (1/f)**exponent, unit variance; last axis = time."""
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    samples = size[-1]
    f = rfftfreq(samples)                          # non-negative freqs, unit sample rate
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)            # lowest finite frequency
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    s_scale = f
    ix = npsum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]                 # flatten below cutoff, including DC
    s_scale = s_scale ** (-exponent / 2.0)         # |coef|^2 ~ 1/f^exponent

    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0              # Nyquist counts half for even length
    sigma = 2 * sqrt(npsum(w ** 2)) / samples      # unit-variance normalizer

    size[-1] = len(f)
    s_scale = s_scale[(newaxis,) * (len(size) - 1) + (Ellipsis,)]
    normal_dist = _get_normal_distribution(random_state)
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)
    if not (samples % 2):                           # even length: Nyquist must be real
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)                      # restore variance lost by zeroed imag
    si[..., 0] = 0                                  # DC must be real
    sr[..., 0] *= sqrt(2)
    s = sr + 1j * si
    return irfft(s, n=samples, axis=-1) / sigma


def _get_normal_distribution(random_state: Optional[Union[int, Generator, RandomState]]):
    if isinstance(random_state, (integer, int)) or random_state is None:
        return default_rng(random_state).normal
    if isinstance(random_state, (Generator, RandomState)):
        return random_state.normal
    raise ValueError("random_state must be one of integer, Generator, RandomState, or None.")
```

## Optimizer

```python
import numpy as np
import colorednoise
from controllers.mpc import MpcController
from misc.rolloutbuffer import RolloutBuffer


class MpcICem(MpcController):
    """CEM-MPC with colored sampling, elite reuse, population decay, and best-action execution."""

    def beginning_of_rollout(self, *, observation, state=None, mode):
        super().beginning_of_rollout(observation=observation, state=state, mode=mode)
        self.mean = self.get_init_mean(relative=True)
        self.std = self.get_init_std(relative=True)
        self.elite_samples = RolloutBuffer()
        self.was_reset = True

    def get_init_mean(self, relative):
        if relative:
            return np.zeros(self.dim_samples) + (
                self.env.action_space.high + self.env.action_space.low
            ) / 2.0
        return np.zeros(self.dim_samples)

    def get_init_std(self, relative):
        if relative:
            width = (self.env.action_space.high - self.env.action_space.low) / 2.0
            return np.ones(self.dim_samples) * width * self.init_std
        return self.init_std * np.ones(self.dim_samples)

    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        if self.noise_beta > 0:
            assert self.mean.ndim == 2
            samples = colorednoise.powerlaw_psd_gaussian(
                self.noise_beta,
                size=(num_traj, self.mean.shape[1], self.mean.shape[0]),
            ).transpose([0, 2, 1])                  # [N, d, h] -> [N, h, d]
        else:
            samples = np.random.randn(num_traj, *self.mean.shape)
        samples = np.clip(
            samples * self.std + self.mean,
            self.env.action_space.low,
            self.env.action_space.high,
        )
        if time_slice is not None:
            samples = samples[:, time_slice]
        return samples

    def prepare_action_sequences(self, *, obs, num_traj, iteration):
        action_sequences = self.sample_action_sequences(obs, num_traj)
        if self.use_mean_actions and iteration == self.opt_iter - 1:
            action_sequences[0] = self.mean          # substitute one sample by the mean
        return action_sequences

    def elites_2_action_sequences(self, *, elites, obs, fraction_to_be_used=1.0):
        actions = elites.as_array("actions")
        reused_actions = actions[:, 1:]              # drop the executed first action
        num_elites = int(reused_actions.shape[0] * fraction_to_be_used)
        reused_actions = reused_actions[:num_elites]
        last_actions = self.sample_action_sequences(
            obs=obs, num_traj=num_elites, time_slice=slice(-1, None)
        )
        return np.concatenate([reused_actions, last_actions], axis=1)

    def get_action(self, obs, state, mode="train"):
        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state
        )

        num_sim_traj = self.num_sim_traj
        for i in range(self.opt_iter):
            if i > 0:
                num_sim_traj = max(
                    self.elites_size * 2, int(num_sim_traj / self.factor_decrease_num)
                )
            action_sequences = self.prepare_action_sequences(
                obs=obs, num_traj=num_sim_traj, iteration=i
            )
            if i == 0 and self.shift_elites_over_time and self.elite_samples:
                shifted = self.elites_2_action_sequences(
                    elites=self.elite_samples,
                    fraction_to_be_used=self.fraction_elites_reused,
                    obs=obs,
                )
                action_sequences = np.concatenate([action_sequences, shifted], axis=0)

            simulated_paths = self.simulate_trajectories(
                obs=obs, state=self.forward_model_state, action_sequences=action_sequences
            )
            if i > 0 and self.keep_previous_elites:
                simulated_paths.extend(
                    self.elite_samples[: int(len(self.elite_samples) * self.fraction_elites_reused)]
                )

            costs = self.trajectory_cost_fn(self.cost_fn, simulated_paths)
            best_traj_idx = np.argmin(costs)
            self.update_distributions(simulated_paths, costs)

        executed_action = simulated_paths[best_traj_idx]["actions"][0]
        self.mean[:-1] = self.mean[1:]              # shift initialization for next step
        self.mean[-1] = self.compute_new_mean(
            obs=simulated_paths[best_traj_idx]["observations"][-1]
        )
        self.std = self.get_init_std(relative=True)
        return executed_action

    def compute_new_mean(self, obs):
        return self.mean[-1]

    def update_distributions(self, sampled_trajectories, costs):
        elite_idxs = np.array(costs).argsort()[: self.num_elites]
        self.elite_samples = RolloutBuffer(rollouts=sampled_trajectories[elite_idxs])
        elite_sequences = self.elite_samples.as_array("actions")
        self.mean = (1 - self.alpha) * elite_sequences.mean(0) + self.alpha * self.mean
        self.std = (1 - self.alpha) * elite_sequences.std(0) + self.alpha * self.std
```

## Relation to prior methods

- **Vanilla CEM for trajectory optimization** samples a fixed-size i.i.d. Gaussian population,
  refits to the best `K`, repeats, and executes the first action of the final mean sequence.
  It has white-noise actions, no elite memory, and no population decay.
- **CEM-MPC / PETS** add shift-initialized means and refit momentum; the common action-limiting
  path uses truncated normals, with PETS using a `2σ` truncation rule and a cap on `σ`. Low-budget
  CEM baselines can already clip and execute best sampled actions, but still keep white noise,
  fixed sample counts, and no elite reuse.
- **MPPI** replaces the elite fit with an exponential cost-weighted average of perturbations
  and warm-starts by shifting the control sequence, but its perturbations are also white and
  it still needs many rollouts (paid down with GPU sampling, not fewer samples).
