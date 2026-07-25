The problem is making sampling-based model-predictive control affordable. In this setting we have a forward dynamics model that can be rolled out, but we do not rely on gradients through it. At every environment step we must pick an action by optimizing a short horizon action sequence, which we do by sampling candidate sequences, simulating each one through the model, and scoring the resulting trajectories. The standard Cross-Entropy Method works well even on learned models and sparse rewards, but published configurations consume thousands of rollouts per control step, which makes real-time robot planning impossible. The concrete goal is to obtain similar control quality with tens to hundreds of rollouts per step.

The baseline waste comes from two shared habits. First, CEM and MPPI perturb action sequences with temporally uncorrelated white noise. Because actions drive state changes through integration, white-noise actions produce Brownian state trajectories whose net displacement grows only like the square root of the horizon; on sparse-reward tasks most rollouts never reach a state where the cost provides any gradient signal. Second, every replanning step discards almost all of the rollouts it just evaluated, including the best ones, and starts the next step from little more than a shifted mean. The information bought by those expensive simulations is used once and thrown away.

The method I propose is iCEM, the improved Cross-Entropy Method. It keeps CEM's gradient-free sample-evaluate-refit loop, but it replaces the noise source with temporally correlated colored noise and reclaims the rollouts that CEM wastes. The central design parameter is a single scalar beta that controls the power-law spectrum of the sampled perturbations, PSD_a(f) proportional to 1/f^beta. With beta equal to zero we recover white noise; with beta around two to four the perturbations become smooth and low-frequency. Passing through the integrator reddens the state spectrum further to 1/f^{beta+2}, so the same action variance produces trajectories that range farther and faster. This is exactly what sparse-reward exploration needs: instead of diffusing in place, sampled sequences make committed low-frequency excursions that are more likely to reach informative states. Beta is task-interpretable, small for high-frequency control and large for smooth reaching or manipulation.

iCEM also treats elite sequences as a memory rather than a disposable byproduct. A fraction of each inner iteration's elite set is carried into the next iteration, and a fraction of the previous environment step's elite set is shifted forward and reused at the start of the next planning call. Only a minority is reused so that the concentrated elites do not collapse the search variance prematurely. Within a fixed rollout budget, the population size is decayed geometrically across inner iterations, because CEM's own narrowing makes late iterations less informative per sample; the saved budget buys extra refinement iterations. The mean is updated with momentum from the elite fit to reduce noise in the high-dimensional distribution estimate, and actions are clipped to bounds rather than drawn from a truncated normal so boundary actions remain well represented. At the end of planning we execute the best evaluated sequence rather than the untested mean, but we also evaluate the mean at the last iteration so it can win when it is genuinely clean.

The colored-noise sampler is the one genuinely non-trivial primitive, so it has to get the spectral bookkeeping exactly right: build the sequence in the rFFT domain, clamp the DC bin to the lowest resolvable frequency, restore the variance the zeroed-imaginary-part convention removes at DC (and at Nyquist for even lengths) with the $\sqrt{2}$ correction, and normalize the whole thing to unit variance so that $\beta$ only sets the shape of the noise and $\sigma$ alone sets its scale.

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

That sampler is a drop-in replacement for CEM's white-noise draw, same interface, so the rest of the controller keeps CEM's machinery intact and only plugs the colored noise, the elite memory, the decay schedule, and the best-action rule into the loop:

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
