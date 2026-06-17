The thing that actually hurts is the sample count. I have a zeroth-order trajectory optimizer running inside an MPC loop: every environment step I have to optimize an `h`-step action sequence by sampling candidate sequences, rolling each one through the dynamics model, scoring it, and executing the first action of the best one. The Cross-Entropy Method does this honestly — sample `N` sequences from a Gaussian, keep the best `K` elites, refit the mean and standard deviation to those elites, repeat — and it works astonishingly well, even on learned models, even on sparse-reward manipulation, even matching model-free RL. But it pays for that with rollouts. To estimate the elite quantile well enough that the refit moves the distribution onto good actions, `N` has to be large; in a standard MPC configuration the optimizer spends thousands of trajectory rollouts, amounting to roughly fifty-five thousand sampled action entries that are then discarded at a single environment step. Each trajectory is a full horizon-length model rollout. At a control period of tens of milliseconds, that is hopeless for real-time control on a robot. So the whole question is narrow and concrete: can I keep CEM's gradient-free robustness but get the same control quality out of tens-to-hundreds of rollouts per step instead of thousands? The budget is the enemy. I'm not going to invent a new optimizer class; I'm going to ask where each rollout's information leaks away and plug the leaks.

Let me look at what the alternatives do, because maybe one of them already spends samples better. MPPI is the other obvious one — Williams, Aldrich, Theodorou — derived from path-integral optimal control. Instead of fitting a Gaussian to elites, it perturbs a nominal control sequence `u` with Gaussian noise `δu`, rolls out, and updates each control by an exponentially cost-weighted average of the perturbations, `u_i ← u_i + Σ_k exp(-S(τ_k)/λ) δu_{i,k} / Σ_k exp(-S(τ_k)/λ)`, with `S(τ_k)` the cost-to-go of rollout `k` and `λ` a temperature. It warm-starts by shifting the control sequence each step and hits real-time by sampling rollouts on a GPU. But notice: the perturbations `δu` are independent Gaussians per timestep, and the soft-weighted average, just like CEM's elite fit, is a high-variance estimator unless `K` is large. MPPI buys its real-time rate with hardware, not with frugality. That's not the lever I want — I want to spend fewer samples, not the same number faster. So both of the methods I'd be measured against share the same two structural habits, and I should name them precisely because they're going to be the leaks. First, the candidate sequences are perturbed by *temporally uncorrelated* noise — white noise along the horizon, because CEM samples each timestep independently from a diagonal Gaussian and MPPI's `δu` are i.i.d. Second, every replanning step *discards* almost all of the rollouts it just evaluated — the whole fitted distribution, every elite set from every inner iteration, gone, and the next step starts over from (at best) the shifted mean. Two leaks: bad sample shape, and amnesia. Let me attack the shape first, because I suspect it's the bigger one.

What does a single sampled action sequence actually do when I feed it to the model? It drives the state somewhere. Think about the crudest possible link between actions and state, just to get intuition: `dx/dt = a(t)`, the state is the running integral of the action. If `a(t)` is white noise — independent increments — then `x(t)` is a Brownian random walk. And the defining fact about a Brownian walk is that it doesn't go anywhere: the expected squared displacement grows only like time, the increments are independent so they cancel, and a fixed budget of action "energy" buys only a small net excursion from the start. That's a problem, and it's exactly the wrong problem for my setting. When reward is sparse — the box hasn't moved, the door is shut, there's no gradient signal anywhere near the current state — the only way a rollout earns any information is by *reaching* somewhere the cost actually distinguishes. A diffusive rollout that jitters around its starting point almost never gets there, so almost all of my precious `N` rollouts come back saying "cost is flat, learned nothing." I'm spending samples on trajectories that structurally cannot explore.

So the question becomes: at the same per-step action variance, can I get sampled trajectories that range *farther*? There's a beautiful piece of evidence from outside control entirely — foraging animals searching sparse environments don't move by Brownian diffusion, they move by long, temporally correlated excursions, Lévy-style walks, which cover far more ground per unit of energy. The common thread is correlation in time. A Brownian walk has independent steps; a far-ranging walk has steps that persist in the same direction for a while. So the thing I want to control isn't the variance of the per-step action — it's the *temporal correlation* of the action sequence. White noise has none. I want noise that's positively autocorrelated across the horizon, so that a sampled sequence pushes consistently in some direction long enough to actually move the state somewhere the cost can see.

How do I parameterize "amount of temporal correlation" cleanly? Read the action sequence as a time series and look at it in the frequency domain. Its power spectral density — the squared magnitude of its Fourier transform — tells me how much energy sits at each temporal frequency. White noise has, in expectation, a *flat* PSD: equal energy at all frequencies, including the highest one, which corresponds to the action flipping sign every single timestep. That high-frequency content is exactly the diffusive jitter. If I want the sequence to persist in a direction, I want to *suppress* the high frequencies relative to the low ones. A clean one-knob family that does this is a power-law PSD,

  PSD_a(f) ∝ 1/f^β,

where `β = 0` recovers white noise and `β > 0` tilts energy toward low frequencies. `β = 1` is "pink", `β = 2` is "Brownian"/"red"; bigger `β` means stronger correlation, smoother sequences. One scalar, `β`, dials from white all the way to very smooth. Good — that's the parameterization.

Now I should check that this actually does what I claimed for the *state*, not just the action, because that's the whole point. Carry the PSD through the integrator `dx/dt = a(t)`. In the frequency domain the derivative is multiplication by `i 2π f`, so `F[dx/dt](f) = i 2π f · F[x](f)`, which means `F[a](f) = i 2π f · F[x](f)`, hence `F[x](f) = F[a](f) / (i 2π f)`. Take squared magnitudes:

  PSD_x(f) = |F[x](f)|² = |F[a](f)|² / (4 π² f²) = PSD_a(f) / (4 π² f²).

Substitute the colored action PSD `PSD_a ∝ 1/f^β`:

  PSD_x(f) ∝ 1 / f^{β+2}.

So the state trajectory's spectrum is the action spectrum reddened by two more powers of `f`. Even white actions (`β=0`) give `PSD_x ∝ 1/f²`, which is the Brownian walk — consistent, good sanity check. And as I crank `β` up, the state spectrum piles even more energy into the low frequencies, i.e. the trajectory makes larger, smoother, longer-range excursions — at *fixed action variance*, because `β` only redistributes the action energy across frequencies, it doesn't add any. That's exactly the trade I wanted: same energy, more reach. The single knob `β` directly controls how far a sampled rollout ranges from its start.

I can sharpen the "more correlation, more reach" intuition with the autocorrelation. By Wiener-Khinchin the autocorrelation of the action is the inverse transform of its PSD, `C(τ) = F^{-1}[PSD_a(f)]`. With a power-law PSD this is self-similar: rescale time `τ → sτ`, and using the frequency-scaling property of the Fourier transform, `C(sτ) = F^{-1}[(1/s) PSD_a(f/s)] = F^{-1}[(1/s) s^β PSD_a(f)] = s^{β-1} C(τ)`. In the stationary range this is the usual power-law lag decay; for the larger exponents I may use in a finite planning horizon, the important point is the same low-frequency self-similarity. There is no built-in correlation length that caps how long the sequence persists; that is precisely the property a Lévy-style far-ranging walk has and a Brownian one does not.

So the design decision is settled and I can see *why*: replace CEM's per-timestep white Gaussian sampling with colored noise of exponent `β`, because white actions integrate to short Brownian excursions that can't reach sparse reward, and colored actions reach far on the same energy. And `β` should be task-dependent in an interpretable way — a task that needs fast, high-frequency control (a running gait that reverses direction often) wants the energy left at high frequencies, so small `β`, near white; a task that needs smooth, low-frequency control (reach the object, open the door) wants the energy pushed low, so large `β`, around 2 to 4. One number with a physical meaning per task, not a tuning mystery.

Now, *how* do I actually draw a length-`h` sequence with PSD `1/f^β`? I don't want to filter white noise with some clumsy time-domain convolution. The clean move is to build it in frequency space directly, because the PSD *is* a statement about frequency-domain magnitudes. Start from white noise `a`, whose Fourier transform `F[a]` has, in expectation, flat magnitude across frequencies. If I scale each frequency bin by `f^{-β/2}` and transform back,

  ā = F^{-1}[ f^{-β/2} · F[a] ],

then the new PSD is `PSD_ā(f) = |f^{-β/2}|² PSD_a(f) = f^{-β} · (flat) ∝ 1/f^β`. Exactly the target. So the recipe is: draw a random spectrum, scale bin `k` by `f_k^{-β/2}`, inverse-FFT. Equivalently — and this is what I'll implement — I don't even need to start from a white time series and transform it; I can draw the random Fourier coefficients directly as complex Gaussians and scale them, since the FFT of Gaussian white noise *is* complex Gaussian. For a real output sequence I use the real FFT, so I only generate the non-negative frequency bins and the inverse rFFT enforces Hermitian symmetry for me.

Let me be careful here, because the zero-frequency bin will bite me. The scaling is `f^{-β/2}`, and at `f = 0` that's `0^{-β/2} = ∞`. I have to regularize the DC bin. The standard fix is to clamp the lowest frequency to the smallest resolvable nonzero frequency `1/T` (with `T = h` the sequence length) — i.e. treat DC as if it were the first real frequency bin, so its scale is finite. That's a low-frequency cutoff; below it the spectrum is flat instead of blowing up, which is physically the right thing (you can't have infinite DC power in a finite, zero-mean-ish sequence).

There's a second subtlety about the realness of the output, and it controls the *variance*, so I can't hand-wave it. For the inverse rFFT to produce a real time series, the DC coefficient must be real, and if `T` is even the Nyquist coefficient (the very top bin, `f = 0.5`) must also be real. So I draw a real part `sr` and an imaginary part `si` for every bin, each Gaussian scaled by `f^{-β/2}`, and then I force `si[DC] = 0` (and `si[Nyquist] = 0` for even `T`). But zeroing the imaginary part removes half the coefficient variance at those bins — a generic complex bin contributes both `sr` and `si`, and I just deleted the `si` half. If I leave it, the realized power at DC and Nyquist is half what the scaling intended. The fix is to compensate: multiply the surviving real part there by `√2`, so `Var(√2·sr) = 2 Var(sr)` restores the full intended power at that bin. This is small but load-bearing — without it the real-only bins come out weaker than designed. Keep the `√2`.

Finally I want the sequence normalized to unit variance, so that the planner's own standard deviation `σ` is the only thing that sets the action scale and `β` only sets the *shape*. The variance of the realized time series is set by the non-DC spectral magnitudes; working it through, the normalizer used by the generator is `σ_norm = 2·√(Σ_k w_k²)/T`, where `w_k` are the scaled magnitudes of the positive-frequency bins excluding DC, and the Nyquist weight is halved for even `T` (it's a single bin shared by `±` Nyquist, so it shouldn't be double-counted). Divide the output by `σ_norm` and I get a mean-zero-in-expectation, unit-variance, `1/f^β`-spectrum sequence of length `h`, one per action dimension, which I then scale by `σ` and add to the mean exactly the way CEM adds its white Gaussian samples. So the colored-noise sampler is a drop-in replacement for the white sampler — same interface, `C^β(d, h)` returns `d` independent length-`h` sequences — and that's important, because it means I can keep the entire rest of the CEM machinery and just swap the noise source.

That handles the *shape* leak. Now the *amnesia* leak: every step, CEM evaluates a huge number of rollouts and then throws essentially all of them away. With a small budget I cannot afford that. Where exactly is the discarded information?

There are two separate discards, at two timescales, and I should treat them separately. The first is *within a single replanning step, across the inner CEM-iterations*. Each inner iteration produces an elite set — the best `K` sequences under the model. Standard CEM uses those only to refit `μ, σ`, then discards the actual elite sequences and draws a fresh population from the new distribution. But those elite sequences were *good* — they were the lowest-cost ones found so far. After I refit and the distribution shifts a little, most of the new population will be worse than the elites I just deleted. Why throw away known-good sequences? So: keep them. At inner iteration `i+1`, add a fraction of iteration `i`'s elite set into the candidate pool, alongside the fresh samples from the updated distribution. Now good sequences survive long enough to compete again instead of being rediscovered. But I should not add *all* the elites back, and I can see why if I think about variance: the elites have, by construction, *small* spread (they're the concentrated best), so if they dominate the pool in the first inner iteration, the refit will collapse `σ` almost immediately and kill exploration before it's even started. So add only a *fraction* — keep a memory, but keep it a minority of the pool so fresh exploration still drives the fit. A fraction around 0.3 is the natural compromise: enough to retain the good ones, not so much that they swamp the fresh draws.

The second discard is *across environment steps*. MPC already does the cheap part of this — shift-initialize the mean: the next step's planning problem overlaps the current one by `h-1` timesteps, so carry the optimized mean forward by one timestep, repeat the last entry to fill the freed slot, and reset `σ` back to its initial value everywhere (because the new last timestep is genuinely unexplored, and resetting `σ` re-opens exploration for the shifted problem). But the *mean* is only the center of the distribution; the actual *elite sequences* I found last step are even more valuable, and those get discarded. So shift them too: take a fraction of the last step's final elite set, drop the first timestep (it's been executed, it's in the past), and append a freshly sampled action at the end to fill the horizon. That gives me, for free, a handful of genuinely-good warm-start sequences for the new step — not just a shifted mean, but shifted *solutions*. Same reasoning as before for why only a fraction: a full elite set, having tiny spread, would dominate and collapse the first iteration's variance. Use the same fraction, ~0.3.

Now a cluster of smaller fixes, each closing a specific small leak, and each one I can justify rather than just assert.

The population is fixed-size across inner iterations in standard CEM, and that's wasteful with a tight budget. CEM's `σ` *automatically shrinks* as the iterations proceed and the elites concentrate — that's the whole mechanism. So a late iteration is sampling from an already-narrow distribution, where the marginal value of each extra sample is low; I don't need as many samples late as I needed early. So decay the population geometrically: `N_i = max(N·γ^{-i}, 2K)` for a decay factor `γ > 1`. The `max` with `2K` floors it — I always need at least about twice the elite count for the elite selection and the Gaussian refit to be meaningful. The budget payoff is direct: because the later iterations are cheaper, for the *same total rollout budget* I can afford *more* inner iterations than fixed-`N` CEM could, i.e. more refinement, which is exactly what I want when every rollout is scarce. A modest `γ` like 1.25 gives a gentle decay — enough to free up iterations, not so steep that late iterations are too sparse to fit.

Next, which action do I actually execute? Standard CEM executes the first action of the *mean* sequence. But the mean was never actually evaluated — it's the centroid of the elites, a sequence that may correspond to no good rollout at all, especially in high dimensions where the mean of a cloud can sit in a low-density region. CEM as an MPC controller has been quietly mis-using its own output: CEM was born to *estimate a distribution*, but I'm using it to *pick the single best control*, and for that the right answer is the first action of the best *evaluated* sequence — the one I actually rolled out and know is good — not the never-tested mean. So execute the first action of the best elite sequence. But I don't want to lose the mean entirely, because for some tasks the mean is a clean, smooth sequence that's genuinely better than any noisy sample (manipulation, reaching, anything wanting a tidy trajectory). So at the *last* inner iteration, add the mean itself as one of the samples — then it gets evaluated and can win on its own merits, and if it does, the best-action rule picks it. Why only the last iteration and not every iteration? Because if I inject the mean into every iteration, it keeps pulling the elite set toward the current center and quickens the variance collapse — the same over-narrowing failure as adding back too many elites. Adding it only at the end gives it a fair evaluation without biasing the search. (And if the mean survives into the final elite set, the cross-step elite shift carries it forward automatically.)

Last small one: how to enforce action bounds. CEM uses truncated normals with adapted bounds. But a truncated normal *under-samples the boundary* — it reshapes the whole distribution to fit inside the box, so it almost never proposes the maximal action, and for control near limits (push as hard as you can) that's exactly the action you want to try. Simpler and better: sample from the *unmodified* normal (or colored-noise) distribution and just *clip* the results to the action interval. Clipping piles probability mass right at the boundary, so maximal actions get sampled frequently, which is what saturated control needs. Also it keeps the colored-noise spectrum intact before clipping, rather than tangling it up with a truncation.

Let me also keep the one standard CEM-MPC piece that's clearly right and not throw the baby out: the momentum in the refit. The elite set is small and it's setting a `d×h`-dimensional mean and std, so the per-iteration fit is noisy; smooth it, `μ ← (1-α)·μ_elite + α·μ_prev` and the same for `σ`, with a small `α` like 0.1. This is just variance reduction on the distribution parameters, and it costs nothing.

Now let me assemble the whole thing and make sure the pieces compose, because there's an ordering question: within an iteration I have fresh colored samples, plus the mean (last iter only), plus kept elites from the previous inner iteration, plus — at iteration 0 — shifted elites from the previous step. Shifted elites have a new final action, so they join the fresh action-sequence batch before simulation. Kept elites from the previous inner iteration are even cheaper: at the same environment state I already simulated them, so I can append their rollout records to the current simulated-path buffer instead of spending model calls on them again. After the combined buffer is scored, the best `K` rollouts become the new elite set, which both refits the distribution (with momentum) and is stored for the next iteration's keep-step. At the final inner iteration, the best trajectory in that final scored buffer is the one whose first action I execute. After the loop, shift the mean for the next environment step, fill the final mean slot by the default continuation rule, and reset the standard deviation back to its initial value so the shifted problem can explore again. Let me write it as the algorithm:

  for each env step t:
    if t == 0: μ ← const;   else: μ ← shift(μ) (repeat last);   σ ← σ_init everywhere
    if t > 0:  prev_shifted_elites ← shift(last step's elites), drop first, append fresh last
    N_i ← N
    for inner iteration i in 0..(iters-1):
      if i > 0: N_i ← max(2K, ⌊N_i / γ⌋)              # population decay
      action_sequences ← clip(μ + σ ⊙ C^β(d, h)),  N_i of them
      if i == last: action_sequences[0] ← μ          # substitute mean, last iter only
      if i == 0 and prev_shifted_elites: action_sequences ← action_sequences ∪ frac(prev_shifted_elites)
      simulated_paths ← rollout(action_sequences)
      if i > 0 and kept_elites: simulated_paths ← simulated_paths ∪ frac(kept_elites)
      costs ← cost_function(simulated_paths)
      best_idx ← argmin(costs)
      elites ← best K rollouts from simulated_paths
      μ, σ ← (1-α)·fit(elites) + α·(μ, σ)             # refit with momentum
      kept_elites ← elites
    execute first action of simulated_paths[best_idx] # best-a, not the mean
    μ ← shift(μ);  fill the final slot;  σ ← σ_init
    save kept_elites for next step's shift

Every line traces to a leak I identified: colored noise fixes the white-noise reach leak; keep/shift elites fix the within-step and across-step amnesia leaks; decay buys extra iterations from the fixed budget; best-a and add-mean fix the never-evaluated-mean leak; clip fixes the boundary under-sampling; momentum reduces refit variance. None of it is a new optimizer — it's CEM with every wasted rollout reclaimed and a smarter noise source.

Now let me write the colored-noise sampler as real code, because it's the one genuinely non-trivial primitive and I have to get the spectral bookkeeping exactly right. Build it in the rFFT domain, regularize DC, do the `√2` correction at DC (and Nyquist for even `T`), normalize to unit variance:

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
    """Gaussian noise with power spectral density (1/f)**exponent, unit variance.
    size = (..., T): the LAST axis is time and carries the 1/f^exponent spectrum;
    all other axes are independent sequences. Frequency-domain (Timmer-Koenig) sampler."""
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    samples = size[-1]                            # samples per sequence

    f = rfftfreq(samples)                         # non-negative frequencies, assume unit rate
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)           # lowest finite frequency
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    s_scale = f
    ix = npsum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]                # flatten below cutoff, including DC
    s_scale = s_scale ** (-exponent / 2.0)        # scale each bin so |coef|^2 ~ 1/f^exponent

    # theoretical std of the realized series, to normalize to unit variance
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0            # Nyquist bin counts half when samples is even
    sigma = 2 * sqrt(npsum(w ** 2)) / samples

    size[-1] = len(f)                             # one complex Fourier coefficient per bin
    s_scale = s_scale[(newaxis,) * (len(size) - 1) + (Ellipsis,)]

    normal_dist = _get_normal_distribution(random_state)
    sr = normal_dist(scale=s_scale, size=size)    # real part of each scaled coefficient
    si = normal_dist(scale=s_scale, size=size)    # imaginary part

    if not (samples % 2):                         # even length: Nyquist coefficient must be real
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)                    # restore variance lost by zeroing imag part
    si[..., 0] = 0                                # DC coefficient must be real
    sr[..., 0] *= sqrt(2)                         # restore variance lost by zeroing imag part

    s = sr + 1j * si
    return irfft(s, n=samples, axis=-1) / sigma   # back to time domain, scaled to unit variance


def _get_normal_distribution(random_state: Optional[Union[int, Generator, RandomState]]):
    if isinstance(random_state, (integer, int)) or random_state is None:
        return default_rng(random_state).normal
    if isinstance(random_state, (Generator, RandomState)):
        return random_state.normal
    raise ValueError("random_state must be one of integer, Generator, RandomState, or None.")
```

And the full optimizer, filling the `plan()` slot — I'll write it the way it actually runs as an MPC controller, keeping mean/std/elites on the object between calls so the cross-step shift works, resetting at the start of a rollout:

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
            # The colored-noise package colors the last axis, so draw [N, d, h] and transpose.
            samples = colorednoise.powerlaw_psd_gaussian(
                self.noise_beta,
                size=(num_traj, self.mean.shape[1], self.mean.shape[0]),
            ).transpose([0, 2, 1])
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
            action_sequences[0] = self.mean
        return action_sequences

    def elites_2_action_sequences(self, *, elites, obs, fraction_to_be_used=1.0):
        actions = elites.as_array("actions")
        reused_actions = actions[:, 1:]                    # drop the already-executed action
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

        self.mean[:-1] = self.mean[1:]                    # shift initialization for next step
        last_ob = simulated_paths[best_traj_idx]["observations"][-1]
        self.mean[-1] = self.compute_new_mean(obs=last_ob)
        self.std = self.get_init_std(relative=True)
        return executed_action

    def compute_new_mean(self, obs):
        return self.mean[-1]

    def update_distributions(self, sampled_trajectories, costs):
        elite_idxs = np.array(costs).argsort()[: self.num_elites]
        self.elite_samples = RolloutBuffer(rollouts=sampled_trajectories[elite_idxs])
        elite_sequences = self.elite_samples.as_array("actions")
        new_mean = elite_sequences.mean(axis=0)
        new_std = elite_sequences.std(axis=0)
        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std
```

Let me walk the causal chain to make sure it closes. The pain was that CEM-style sampling MPC needs thousands of rollouts per step, which kills real-time control, and I have to get to tens-to-hundreds. I traced the waste to two leaks. The sample-shape leak: white-noise action perturbations integrate to Brownian state trajectories that don't go anywhere, so on sparse reward most rollouts learn nothing — fixed by sampling temporally correlated colored noise with PSD `1/f^β`, which I showed reddens the state spectrum to `1/f^{β+2}` and gives far-ranging excursions at the same action energy, with `β` an interpretable per-task knob (small for high-frequency control, large for smooth control), and which I can draw efficiently in the rFFT domain with the DC/Nyquist realness-and-variance bookkeeping done right. The amnesia leak: every step discards all evaluated rollouts — fixed by keeping a fraction of the elite set across inner iterations and shifting a fraction across environment steps, so known-good sequences survive instead of being re-discovered, using only a fraction each time so concentrated elites don't collapse the variance. Then sample decay turns the budget into more refinement iterations by exploiting CEM's own narrowing; best-action execution stops wasting the optimization on a never-evaluated mean while add-mean (last iteration only) keeps the clean mean available without over-narrowing; clipping instead of truncation lets the planner actually propose boundary actions; and momentum reduces the noise in fitting a high-dimensional distribution from a small elite set. The result is CEM with every wasted rollout reclaimed and a noise source matched to how trajectories actually explore — same robustness, a fraction of the samples, fast enough to plan in real time.
