CEM's numbers confirmed that the iteration-and-refit fix worked: the wandering collapsed — random's `mean_steps_to_success` of $64/82/112$ fell to $24/23/46$ — and the success rate rose at every horizon ($0.55\to0.70$ at 30, $0.70\to0.85$ at 60, $0.70\to0.95$ at 90) with `mean_dist` dropping to $9.5/6.5/3.4$. But the *shape* across horizons exposes where CEM still leaves episodes on the table. It is excellent at the long horizon (90: $0.95$, residual $3.4$) and weakest at the *short* one (30: $0.70$, residual $9.5$, the highest of the three). That is the tell: a short horizon gives the re-planning loop fewer corrections, so each individual `plan()` call has to find a good route *the first time* — and CEM, drawing white Gaussian perturbations from a zero mean every call, spends most of its $20$ iterations just discovering a route at all, then has no budget left to refine it. Two leaks share that root. Every call starts cold and rediscovers the route from scratch even though consecutive control steps face almost the same problem — the same amnesia random had, one level up: random discarded scores *within* a call, CEM discards the whole solution *across* calls. And the white-noise sampling integrates to a Brownian walk whose net displacement grows like only the square root of the horizon, so most sampled perturbations jitter near the mean rather than committing to a far-ranging excursion toward the door — papered over at long horizons by the re-planning loop, fatal at the short one. The budget is the enemy ($200$ samples, $20$ iters, each a full rollout), so every fix must *reclaim* wasted rollouts, not ask for more.

I propose iCEM (Pinneri et al. 2020) — CEM with a better noise source and the discarded rollouts reclaimed. Start with the sample shape, the bigger leak at the short horizon. To first order, position is the running integral of the action, so a white action sequence drives position along a Brownian walk that does not go anywhere. The thing I actually want to control is not the per-step variance (CEM already adapts that through the elite std) but the *temporal correlation* of the sampled sequence, which white noise lacks entirely. Read the sequence as a time series and look at its power spectral density: white noise has a flat PSD — equal energy at all frequencies, including the highest, which is the action flipping sign every timestep, pure diffusive jitter. To make a sequence persist in a direction I want to *suppress* the high frequencies relative to the low ones, and the clean one-knob family that does this is a power-law spectrum, $\text{PSD}(f) \propto 1/f^{\beta}$, where $\beta = 0$ recovers white noise and larger $\beta$ tilts energy toward low frequencies — smoother, more persistent sequences. The knob propagates correctly to *position*: carrying a $1/f^{\beta}$ action spectrum through the integrator reddens it by two more powers of $f$, so position has spectrum $1/f^{\beta+2}$. Even white actions give the $1/f^2$ Brownian walk (the sanity check), and cranking $\beta$ up piles more energy into the low frequencies — larger, smoother, longer-range excursions at the *same* action variance, because $\beta$ only redistributes energy across frequencies, never adds any. That is the trade I want: same energy, more reach toward the door. I set $\beta = 2.5$, firmly in the smooth regime, since this task wants committed low-frequency motion (cross the room, pass the door), not high-frequency control.

Drawing a length-$T$ sequence with a $1/f^{\beta}$ spectrum is cleanest in the frequency domain, because the PSD *is* a frequency-domain statement: draw random complex Fourier coefficients, scale bin $k$ by $f_k^{-\beta/2}$, inverse-FFT. The zero-frequency bin blows up under $f^{-\beta/2}$, so I clamp the lowest frequency to the smallest resolvable nonzero value $1/T$ to keep DC finite. There is a realness subtlety that controls the *variance*, so I cannot wave it away: for the inverse real-FFT to produce a real sequence the DC coefficient must be real (and the Nyquist coefficient too, when $T$ is even). Zeroing those imaginary parts deletes half the variance at those bins, so I multiply the surviving real part there by $\sqrt{2}$ to restore the full intended power — without it the DC drift comes out at $\sim 0.71\times$ of the design, which is precisely the long-range directional component I am introducing colored noise *for*. The colored sampler is a drop-in replacement for CEM's white `torch.randn`: same $[T,N,A]$ shape, scaled by the same elite std, added to the same mean — so the entire rest of the CEM machinery survives unchanged, and the only thing different per sample is that it now ranges.

Now the amnesia, at two timescales. *Within* a call, across inner iterations: each iteration produces an elite set — the lowest-cost sequences so far — and standard CEM uses them only to refit $\mu,\sigma$ and then discards the actual sequences, drawing a fresh population. But those elites were *good*; after the small refit most of the new population will be worse than the elites I just deleted. So I keep a fraction of them and add them back into the next iteration's pool alongside the fresh colored draws — but only a fraction ($0.3$), because the elites have small spread by construction, and if they dominate the pool the refit collapses $\sigma$ immediately and kills exploration before it starts. *Across* env steps: consecutive steps overlap by all but one timestep, so I shift the optimized mean forward by one, repeat the last entry to fill the freed slot, and reset $\sigma$ to its initial wide value (the new final timestep is genuinely unexplored, so it must re-open exploration). I also shift the kept elites — drop their executed first timestep and append a freshly sampled final action — giving the new step a handful of genuinely-good warm-start sequences, not just a shifted mean. This is the persistent state CEM lacked: I carry `_mean`, `_std`, `_kept_elites` on the planner instance between calls and reset them only when `t0=True` flags an episode start (or when the horizon changes). The `t0` flag that random and CEM both ignored is now load-bearing.

A few smaller reclaim-the-budget moves, each closing a specific leak. CEM's $\sigma$ shrinks automatically as iterations concentrate, so a late iteration samples from an already-narrow distribution where each extra sample buys little — so I decay the population geometrically, $N_i = \max(2\cdot\text{elites},\, N/\gamma)$ with $\gamma = 1.25$, floored at twice the elite count so the elite selection and refit stay meaningful. The payoff is direct: cheaper late iterations mean the *same* total rollout budget buys more refinement than fixed-$N$ CEM could afford — exactly what I need against the short-horizon failure where CEM ran out of optimization. I smooth the refit with momentum, $\mu \leftarrow (1-\alpha)\,\mu_{\text{elite}} + \alpha\,\mu_{\text{prev}}$ and likewise for $\sigma$, with $\alpha = 0.1$, because fitting a high-dimensional mean and std from a small elite set is noisy and the momentum is free variance reduction. And I keep a *best-so-far* across all iterations: CEM returns the final distribution mean, but that mean was never actually evaluated — it is the centroid of the last elites, and in high dimensions a centroid can sit in a low-density region that rolls out worse than the best sequence I actually scored. So I track the single lowest-cost evaluated sequence across every iteration, then roll the final mean through the model once and return whichever is actually better. That stops the optimizer from discarding a known-good plan in favor of an untested centroid — most valuable precisely at the short horizon where a single good route matters more than a well-fit distribution.

One honest note on scope. The full method this descends from is designed for a per-step MPC loop where the across-step shift is the headline mechanism, leaning on a per-task $\beta$ tuned for the control bandwidth and on executing the first action of the *best evaluated* sequence each step. Here the loop already executes only the first action and re-plans, the horizons are fixed by the benchmark rather than tuned, and I fold the best-evaluated idea into the best-so-far + final-mean fallback rather than a per-iteration mean injection. So what I land is the core — colored noise, elite reuse within and across steps, population decay, momentum, best-so-far — fit to this task's single-plan-per-observation interface, not the full MPC harness. I also bump the budget to match the regime this method was characterized in: $\text{num\_samples} = 900$, $\text{n\_iters} = 10$, $K = 10\%$, which after the $\gamma = 1.25$ decay is a comparable total rollout count to CEM's $200\times20$ but spent with the colored, reuse-and-decay structure instead of flat white re-sampling. The cleanest prediction is at horizon 30: CEM's $0.70$ with a $9.5$ residual should rise as committed colored excursions reach the door inside one call and the across-step shift means each call no longer starts cold, with the short-horizon `mean_dist` dropping the most. At horizon 90, where CEM was already $0.95$, I expect to roughly *hold* rather than leap — there is little headroom, and the named risk is over-commitment: colored noise plus aggressive elite reuse can leave even less spread than CEM to escape a wrong route, so a long-horizon seed could regress. If that shows up, the diagnosis is already written: the hard top-k cut is too greedy, and the next move is a *soft*, cost-weighted update.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 3: iCEM
class CustomPlanner(Planner):
    """iCEM (Pinneri et al., CoRL 2020 / PMLR 2021) - colored noise + elite reuse + momentum."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=900, n_iters=10, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        # The task harness passes CEM/MPPI defaults (200, 20). Override them
        # here to match Pinneri et al. Table S4's 4000-budget iCEM setting:
        # 900 samples, 10 iterations, K=10%, gamma=1.25 -> ~4101 fresh
        # trajectories after decay.
        self.num_samples = 900
        self.n_iters = 10
        self.elites_size = max(10, self.num_samples // 10)
        # iCEM paper / reference-code defaults
        self.fraction_elites_reused = 0.3
        self.factor_decrease_num = 1.25
        self.alpha = 0.1              # momentum smoothing factor
        self.noise_beta = 2.5         # learned-model PlaNet setting
        self.var_scale = 1.5
        self.max_norm = 2.45
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Persistent state for across-env-step shift (paper Sec 3.2 / Fig S9).
        # The Planner instance lives across all plan() calls within
        # an evaluation; t0=True at the start of each episode triggers reset.
        self._mean = None
        self._std = None
        self._kept_elites = None

    def _colored_noise(self, T, N, A):
        # rFFT-based 1/f^noise_beta Gaussian, matching
        # colorednoise.powerlaw_psd_gaussian (used by the iCEM repo).
        if T <= 1:
            return torch.randn(T, N, A, device=self.device)
        freqs = torch.fft.rfftfreq(T, device=self.device)
        # Avoid the DC zero-frequency blow-up the same way colorednoise does:
        freqs = freqs.clone()
        freqs[0] = 1.0 / T
        s_scale = freqs ** (-self.noise_beta / 2.0)
        w = s_scale[1:].clone()
        if T % 2 == 0:
            w[-1] = w[-1] / 2
        sigma = 2.0 * torch.sqrt((w ** 2).sum()) / T
        # Random amplitudes in frequency domain, one spectrum per (N, A) trajectory.
        nf = s_scale.shape[0]
        sr = torch.randn(N, A, nf, device=self.device) * s_scale
        si = torch.randn(N, A, nf, device=self.device) * s_scale
        si[..., 0] = 0.0
        # The colorednoise.powerlaw_psd_gaussian reference compensates for
        # the dropped imaginary parts at DC (and Nyquist when T is even) by
        # scaling the corresponding real parts by sqrt(2). Without this fix
        # the DC drift component is ~0.71x of the reference, weakening the
        # long-trajectory directional bias colored noise is meant to provide.
        sr[..., 0] = sr[..., 0] * (2 ** 0.5)
        if T % 2 == 0:
            si[..., -1] = 0.0
            sr[..., -1] = sr[..., -1] * (2 ** 0.5)
        spec = torch.complex(sr, si)
        noise = torch.fft.irfft(spec, n=T, dim=-1) / sigma
        # Return as [T, N, A] to match the CEM baseline's sampling convention.
        return noise.permute(2, 0, 1)

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # iCEM "shift" mechanism (paper Sec 3.2 / icem.py:165-175). At episode
        # start (t0=True) or when plan_length changes, reset to fresh state.
        # Otherwise shift mean/kept-elites left by 1 timestep. Per Alg. 1 /
        # Suppl. E.1, repeat the last mean timestep and reset std everywhere
        # to sigma_init.
        need_reset = (t0 or self._mean is None
                      or self._mean.shape[0] != plan_length)
        if need_reset:
            mean = torch.zeros(plan_length, self.action_dim, device=self.device)
            std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
            prev_elites = None
        else:
            mean = torch.empty_like(self._mean)
            mean[:-1] = self._mean[1:]
            mean[-1] = self._mean[-1]
            std = self.var_scale * torch.ones_like(self._std)
            # Shift kept elites: drop the executed timestep and append a
            # freshly sampled final timestep from the reset distribution.
            if self._kept_elites is not None:
                ke = torch.zeros_like(self._kept_elites)
                ke[:-1] = self._kept_elites[1:]
                K = self._kept_elites.shape[1]
                last_noise = self._colored_noise(1, K, self.action_dim)[0]
                ke[-1] = mean[-1] + std[-1] * last_noise
                prev_elites = ke
            else:
                prev_elites = None

        best_actions = None
        best_cost = torch.tensor(float("inf"), device=self.device)
        losses, elite_means, elite_stds = [], [], []

        num_sim_traj = self.num_samples
        prev_iter_elites = None
        prev_iter_cost = None

        for i in range(self.n_iters):
            # Sample decay per iCEM reference:
            # num_sim_traj = max(elites_size * 2, num_sim_traj / factor_decrease_num)
            if i > 0:
                num_sim_traj = max(self.elites_size * 2,
                                   int(num_sim_traj / self.factor_decrease_num))

            n_reused = int(self.elites_size * self.fraction_elites_reused)

            noise = self._colored_noise(plan_length, num_sim_traj, self.action_dim)
            actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            reused_cost = None
            if i == 0 and prev_elites is not None and n_reused > 0:
                actions = torch.cat([actions, prev_elites[:, :n_reused]], dim=1)
            elif i > 0 and prev_iter_elites is not None and n_reused > 0:
                actions = torch.cat([actions, prev_iter_elites[:, :n_reused]], dim=1)
                reused_cost = prev_iter_cost[:n_reused]

            # Clip action norms (consistent with CEM baseline and planning_cem.yaml)
            norms = actions.norm(dim=-1, keepdim=True)
            coeff = (self.max_norm / norms.clamp(min=1e-6)).clamp(max=1.0)
            actions = actions * coeff

            if reused_cost is None:
                cost = self.cost_function(
                    rearrange(actions, "t b a -> b a t"), obs_init
                )
            else:
                fresh_actions = actions[:, :num_sim_traj]
                fresh_cost = self.cost_function(
                    rearrange(fresh_actions, "t b a -> b a t"), obs_init
                )
                cost = torch.cat([fresh_cost, reused_cost], dim=0)
            losses.append(cost.min().item())

            # Best-so-far across all iterations (iCEM "memory" within a plan).
            min_idx = cost.argmin()
            if cost[min_idx] < best_cost:
                best_cost = cost[min_idx]
                best_actions = actions[:, min_idx].clone()

            elite_idxs = torch.topk(-cost, self.elites_size, dim=0).indices
            elite_actions = actions[:, elite_idxs]
            elite_cost = cost[elite_idxs]
            elite_means.append(elite_cost.mean().item())
            elite_stds.append(elite_cost.std().item())

            # Momentum update of mean/std (alpha=0.1 per iCEM reference).
            new_mean = elite_actions.mean(dim=1)
            new_std = elite_actions.std(dim=1)
            mean = (1.0 - self.alpha) * new_mean + self.alpha * mean
            std = (1.0 - self.alpha) * new_std + self.alpha * std

            prev_iter_elites = elite_actions.detach()
            prev_iter_cost = elite_cost.detach()

        # Return best-ever trajectory if it beats the final mean (iCEM keeps best).
        final_mean_cost = self.cost_function(
            rearrange(mean.unsqueeze(1), "t b a -> b a t"), obs_init
        )[0]
        out = best_actions if best_cost < final_mean_cost else mean

        # Save state for the next env step's shift (paper Sec 3.2).
        # prev_iter_elites is the last iteration's elite_actions tensor [T, K, A].
        self._mean = mean.detach()
        self._std = std.detach()
        self._kept_elites = prev_iter_elites.detach() if prev_iter_elites is not None else None

        return PlanningResult(
            actions=out,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
