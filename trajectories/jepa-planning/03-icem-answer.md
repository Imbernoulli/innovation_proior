**Problem.** CEM fixed random's wandering (steps-to-success 64/82/112 → 24/23/46) but is weakest at
the *short* horizon (0.70, residual 9.5 at horizon 30): white-noise perturbations integrate to
Brownian walks that don't reach the door inside one cold-started `plan()` call, and CEM throws away the
whole optimized distribution between control steps. Both leaks waste rollouts the tight budget can't
spare.

**Key idea (iCEM — CEM with reclaimed rollouts and a better noise source).** Replace white sampling
with temporally-correlated *colored* noise (PSD ∝ 1/f^β, β = 2.5), drawn in the rFFT domain with the
DC/Nyquist realness-and-√2-variance bookkeeping done right, so each sampled sequence ranges far per
unit of action energy. Reclaim discarded rollouts: keep a fraction (0.3) of the elites across inner
iterations and *shift* them (plus the mean) across env steps, warm-starting each call from the previous
solution. Decay the population geometrically (γ = 1.25, floored at 2·elites) to buy more refinement
iterations from a fixed budget, smooth the refit with momentum (α = 0.1), and track the *best-so-far*
evaluated sequence, returning it instead of the untested final mean when it scores better.

**Why.** Colored noise reddens the position spectrum to 1/f^(β+2): committed low-frequency excursions
reach and pass the door where white jitter could not — aimed at the short-horizon failure. The
across-step shift uses the `t0` flag (which random and CEM ignored) to stop re-discovering the route
every call. Best-so-far guards against CEM's centroid-in-a-low-density-region mean. This is the core of
the method fit to the task's single-plan interface — not the full per-step MPC harness — folding
"execute the best evaluated sequence" into the best-so-far + final-mean fallback.

**Hyperparameters.** Budget override `num_samples = 900`, `n_iters = 10`, `elites_size =
num_samples//10 = 90`; `noise_beta = 2.5`; `fraction_elites_reused = 0.3`; `factor_decrease_num =
1.25`; `alpha = 0.1`; `var_scale = 1.5` (reset std); `max_norm = 2.45`. Persists `_mean/_std/
_kept_elites` across calls, resets on `t0`. `no_grad`.

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
