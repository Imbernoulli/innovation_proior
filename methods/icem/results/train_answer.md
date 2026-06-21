The problem is making sampling-based model-predictive control affordable. In this setting we have a forward dynamics model that can be rolled out, but we do not rely on gradients through it. At every environment step we must pick an action by optimizing a short horizon action sequence, which we do by sampling candidate sequences, simulating each one through the model, and scoring the resulting trajectories. The standard Cross-Entropy Method works well even on learned models and sparse rewards, but published configurations consume thousands of rollouts per control step, which makes real-time robot planning impossible. The concrete goal is to obtain similar control quality with tens to hundreds of rollouts per step.

The baseline waste comes from two shared habits. First, CEM and MPPI perturb action sequences with temporally uncorrelated white noise. Because actions drive state changes through integration, white-noise actions produce Brownian state trajectories whose net displacement grows only like the square root of the horizon; on sparse-reward tasks most rollouts never reach a state where the cost provides any gradient signal. Second, every replanning step discards almost all of the rollouts it just evaluated, including the best ones, and starts the next step from little more than a shifted mean. The information bought by those expensive simulations is used once and thrown away.

The method I propose is iCEM, the improved Cross-Entropy Method. It keeps CEM's gradient-free sample-evaluate-refit loop, but it replaces the noise source with temporally correlated colored noise and reclaims the rollouts that CEM wastes. The central design parameter is a single scalar beta that controls the power-law spectrum of the sampled perturbations, PSD_a(f) proportional to 1/f^beta. With beta equal to zero we recover white noise; with beta around two to four the perturbations become smooth and low-frequency. Passing through the integrator reddens the state spectrum further to 1/f^{beta+2}, so the same action variance produces trajectories that range farther and faster. This is exactly what sparse-reward exploration needs: instead of diffusing in place, sampled sequences make committed low-frequency excursions that are more likely to reach informative states. Beta is task-interpretable, small for high-frequency control and large for smooth reaching or manipulation.

iCEM also treats elite sequences as a memory rather than a disposable byproduct. A fraction of each inner iteration's elite set is carried into the next iteration, and a fraction of the previous environment step's elite set is shifted forward and reused at the start of the next planning call. Only a minority is reused so that the concentrated elites do not collapse the search variance prematurely. Within a fixed rollout budget, the population size is decayed geometrically across inner iterations, because CEM's own narrowing makes late iterations less informative per sample; the saved budget buys extra refinement iterations. The mean is updated with momentum from the elite fit to reduce noise in the high-dimensional distribution estimate, and actions are clipped to bounds rather than drawn from a truncated normal so boundary actions remain well represented. At the end of planning we execute the best evaluated sequence rather than the untested mean, but we also evaluate the mean at the last iteration so it can win when it is genuinely clean.

```python
import torch
from einops import rearrange


class CustomPlanner(Planner):
    """iCEM planner: colored-noise CEM with elite reuse and population decay."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=900, n_iters=10, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.elites_size = max(10, self.num_samples // 10)
        self.fraction_elites_reused = 0.3
        self.factor_decrease_num = 1.25
        self.alpha = 0.1
        self.noise_beta = 2.5
        self.var_scale = 1.5
        self.max_norm = 2.45
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean = None
        self._std = None
        self._kept_elites = None

    def _colored_noise(self, T, N, A):
        """Draw N independent length-T action sequences with PSD ~ 1/f^noise_beta."""
        if T <= 1:
            return torch.randn(T, N, A, device=self.device)
        freqs = torch.fft.rfftfreq(T, device=self.device)
        freqs[0] = 1.0 / T
        s_scale = freqs ** (-self.noise_beta / 2.0)
        w = s_scale[1:].clone()
        if T % 2 == 0:
            w[-1] = w[-1] / 2.0
        sigma = 2.0 * torch.sqrt((w ** 2).sum()) / T
        nf = s_scale.shape[0]
        sr = torch.randn(N, A, nf, device=self.device) * s_scale
        si = torch.randn(N, A, nf, device=self.device) * s_scale
        si[..., 0] = 0.0
        sr[..., 0] = sr[..., 0] * (2 ** 0.5)
        if T % 2 == 0:
            si[..., -1] = 0.0
            sr[..., -1] = sr[..., -1] * (2 ** 0.5)
        spec = torch.complex(sr, si)
        noise = torch.fft.irfft(spec, n=T, dim=-1) / sigma
        return noise.permute(2, 0, 1)

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

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
            if i > 0:
                num_sim_traj = max(
                    self.elites_size * 2,
                    int(num_sim_traj / self.factor_decrease_num)
                )
            n_reused = int(self.elites_size * self.fraction_elites_reused)

            noise = self._colored_noise(plan_length, num_sim_traj, self.action_dim)
            actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            reused_cost = None
            if i == 0 and prev_elites is not None and n_reused > 0:
                actions = torch.cat([actions, prev_elites[:, :n_reused]], dim=1)
            elif i > 0 and prev_iter_elites is not None and n_reused > 0:
                actions = torch.cat([actions, prev_iter_elites[:, :n_reused]], dim=1)
                reused_cost = prev_iter_cost[:n_reused]

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

            min_idx = cost.argmin()
            if cost[min_idx] < best_cost:
                best_cost = cost[min_idx]
                best_actions = actions[:, min_idx].clone()

            elite_idxs = torch.topk(-cost, self.elites_size, dim=0).indices
            elite_actions = actions[:, elite_idxs]
            elite_cost = cost[elite_idxs]
            elite_means.append(elite_cost.mean().item())
            elite_stds.append(elite_cost.std().item())

            new_mean = elite_actions.mean(dim=1)
            new_std = elite_actions.std(dim=1)
            mean = (1.0 - self.alpha) * new_mean + self.alpha * mean
            std = (1.0 - self.alpha) * new_std + self.alpha * std

            prev_iter_elites = elite_actions.detach()
            prev_iter_cost = elite_cost.detach()

        final_mean_cost = self.cost_function(
            rearrange(mean.unsqueeze(1), "t b a -> b a t"), obs_init
        )[0]
        out = best_actions if best_cost < final_mean_cost else mean

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
