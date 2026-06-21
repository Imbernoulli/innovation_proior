iCEM's numbers confirmed the bet *and* exposed the risk I flagged when I built it. The bet paid off at the short horizon exactly as predicted: CEM's weakest benchmark, horizon 30, went from $0.70$ to $0.80$ and the residual `mean_dist` dropped from $9.5$ to $5.7$ — the colored, far-ranging excursions plus the across-step warm start did reach the door inside a single `plan()` call. Horizon 60 held respectably ($0.90$, residual $4.1$). But horizon 90 scored $0.85$ — *below* CEM's $0.95$ there — with residual $4.8$ against CEM's $3.4$. That is the regression I named in advance: at the long horizon, where CEM's white sampling and the re-planning loop had room to converge on a clean route, iCEM's aggressive elite reuse and smooth low-frequency draws *over-commit*. When the kept elites and the persistent shifted mean all agree on a route and that route is the wrong side of the door, the search has even less spread to escape than CEM did — the hard top-k cut plus the colored over-smoothing leaves no candidate that disagrees enough to pull the distribution onto the other route. Averaged across the three benchmarks iCEM is roughly a wash with CEM (mean success $\sim0.85$ vs $\sim0.83$), which is not the clean dominance I wanted.

The root is the *hard* elite selection. Both CEM and iCEM refit by taking the top-k lowest-cost sequences and computing their plain mean and variance, which throws away two things. First, *how much* better the best elite was than the worst: a dramatically-better sequence and one that barely cleared the cut count the same, each getting weight $1/k$. Second, the hard cut is brittle near the decision boundary between the two routes — a handful of mediocre elites on the wrong route can drag the mean and keep the variance from shrinking onto the right one, or, in iCEM's over-committed case, the elites all land on one route and the variance collapses there with no soft pressure from the near-elite samples on the other. I want an update that is *soft*: every sampled sequence contributes, weighted by how good it is, so a markedly-better rollout pulls the mean and tightens the variance far more than a marginal one, and no information is discarded at a hard threshold — and I want it to be the *principled* soft update, the same demand I made moving from random shooting to CEM.

I propose MPPI (Williams et al. 2015) — the exponential cost-weighted update. Go back to the object and ask what the right weighting is. Frame it as the free-energy / partition-function object: assign each trajectory $\tau$ a Boltzmann weight exponential in its negative cost, $w(\tau) \propto \exp(-S(\tau)/\lambda)$, where $S(\tau)$ is the cost of rolling that sequence through the model and $\lambda$ is a temperature. Low-cost trajectories get large weight, high-cost ones are suppressed, and $\lambda$ controls how sharply: $\lambda \to 0$ puts all the weight on the single best sequence (a hard argmax), $\lambda \to \infty$ weights everything equally (plain averaging). The optimal control is the weighted average of the sampled controls,

$$u_i \;\leftarrow\; \sum_k w_k\, a_{i,k}, \qquad w_k = \frac{\exp(-S(\tau_k)/\lambda)}{\sum_j \exp(-S(\tau_j)/\lambda)},$$

the soft, information-preserving analogue of CEM's hard elite mean. This is the path-integral reading of optimal control: each sequence votes with a weight that decays smoothly in its cost, so the dramatically-better wrong-route-avoiding sequence dominates without a hard line, and a cluster of mediocre wrong-route sequences gets exponentially suppressed rather than averaged in at full weight — exactly the long-horizon failure iCEM showed.

I have to be honest about the form the task's harness gives me. The full path-integral derivation assumes a control-affine system with a particular noise-cost coupling and yields a Girsanov likelihood-ratio correction that scales both the mean and the covariance of the sampling distribution. I do not have a control-affine analytic model here — only a learned JEPA world model I can roll forward and a black-box cost — so I do not import the covariance-scaling Girsanov machinery. What I land instead is the discrete, sampling-based core as it is actually used on top of learned models: keep the CEM skeleton — sample a batch around the current mean, roll through the model, take the lowest-cost elites — but replace the *update* with the exponential cost-weighting. After scoring I still select the top elites ($\text{num\_elites} = 20$, top 10%), then compute weights over them, $\text{score}_k = \exp(\text{temperature}\cdot(\text{min\_cost} - \text{cost}_k))$, normalized to sum to $1$, and set the new mean to the weighted average of the elite sequences and the new std to the weighted standard deviation. Subtracting $\text{min\_cost}$ inside the exponential is the standard numerical stabilization — it shifts the largest exponent to zero so nothing overflows, and because the weights are normalized the shift cancels and leaves the Boltzmann weighting unchanged. The temperature is small, $0.005$, which keeps the weighting gentle relative to the cost scale — sharp enough that a clearly better rollout dominates, soft enough that the near-best sequences still contribute and the variance does not collapse onto one route the way iCEM's hard reuse did.

I keep the sampling *white* here — plain `torch.randn` perturbations, not colored noise — a deliberate choice given what iCEM's numbers showed. The colored over-smoothing was part of what made iCEM over-commit at the long horizon, and the soft weighting is supposed to do the reach-and-route work through the *update*, not through a far-ranging noise prior. White sampling with a soft cost-weighted mean lets the distribution be pulled toward whichever route the rollouts actually score well, without a low-frequency prior baking in a committed direction before the cost is read. I also widen the initial spread to $\text{max\_std} = 2.0$ (above CEM's $1.5$), because the soft update is more forgiving of a wide start — it will not be dragged around by a few wide outliers the way a hard elite mean can be, since those outliers carry exponentially small weight — so I can afford to explore both routes more aggressively at the first iteration and let the Boltzmann weighting concentrate from there.

One more place the soft framing changes the *output*, and it matters for the maze. CEM and iCEM return the distribution's center, but the mean of a soft-weighted distribution, in a bimodal cost surface with two routes, can land *between* the two modes — in the wall — which is worse than either route. So instead of executing the mean I select the executed sequence by *sampling* from the final elite weights: draw one elite with probability proportional to its score and return that actual evaluated sequence. This is the path-integral controller's stochastic action selection, and it is the right thing in a multi-modal surface — it commits to one real, rolled-out route rather than averaging two incompatible routes into an infeasible compromise, and the chosen sequence is one I actually scored, so it is feasible and known-good, not an untested centroid. Nothing persists across env steps — each call re-optimizes from the zero mean and wide std — so `t0` is irrelevant again, and the whole loop runs under `no_grad` because, like every rung before it, MPPI reads only function values of the cost and never touches the gradient the differentiable model could hand it. That gradient is still sitting unused; it is the lever a method beyond this ladder would pull. The cleanest prediction is at horizon 90, where iCEM slipped to $0.85$ by over-committing: the Boltzmann weighting should recover CEM's long-horizon performance, back up toward $0.95$ with `mean_dist` near $3.4$, because the soft update plus stochastic-elite selection stops a few wrong-route elites from collapsing the search and stops the planner from executing an in-the-wall mean. At horizon 60 I expect a clear lift over iCEM's $0.90$, where both routes are live and a hard cut is most brittle. At horizon 30 the open question is whether dropping the colored noise costs me reach — I expect to roughly hold iCEM's $0.80$ — and if it instead drops below $0.75$, that is the signal the short horizon genuinely needs the colored prior and the right method would be the soft update *over* colored noise. The headline bet is that this soft, information-preserving update gives the first planner strong at *all three* horizons at once — the clean dominance neither hard-cut baseline achieved.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 4: MPPI
class CustomPlanner(Planner):
    """MPPI (Model Predictive Path Integral) planner for JEPA world models."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        # Match upstream MPPIPlanner defaults — planning_mppi.yaml sets
        # var_scale=1.5 but MPPIPlanner doesn't accept that kwarg, so the
        # effective config is max_std=2 (class default), temperature=0.005,
        # num_elites=20 (yaml). Mirror that here.
        self.num_elites = max(20, num_samples // 10)
        self.max_std = 2.0
        self.temperature = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)

        losses = []
        elite_means = []
        elite_stds = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=self.device,
            )

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # MPPI weighted update
            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )
            score /= score.sum(0) + 1e-9
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
            )

        # Select action via weighted sampling
        score_np = score.cpu().numpy()
        selected = elite_actions[
            :, np.random.choice(np.arange(score_np.shape[0]), p=score_np)
        ]

        return PlanningResult(
            actions=selected,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
