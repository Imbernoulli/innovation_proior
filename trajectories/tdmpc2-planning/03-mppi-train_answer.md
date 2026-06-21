Bare CEM did most of what I predicted, and the half it left undone points straight at the next move. Cheetah-run rose from iCEM's $796$ to $833$ mean — removing the colored-noise prior helped, exactly as the spectral argument said. But the per-seed numbers were $\{870.86, 907.20, 722.37\}$: the good seeds climbed to $870$ and $907$, well above anything iCEM reached, yet seed $456$ *collapsed* to $722$ — a wider spread, not the tighter one I had predicted. Cartpole-swingup actually slipped, $881$ down to $867$, dragged by seed $123$ at $842$; walker held at ceiling. The pattern across both non-saturated tasks is the same: CEM's *mean* is decent but its *variance across seeds* is bad. It is fragile, and the fragility lives in the refit, not the noise. CEM's refit is a *hard* top-k average — it weights the single best elite, the one the value head says is far better than the rest, exactly as much as the marginal $64$th barely above the cutoff, and discards everything below the cutoff as if it carried no information. On a seed whose elite set happens to be contaminated by a few mediocre-but-lucky sequences near the threshold, the equal-weight mean is dragged toward them, the next population is centered slightly wrong, and the error compounds over six iterations. That is the $722$ cheetah seed and the $842$ cartpole seed — not the average behavior, but the failure mode of a refit that treats a barely-elite sequence the same as the best one.

I propose **MPPI** — the model-predictive path-integral refit — which replaces the hard equal-weight average with a *soft, value-weighted* Gibbs refit. I do not want a heuristic re-weighting; I want the principled one, so go back to the optimal sampling target. The value-maximizing distribution over action sequences is not the hard-restricted indicator density CEM fits — that was already an approximation. The honest optimal target is the *Gibbs* distribution

$$q^*(\tau) \;\propto\; \exp\!\big(V(\tau)/\lambda\big)\, p(\tau),$$

the base distribution tilted by the exponentiated return. This falls out of the free-energy view of optimal control: define the free energy $F = \log \mathbb{E}_p[\exp(V/\lambda)]$, and by Jensen's inequality on the concave $\log$, $-\lambda F$ lower-bounds the control objective $\mathbb{E}_q[-V] + \lambda\, D_{\mathrm{KL}}(q \,\|\, p)$ — expected negative return plus a KL penalty pulling the sampler toward the base — with equality exactly when $q$ is $q^*$. So the variance-optimal target is not "keep the top-k equally"; it is "weight each trajectory by $\exp(V/\lambda)$." CEM's hard top-k is the crude approximation that swaps the smooth exponential weight for a $0/1$ indicator at a threshold, and the fragility I measured is the price of that crudeness.

I cannot sample $q^*$ directly — it has an unknown normalizer — but I can importance-sample it from my current Gaussian: the refit mean becomes the $\exp(V/\lambda)$-weighted average of the sampled action sequences, and the refit std their $\exp(V/\lambda)$-weighted spread. This is the path-integral update, and it is precisely the soft version of CEM's hard average. The single best elite, with the largest $V$, gets the largest weight and pulls the mean hardest; a marginal elite barely above the old cutoff gets a tiny weight and barely moves it. A few lucky near-threshold sequences can no longer hijack the fit, because the exponential has already down-weighted them to nearly nothing — the graded information CEM discarded is exactly what stabilizes the refit, and the std update inherits the same robustness since its weighted spread is dominated by the genuinely-good sequences.

There is a numerical landmine in the exponential, and handling it is load-bearing. Raw $\exp(V/\lambda)$ over predicted returns either overflows or underflows depending on the return scale, which varies across the three tasks. The fix is to shift every value by the *maximum* elite value before exponentiating,

$$\text{score} = \exp\!\big((V - V_{\max})/\lambda\big),$$

which multiplies numerator and denominator by the same constant $\exp(-V_{\max}/\lambda)$ — so it changes nothing about the weighted average — but guarantees the best elite has weight $\exp(0) = 1$ and every other weight lies in $(0, 1]$. The weight I actually compute is the exponentiated *gap from the best elite*: self-normalizing, scale-robust, behaving identically across walker, cheetah and cartpole despite their different return magnitudes. That is the same task-robustness argument that made me drop the per-task $\beta$ — one temperature, no per-task tuning. The temperature $\lambda$, stored as its inverse $\texttt{temperature} = 1/\lambda$, is the one knob, and it interpolates between the two rungs below: as $\lambda \to \infty$ (temperature $\to 0$) every elite gets equal weight and the soft refit collapses back to CEM's hard average, so CEM is literally the high-temperature limit of this method — I am generalizing it, not replacing it; as $\lambda \to 0$ all the weight piles on the single best elite, greedy and high-variance. The substrate sets $\texttt{temperature} = 0.5$, moderate: soft enough to use the value magnitudes and stabilize against near-threshold contamination, sharp enough to actually pull toward the good elites. I keep the top-k truncation *before* the soft weighting — not because the exponential needs it, but because with a finite population the far-out low-value samples still carry tiny nonzero weight that contaminates the outlier-sensitive variance fit; restricting to the top $64$ cleans the fit while the soft weighting preserves the graded quality among the survivors. The refit is therefore hard-truncate-then-soft-weight: CEM's elite selection kept, CEM's equal-weight average replaced by the Gibbs weight.

One more change follows the same logic into the final action. CEM committed the first entry of the *mean* — but the mean is the centroid of the elites, a sequence that may correspond to no actually-evaluated rollout, and in a high-dimensional action space the centroid of a cloud can sit in a low-value region *between* the good samples. I would rather commit an action from a sequence I actually rolled out and know is good, so instead of the mean I draw one elite according to its Gibbs weight — Gumbel-softmax sampling over the elite scores via the substrate's `gumbel_softmax_sample` — and execute the first action of that drawn elite. The genuinely-best elites are most likely drawn, but the stochasticity keeps exploration alive during data collection, and it removes another way a bad centroid could sink a seed.

Everything up to the refit is identical to CEM: encode, warm-start `num_pi_trajs = 24` policy trajectories into the first slots, fill the rest with white Gaussian samples $\texttt{mean} + \texttt{std}\cdot\texttt{randn}$ clamped to $[-1,1]$, score with `agent._estimate_value`, select the `num_elites = 64` top-value elites. The change is three lines of refit — compute `max_value` over the elites, $\texttt{score} = \exp(\texttt{temperature}\cdot(\texttt{elite\_value} - \texttt{max\_value}))$ normalized to sum to one, then $\texttt{mean} = \sum \texttt{score}\cdot\texttt{elite\_actions}$ and $\texttt{std} = \sqrt{\sum \texttt{score}\cdot(\texttt{elite\_actions} - \texttt{mean})^2}$ clamped to $[\texttt{min\_std}, \texttt{max\_std}]$ — plus drawing the final action via Gumbel-softmax over the elite scores. Having climbed from a trimmed iCEM through bare CEM, the value-weighting argument lands me precisely on the TD-MPC2 default planner — the optimizer the world model was designed to plan with, which is the right place to arrive. Against CEM's measured numbers I expect cheetah's *ceiling* (the $870$–$907$ good seeds) to barely move — the soft refit cannot find plans the model cannot represent — but its *floor* to recover: the $722$ collapse seed should come back as the soft weighting stops a contaminated elite set from dragging the mean, lifting the cheetah mean toward $\sim 890$ with a tighter spread, the tightening I wrongly predicted for CEM finally showing up here where the mechanism matches the claim. Cartpole should recover the point it lost back above $875$, walker stays at ceiling. The bar to clear is CEM's: beat $833$ cheetah and $867$ cartpole, and do it by *closing the seed gap*.

```python
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    """MPPI baseline -- Model Predictive Path Integral control."""
    cfg = agent.cfg

    # Sample policy trajectories as warm-starts
    z = agent.model.encode(obs, task)
    if cfg.num_pi_trajs > 0:
        pi_actions = torch.empty(
            cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
            device=agent.device,
        )
        _z = z.repeat(cfg.num_pi_trajs, 1)
        for t in range(cfg.horizon - 1):
            pi_actions[t], _ = agent.model.pi(_z, task)
            _z = agent.model.next(_z, pi_actions[t], task)
        pi_actions[-1], _ = agent.model.pi(_z, task)

    # Initialize sampling distribution
    z = z.repeat(cfg.num_samples, 1)
    mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    std = torch.full(
        (cfg.horizon, cfg.action_dim), cfg.max_std,
        dtype=torch.float, device=agent.device,
    )
    if not t0:
        mean[:-1] = agent._prev_mean[1:]
    actions = torch.empty(
        cfg.horizon, cfg.num_samples, cfg.action_dim,
        device=agent.device,
    )
    if cfg.num_pi_trajs > 0:
        actions[:, :cfg.num_pi_trajs] = pi_actions

    # Iterate MPPI
    for _ in range(cfg.iterations):
        # Sample actions from Gaussian
        r = torch.randn(
            cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
            cfg.action_dim, device=agent.device,
        )
        actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
        actions_sample = actions_sample.clamp(-1, 1)
        actions[:, cfg.num_pi_trajs:] = actions_sample
        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        # Evaluate trajectories and select elites
        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(
            value.squeeze(1), cfg.num_elites, dim=0,
        ).indices
        elite_value = value[elite_idxs]
        elite_actions = actions[:, elite_idxs]

        # Update sampling distribution (softmax-weighted)
        max_value = elite_value.max(0).values
        score = torch.exp(cfg.temperature * (elite_value - max_value))
        score = score / score.sum(0)
        mean = (
            (score.unsqueeze(0) * elite_actions).sum(dim=1)
            / (score.sum(0) + 1e-9)
        )
        std = (
            (
                score.unsqueeze(0)
                * (elite_actions - mean.unsqueeze(1)) ** 2
            ).sum(dim=1)
            / (score.sum(0) + 1e-9)
        ).sqrt()
        std = std.clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

    # Select action from elites via Gumbel sampling
    rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
    actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
    a, std_final = actions[0], std[0]
    if not eval_mode:
        a = a + std_final * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
```
