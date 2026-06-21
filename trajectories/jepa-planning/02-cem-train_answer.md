The random-shooting floor told me exactly where it bleeds, and it told me in the step counts. At horizon 30 it cleared only $0.55$ of the episodes and finished a mean distance of $12.6$ from the goal — more than half the time it never got near. At horizons 60 and 90 the success rate crept to $0.70$, but `mean_steps_to_success` ran $64 \to 82 \to 112$, climbing with the horizon: when random shooting succeeds, it succeeds *slowly*, because it wanders. That is the Brownian signature — white actions integrate to a random walk whose net displacement grows only like the square root of its length — and the deeper problem behind it is that the floor learns nothing within a step. It spends the same $200$-sample batch the iterative methods will spend *per iteration*, once, and throws every score away. Every one of those rollouts told me something about where the low-cost sequences live, and the floor discards all of it. That is the leak to plug, and the fix is not a better noise source yet — it is to stop drawing from a frozen distribution and instead let the sampling distribution *move toward* the sequences that scored well, batch after batch.

I want that move to be principled rather than a nudge, because "move toward the good points" can be done a dozen ad-hoc ways and I have no idea which is right or what it converges to. So I propose the Cross-Entropy Method (Rubinstein 1997; de Boer et al. 2005), and I derive it from rare-event sampling so the update is forced rather than chosen. Reframe the problem: instead of "where is the lowest cost?", work with the score $S = -\text{cost}$ and ask, for a high threshold $\gamma$, how probable it is that a randomly drawn sequence clears it — $\ell(\gamma) = P_u(S(X) \ge \gamma)$ under a broad baseline density. Push $\gamma$ toward the optimum and the set $\{x : S(x) \ge \gamma\}$ shrinks toward the maximizers; under a broad density its probability tends to zero — it becomes a *rare* event. A density that made this rare event likely — that put its mass on $\{S(X) \ge \gamma\}$ for $\gamma$ near the optimum — is exactly a density that samples almost only the best action sequences. Finding the optimal plan and making the near-optimum event common are the same problem, so I can borrow the rare-event toolkit.

That toolkit is importance sampling, and it hands me an ideal target. Don't sample the broad $f(\cdot;u)$, sample a cleverer $g$ and undo the change of measure with the likelihood ratio, $\ell = E_g[\,\mathbb{1}\{S(X)\ge\gamma\}\,f(X;u)/g(X)\,]$. There is a *best* $g$ — the zero-variance one. Zero variance means the summand $\mathbb{1}\{S(x)\ge\gamma\}\,f(x;u)/g(x)$ is constant in $x$; setting it equal to its own mean $\ell$ forces

$$g^*(x) = \frac{\mathbb{1}\{S(x)\ge\gamma\}\,f(x;u)}{\ell}.$$

That is just the baseline density *restricted to the elite event* and renormalized — precisely the "throw away everything below the bar, keep the mass on the good region" distribution I was hand-waving about, except now it is the variance-minimizing density, not a heuristic. I cannot sample $g^*$ directly (it carries the unknown $\ell$ and an arbitrary shape), so I find the member of a tractable family $f(\cdot;v)$ closest to it in KL divergence, $D(g,h) = \int g\ln(g/h)$. Minimizing $D(g^*_t, f(\cdot;v))$ over $v$ drops the $v$-free first term and leaves $\max_v\, E_{v_{t-1}}[\,\mathbb{1}\{S(X)\ge\gamma\}\ln f(X;v)\,]$ (the positive constant $1/\ell_t$ does not move the argmax). Replace the expectation by its Monte-Carlo counterpart over the batch I actually drew, and the indicator is $1$ on the threshold-beating samples and $0$ elsewhere — so the objective is the *log-likelihood of the elites under $f(\cdot;v)$*. The principled answer to "how do I move the distribution toward the good sequences" is therefore: take the sequences that beat the bar and fit your sampling distribution to them by maximum likelihood.

A *fixed* high $\gamma$ is a dead end — the very rarity that motivated importance sampling means almost no sequence clears it, every indicator is zero, and the maximum-likelihood program has no data. So I let $\gamma$ track the sample: each iteration, sort the costs and take the top fraction $\rho$ as elites — equivalently the $(1-\rho)$ sample quantile. Now there are always exactly that many elites, never zero. And it is self-tuning: early, when the distribution is broad, the elite cutoff is a modest score; as the distribution concentrates, the same top fraction is a *higher* score, so the bar ratchets upward toward the optimum on its own. The loop becomes: sample from the current distribution, score by rolling through the model, take the lowest-cost elites, refit by maximum likelihood, repeat — the iteration the floor structurally could not do.

The family follows from running the fit *every* iteration: I want a closed form and cheap sampling, which both point at a diagonal Gaussian over every (timestep, coordinate), $x \sim \mathcal{N}(\mu, \sigma^2)$ per entry. Its elite-restricted MLE drops out cleanly. The Gaussian log-density of one elite is $\sum[-\tfrac12\ln(2\pi) - \ln\sigma - (x-\mu)^2/(2\sigma^2)]$; summing over the elites and setting $\partial/\partial\mu = 0$ gives $\sum(x-\mu)/\sigma^2 = 0$, and since $\sigma^2$ is a common positive factor it cancels, leaving $\mu = $ the **elite sample mean**, coordinate by coordinate. Setting $\partial/\partial\sigma = 0$ gives $-N_e/\sigma + (1/\sigma^3)\sum(x-\mu)^2 = 0$, i.e. $\sigma^2 = $ the **elite sample variance**. The whole update is therefore: compute the elites' per-(timestep,coordinate) mean and spread, and those *are* next iteration's parameters. The mean re-centers the search where the good action sequences clustered; the variance re-sizes exploration — automatically tighter on coordinates the elites agreed about, wider where they did not. That second part is the thing the fixed-spread floor never had: the optimizer sets its own exploration width from the data. In code I store the spread as `torch.std` over the elite dimension.

Reading the limit tells me both the right fixed point and the danger. As the distribution concentrates the elites cluster tighter and their variance shrinks toward zero; in the limit $\sigma \to 0$ the Gaussian collapses to a point mass at $\mu$ — a sampler fully concentrated on the best region, the *right* fixed point. But the same collapse can bite: if an early batch's elites land in a small cluster by luck, the variance update slams $\sigma$ small there, and now I am sampling a tiny region around a possibly-wrong plan with no spread to escape. In this maze that is a real risk — there are two routes through the door, and if the first elites all cluster on the wrong-side route the variance can commit before the right route is even sampled. The standard guard is to start the spread deliberately *wide* so the first iterations explore aggressively; I set the initial standard deviation to $\text{var\_scale} = 1.5$ everywhere, larger than unit spread, exactly so the early batches cover both routes before any collapse. (A momentum blend between the fresh fit and the previous parameters is a second guard, but this rung uses the raw mean/std update; momentum is the knob I would turn only if premature collapse proved to be the bottleneck.)

The continuous action space forces the same projection the floor used: draw from the plain Gaussian, then for any action whose 2-D norm exceeds $\text{max\_norm} = 2.45$ rescale it to the boundary, leaving shorter actions untouched — with $n = \|a\|$, multiply by $\text{clamp}(\min(\max(n,0),\,\text{max\_norm}))/(n+\varepsilon)$. Feasibility is enforced before the cost ever sees a candidate, and the elites are fit on the *projected* sequences, so the refit respects the constraint too. The bookkeeping: the constructor receives $\text{num\_samples} = 200$ and $\text{n\_iters} = 20$, and unlike the floor I *use* `n_iters` — $20$ refinement passes of $200$ sequences each. The elite count is $\max(10, \text{num\_samples}//10) = 20$ (the top 10%, floored at $10$ so the mean and variance stay estimable), the initial mean is the zero action sequence (no prior bias, the same neutral start the floor used), and because my objective is a cost to minimize while the derivation maximized $S = -\text{cost}$, the elites are the *lowest*-cost samples, picked as the top-k of $-\text{cost}$. I return the distribution's center: after the final refit, the mean sequence is the Gaussian's point estimate of the optimum, and in the degenerate limit it *is* the optimum. Nothing persists across env steps — each call re-optimizes from the zero-mean, wide-$\sigma$ start — so `t0` is irrelevant, and the whole loop runs under `no_grad` because, like the floor, CEM only ever reads function values and never touches the gradient the differentiable model could hand it. The cleanest prediction against random's numbers is the step count: if refitting toward the elites is the right fix for the wandering, `mean_steps_to_success` must fall sharply from $64/82/112$, because the planner now commits to a direction instead of re-rolling white noise every call, with the success rate rising and `mean_dist` dropping at every horizon.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 2: CEM
class CustomPlanner(Planner):
    """CEM (Cross-Entropy Method) planner for JEPA world models."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.num_elites = max(10, num_samples // 10)
        self.var_scale = 1.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)

        losses = []
        elite_means = []
        elite_stds = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=self.device,
            )

            # Clip action norms
            max_norm = 2.45
            eps = 1e-6
            norms = actions.norm(dim=-1, keepdim=True)
            max_norms = torch.ones_like(norms) * max_norm
            min_norms = torch.ones_like(norms) * 0
            coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
            actions = actions * coeff

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
