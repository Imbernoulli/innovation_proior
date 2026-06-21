PPO topped the ladder by being the most balanced rung, and the numbers show exactly where its remaining weakness lives. Its Swimmer is the tightest and highest of any baseline — 111.9 / 109.7 / 118.1 for a mean of 113.2, a seed range of barely 8 points where AWR's Swimmer ranged 46–124 — so the hard per-minibatch clip did exactly what I predicted, correcting the spurious-action overshoots that cratered AWR. Its InvertedDoublePendulum is steady at 7048.4, and its HalfCheetah came in *below* AWR's mean (1757.6 vs 1996.7), as I bet, because AWR's mean was inflated by one lucky 3301 seed while PPO traded that tail for consistency (1470.6 / 1361.3 / 2441.0). PPO wins on the geometric mean by refusing to be worst anywhere — but look at where it is not *best*: it gives up peak return on HalfCheetah, it is essentially tied rather than ahead on InvertedDoublePendulum, and even its own HalfCheetah still has a one-seed tail (2441) it does not consistently reach. The diagnosis is no longer the trust region — the clip nailed that. It is *exploration*. Exploration here flows entirely through one object, the state-independent learned log-std vector `actor_logstd`; but the policy gradient, the clip, and especially the GAE advantages all *reward* sharpening the policy onto whatever has been working, and the cleanest way the optimizer raises return is to shrink the std toward a near-deterministic policy. Once it collapses, exploration is over and the policy commits to a basin — which is why two of three HalfCheetah seeds plateau near 1400 while one finds the 2441 gait. The clip bounds the *step*, not the *entropy*. Fixing this must not touch the loss the clip earned and cannot add capacity, since the parameter count is frozen — which rules out the blunt instrument of a tuned `ent_coef`, whose fixed bonus would over-perturb InvertedDoublePendulum's delicate balancing while keeping Swimmer exploring, the same per-environment-coefficient problem the penalty rung's $\beta$ had.

I propose **RPO**, Robust Policy Optimization (Rahman & Xue, 2022). Keep PPO's clipped surrogate and clipped value loss *completely unchanged*, and perturb the policy's action-mean during the update with a small uniform noise $z\sim\mathcal U(-\alpha,\alpha)$ added to `action_mean` before the log-probability is evaluated. Crucially the perturbation is applied **only during the update** — when an action is being re-scored, i.e. when `action is not None` in `get_action_and_value` — and *not* during data collection, when the policy samples actions to step the environment. So the agent still acts with its clean, unperturbed Gaussian: the rollout is undisturbed and the actions stored in the buffer are honest samples from $\pi_{old}$, but every time the loss re-evaluates $\log\pi_\theta(a|s)$ over the K epochs it does so under a *family* of slightly shifted means.

Here is why that maintains entropy without an entropy coefficient. When I re-score a stored action $a$ under a mean perturbed by $z$, the log-probability $\log\mathcal N(a;\mu+z,\sigma)$ is evaluated against a moving target. Averaged over $z\sim\mathcal U(-\alpha,\alpha)$ across the K epochs, the gradient the policy receives is no longer "concentrate all mass on the single mean that maximizes the clipped surrogate"; it is "be consistent with the action under *any* mean within $\pm\alpha$." A near-deterministic policy — tiny $\sigma$, sharp peak at $\mu$ — is *penalized* by this, because under a shifted mean $\mu+z$ a sharp Gaussian assigns the stored action a wildly different (usually much lower) log-prob, so the perturbation makes the sharp policy's objective noisy and worse on average. A policy with a wider $\sigma$ is *robust* to the shift — its log-prob barely moves when the mean wiggles by $\pm\alpha$ — so the perturbation implicitly rewards keeping the std wide. That is the mechanism: the uniform mean-perturbation is an *implicit, scale-free entropy regularizer* that fights the std collapse, and because it perturbs the mean in action-space units that the policy's own $\sigma$ adapts to, a single $\alpha$ transfers across environments far better than a single `ent_coef` would. It also smooths the loss landscape — the optimizer sees an average over a neighborhood of means rather than a single sharp surrogate — which damps exactly the over-confident sharpening that caps PPO's HalfCheetah at its lucky-seed tail.

Two design choices are load-bearing. Why uniform noise and not Gaussian: uniform gives a hard, bounded perturbation — every update sees a mean shifted by at most $\alpha$, no heavy tail — so the perturbation can never be large enough to break the trust region the clip is maintaining, whereas a Gaussian perturbation would occasionally throw a large shift that fights the clip. And why perturb the *mean* rather than the *std*: inflating the std directly would change the *sampling* distribution and, if applied at collection time, corrupt the honest $\pi_{old}$ rollout, whereas perturbing the mean only at *re-scoring* time leaves the rollout untouched and acts purely as a regularizer on the gradient. This is precisely why the perturbation must be gated on `action is not None`: in the data-collection call `action is None`, the policy samples cleanly; only in the re-scoring call, where the stored action is passed back in, does the noise enter. Getting that gate wrong — perturbing at collection — would inject noise into the behavior policy and silently break the on-policy assumption, so the `if action is None: sample else: perturb-then-rescore` structure is essential, not incidental.

The harness fit is the cleanest of any rung on this ladder. RPO changes *only* `get_action_and_value`; `compute_losses` is byte-for-byte PPO's clipped surrogate plus clipped value loss — the loss the previous rung earned, untouched, which is exactly what I wanted: improve exploration without disturbing the trust region. The edit adds three lines to the `else` branch — sample $z=$ `torch.FloatTensor(action_mean.shape).uniform_(-rpo_alpha, rpo_alpha)` on the observation's device, set `action_mean = action_mean + z`, and rebuild `probs = Normal(action_mean, action_std)` before computing the log-prob, entropy, and value against the *perturbed* distribution. The default $\alpha=0.5$ is the value reported as robust across a broad continuous-control suite; I keep it hardcoded as a local since the fixed `Args` does not expose it and the parameter-count guard forbids adding learnable state — the perturbation is sampled noise, not a parameter, so it costs nothing against the count assertion. No replay buffer, no extra network, no new config field; it slots into the frozen loop exactly. I checked this against the method's canonical reference implementation line by line: the reference perturbs the mean with `uniform_(-rpo_alpha, rpo_alpha)` in the `else` branch only, `rpo_alpha=0.5` default, reconstructs the Normal before evaluating, and its `compute_losses` is verbatim PPO — identical down to the gate and the default, the only cosmetic differences being that the edit hardcodes $\alpha=0.5$ as a local rather than reading a config field and uses `obs.device` rather than a module-level `device`.

This rung carries no measured result — the numbers it must beat are PPO's — so the bar is concrete and three-fold. First, HalfCheetah: I expect RPO to lift the mean above 1757.6 and, more tellingly, to *narrow* the gap between the 1400-plateau seeds and the 2441 tail, so I would validate by checking whether the per-seed minimum rises, not just the mean — sustained exploration should let more seeds find the better gait. Second, Swimmer: PPO already nailed it at 113.2 with tight seeds, so the bar is *do no harm* — the perturbation must not destabilize the long-horizon credit assignment, so I would validate that RPO stays in PPO's tight band rather than reintroducing AWR's collapse. Third, InvertedDoublePendulum: this is the risk environment, an unstable balancing task where too much exploration can knock the pole over, so the bar is that $\alpha=0.5$ does not over-perturb it below 7048.4, and I would watch the worst seed most closely. The overall geometric-mean claim is that RPO clears PPO by improving the dense, exploration-limited environment while holding the other two — a scale-free implicit-entropy regularizer that fixes the one place PPO leaves return on the table, without touching the trust region that made PPO reliable. If instead the perturbation drags InvertedDoublePendulum or destabilizes Swimmer, that would falsify the claim that $\alpha=0.5$ transfers across these three dynamics, and the move past it would be to anneal $\alpha$ or make it state-dependent — but the bet this finale makes is that the fixed perturbation holds.

```python
    def get_action_and_value(self, obs, action=None):
        rpo_alpha = 0.5
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            # RPO: add uniform noise to action mean during update
            z = torch.FloatTensor(action_mean.shape).uniform_(-rpo_alpha, rpo_alpha).to(obs.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """PPO clipped surrogate objective + clipped value loss (same as PPO)."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — clipped surrogate
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss — clipped
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
