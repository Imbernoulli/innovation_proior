The loop hands me one on-policy batch per iteration and asks me to run ten epochs of mini-batch SGD over the same 2048 transitions before discarding them. That reuse is the entire economy of an on-policy actor-critic — it is how I extract many gradient steps from data that cost a full rollout to collect — but it is also the precise thing that makes naive ascent dangerous. The scaffold's placeholder fill is the un-clipped policy gradient $-\hat{\mathbb E}[\hat A\,r]$ with $r_t=\pi_\theta/\pi_{old}$, and over ten epochs it walks the policy off a cliff: as $\theta$ drifts from the data-generating $\theta_{old}$ the ratio fans out, the optimizer discovers that the cheapest way to raise the objective on a positive-advantage sample is to *inflate* $r_t$ rather than find a genuinely better action, and a handful of large ratios yank the update around until the policy collapses. So the first update rule I write has to put a leash on how far the policy may move per batch, and I want the *most theoretically direct* leash — the one the surrogate bound literally hands me — to see exactly how far it gets and what it leaves for the next rung to fix.

I propose **PPO-Penalty**: the adaptive-KL-penalty policy update. The bound to attach to is Kakade–Langford's: $\eta(\pi)-\eta(\pi_{old})$ equals the new policy's expected old-advantage, and evaluated under the old state distribution it becomes the surrogate $L(\theta)=\hat{\mathbb E}_t[r_t(\theta)\hat A_t]$. This is honest only to first order at $\theta=\theta_{old}$, where every $r_t=1$; its error grows with how far $\pi_\theta$ strays from $\pi_{old}$, and that distance is bounded by a KL term. The cleanest way to express "stay close" as a first-order loss — no constraint solver, no second-order machinery — is to subtract a KL penalty and maximize

$$\hat{\mathbb E}_t\big[\,r_t\,\hat A_t-\beta\,\mathrm{KL}[\pi_{old},\pi_\theta]\,\big],$$

with the *un-clipped* ratio $r_t$. I start here precisely because the penalty is *just a loss*: it differentiates cleanly, it costs nothing beyond the ratio I already compute, and it fits the shared-optimizer K-epoch loop the scaffold gives me. The job of holding the policy near $\pi_{old}$ is delegated entirely to the KL penalty — there is no ratio clipping anywhere — which is exactly what distinguishes this rung from the clipped variants further up the ladder.

The catch, and the thing that defines the method, is the coefficient $\beta$. The value the bound itself prescribes comes from a worst-case max-over-states KL inequality, so it is enormous and the permitted steps are microscopic — correct but useless, no better than not reusing the data at all. And a fixed hand-picked $\beta$ will not hold still, because it has to balance $r\hat A$ against $\beta\,\mathrm{KL}$, and that balance depends on the *scale* of the advantages and on how sensitive the KL is to a parameter step. Both change across the three environments — HalfCheetah's dense rewards produce very different advantage magnitudes than Swimmer's even after the loop's per-minibatch normalization, once the value function is inaccurate — and both change *over a single run* as the policy sharpens and the returns grow. A $\beta$ that gives reasonable steps at iteration 1 gives tiny steps at iteration 400. So I stop *guessing* $\beta$ and start *servoing* it: pick a target KL $d_{targ}$ — the size of policy move I will tolerate per batch — and after each update measure the realized KL. If it overshot, the penalty was too weak, so multiply $\beta$ up; if it undershot, the penalty was too strong, so divide $\beta$ down. This is why the adaptive version works where a fixed coefficient cannot — I am no longer fighting the advantage scale or the run-time drift by hand, I am closing a feedback loop on the quantity I actually care about.

Two harness details shape the implementation, and both are load-bearing. First, `compute_losses` is a free function called fresh on every minibatch — there is no persistent place to keep $\beta$ across iterations. So I attach the adaptive state to the `agent` object itself: on the first call I lazily initialize `agent._kl_beta = 0.5` and `agent._target_kl = 0.01`, then read and mutate them in place thereafter, so the coefficient persists across minibatches, epochs, and iterations exactly as the servo needs. The $d_{targ}=0.01$ is the standard small-move regime where the surrogate stays tight; $\beta=0.5$ is a neutral start that the servo pulls to the right scale within a handful of updates regardless. Second is *which* KL I penalize. The exact diagonal-Gaussian KL is available, but the loop's natural vocabulary is the log-ratio, and the cheap estimator that lives here is $\hat{\mathrm{KL}}=\hat{\mathbb E}[(r-1)-\log r]$ — the same quantity the scaffold already computes for `approx_kl`. I reuse it, but with one critical difference: the diagnostic is computed under `torch.no_grad()`, whereas my *penalty* term must carry a gradient. The penalty is the entire mechanism by which "stay close" reaches the policy parameters; if I detached it, the KL term would contribute nothing to the gradient and I would be back to the un-clipped placeholder with a useless constant added. So I compute `kl = ((ratio - 1) - logratio).mean()` *with* gradient and use it in the policy loss, and keep a detached copy purely for the adaptation rule and logging. Getting that `detach` placement backwards silently turns the method into the broken default, so it is worth stating plainly: penalty KL has a gradient, adaptation KL does not.

The adaptation rule itself is a banded geometric servo. After computing the loss I read the detached realized KL; if it exceeds $1.5\times d_{targ}$ I double $\beta$ (capped at $100$ so a runaway minibatch cannot blow the coefficient up), and if it falls below $d_{targ}/1.5$ I halve $\beta$ (floored at $10^{-4}$ so it can always recover). The $1.5$ band is a dead-zone so the servo is not thrashing on every minibatch's noise; the doubling/halving gives it geometric reach to cross several orders of magnitude in a few iterations if the advantage scale demands it. The value head gets a plain MSE loss toward the GAE returns — no value clipping, since I have introduced no clipping discipline anywhere in this rung and adding it asymmetrically would be incoherent — folded in with the loop's `vf_coef`, and the entropy term enters through `ent_coef` (which defaults to 0 on MuJoCo, where the Gaussian's learned log-std already supplies exploration).

I should be clear about what this rung is *not*, so the implementation lands the harness's version and not an imported one. The adaptation happens *inline*, per minibatch, mutating `agent._kl_beta` as the K epochs proceed — not an outer per-iteration servo that re-runs the whole batch at fixed $\beta$ and only then adjusts; that is finer-grained and a little noisier but is the only shape the free-function contract supports cleanly. There is no separate KL early-stopping break (`target_kl` in the loop is `None`); the penalty is the sole brake. And the KL is the cheap log-ratio estimator, not the closed-form Gaussian KL.

This is a working on-policy method, so I expect real learning on all three environments — but I expect it to be the *least balanced* update rule on the ladder, and that is the weakness it leaves on the table. The inline adaptation is reactive: it only shrinks $\beta$ *after* a minibatch has already overshot the KL, so within a batch the policy can take a too-large step before the coefficient catches up, and that occasional overshoot is exactly the kind of instability that inflates seed-to-seed variance. And a *soft* penalty asks the optimizer to *trade off* return against distance, so on a noisy minibatch it will sometimes pay the KL cost to chase a large advantage where a hard band would simply forbid the move. Because the task scores by geometric mean across the three environments, any single environment where the penalty servo gets unlucky drags the whole score down hard. If that is what the numbers say, the next move writes itself: replace the soft, reactive KL penalty with a leash built into the loss itself that cannot be traded away.

```python
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """PPO-Penalty: adaptive KL penalty instead of clipped surrogate."""
    if not hasattr(agent, '_kl_beta'):
        agent._kl_beta = 0.5
        agent._target_kl = 0.01

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # KL divergence — WITH gradient for the penalty term
    kl = ((ratio - 1) - logratio).mean()

    with torch.no_grad():
        approx_kl = kl.detach()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — KL-penalized (no clipping)
    pg_loss = -(mb_advantages * ratio).mean() + agent._kl_beta * kl

    # Adapt KL penalty coefficient
    with torch.no_grad():
        if approx_kl > 1.5 * agent._target_kl:
            agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
        elif approx_kl < agent._target_kl / 1.5:
            agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)

    # Value loss — simple MSE (no clipping)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
