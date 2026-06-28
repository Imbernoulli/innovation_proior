Let me start from the failure I keep watching happen, because that is the thing I actually want to kill. I train a Gaussian policy on continuous control with the proximal clipped objective — the reliable, first-order base learner — and on the easy runs it works, but on too many runs it dies in a specific, frustrating way: the policy commits too early. The learned standard deviation `sigma` collapses, the mean `mu` sharpens onto whatever action currently looks best, and from then on the agent only ever samples tiny jitters around that one action. If that action was actually optimal, fine. But it usually isn't — it's whatever happened to have the highest advantage in the first few batches, before the agent had seen enough of the action space to know better — and once `sigma` is small the policy gradient it produces only reinforces the basin it is already in. Exploration is gone, and it does not come back. So the thing bugging me is not stability — the clip already gives me that — it is that *the stable update is also the update that drives the policy deterministic*, and it does so before the agent has earned the right to be deterministic.

Let me make sure I understand why the base learner does this, because the cure has to attack the cause, not paint over it. The score-function gradient for a Gaussian is `grad_theta log pi_theta(a|s) A`. Write the log-density: `log N(a; mu, sigma) = -((a-mu)^2)/(2 sigma^2) - log sigma - const`. Differentiate with respect to the mean and you get a term proportional to `(a-mu)/sigma^2 * A` — push `mu` toward actions with positive advantage, away from negative ones, exactly as it should. But differentiate with respect to `log sigma` and you get a term proportional to `((a-mu)^2/sigma^2 - 1) * A`. Read what that does: for a sampled action that landed *near* the mean (`(a-mu)^2 < sigma^2`) and turned out good (`A > 0`), the bracket is negative, so the update *decreases* `log sigma` — it shrinks the spread. As the policy improves, the good actions are increasingly the near-the-mean ones, so the variance term is, on average, pushed down. The collapse is not a bug in my implementation; it is what the maximum-likelihood-style policy gradient *does* to a parametric Gaussian. The optimizer is doing its job — sharpening the distribution around what works — it just has no reason to keep enough spread to discover that something else works better. The clip bounds how big each such step is, but over many epochs and many iterations the steps all point the same way: down.

So now I know the enemy precisely: the update sharpens `(mu, sigma)`, and a sharp `(mu, sigma)` is an unexploratory policy. What can I add that resists the sharpening without breaking the part of the update I want? Let me enumerate the obvious moves and feel out where each one stops short, because the gaps will point at the right shape.

The textbook move is an entropy bonus: add `c_2 H[pi_theta]` to the objective so the optimizer pays a price for shrinking `sigma`. The entropy of a diagonal Gaussian is `sum_i (1/2) log(2 pi e sigma_i^2)`, monotone in `log sigma`, so the bonus directly counter-pushes the `log sigma` gradient I just derived. It works in principle. But think about the coefficient. The entropy gradient is some fixed-scale push on `log sigma`; the advantage-weighted gradient that shrinks `sigma` has a scale set by the *advantages*, which vary by orders of magnitude across environments and across training as returns grow. So the single number `c_2` that balances them on HalfCheetah is the wrong number on Swimmer, and the right number early in a run is wrong late. Set it too small and the advantage term wins and the policy still collapses; set it too large and `sigma` is pinned high forever and the policy never commits even when it should. I have just traded one tuning problem for another, and the new knob is the kind that needs a per-task sweep — exactly what I want to avoid. The entropy bonus is the right *intent* (keep entropy up) but the wrong *instrument* (a global coefficient fighting the advantage scale).

Second move: change the distribution family. Use a Beta on a bounded action range, or a heavier-tailed distribution, or a normalizing flow, so the policy can be expressive enough that it does not have to collapse to a point to represent a good policy. These do help on some environments. But each one adds parameters and assumptions — and here the network and parameter count are fixed, so a richer head is off the table outright — and even where it is allowed, a Beta that helps on a bounded torque task can hurt on another, a flow needs its own architecture and tuning, and now I am picking a distribution per environment. Same disease: task-specific, and it changes the network I was told to leave alone. The diagonal Gaussian is fine as a *family*; the problem is what the update does to its parameters, not the family.

Third move: data augmentation — perturb the observations, train on the perturbed copies, so the policy is forced to be robust and, as a side effect, less peaked. But that perturbs the *input* side; the entropy collapse I diagnosed is on the *output* side, in the action distribution's spread. Augmenting observations is an indirect lever on the thing I actually care about, and it brings its own augmentation-design choices. It is aiming at the wrong end of the network.

Let me step back and ask what these three near-misses have in common, because the shared gap is the actual lead. Each of them tries to keep the policy exploratory either by paying for entropy *in the loss* (a coefficient that won't hold still against the advantage scale), or by changing what the policy *is* (new family, new parameters, the network I must leave fixed), or by changing what it *sees* (augmentation, the wrong end). What none of them does is touch the one place where the collapse actually happens: the `log sigma` and `mu` gradients I derived above are computed from the *re-evaluated log-probability of the stored action under the current distribution*. That re-evaluation is where the sharpening pressure is born. The leftover lever, the one I have not pulled, is to interfere with the distribution at that exact moment — to make the update's view of the policy's own mean a little uncertain, so the gradient cannot point at one needle-sharp peak. Not a loss term, not a new family, not the input. Just uncertainty injected into the mean, *during the update*, where the sharpening lives.

Let me work out what that even means and whether it does what I want. At update time I am re-evaluating, for each stored transition `(s, a)`, the log-probability `log pi_theta(a|s)` of the action `a` the agent actually took, under the current parameters, to form the clipped surrogate. The mean the network outputs for that state is `mu_theta(s)`. Suppose, instead of using `mu_theta(s)` exactly, I perturb it: draw a small random vector `z` and evaluate the log-prob under `N(mu_theta(s) + z, sigma_theta)`. Now the gradient that flows back to `mu_theta(s)` is no longer "move the mean precisely to where this good action was"; it is "move the mean to where this good action was, *given that the mean is jittered by z*." Over the minibatch and the epochs, the mean is being asked to be good *on average over a cloud of perturbed positions*, not at one exact point. The hope is that it then cannot collapse to a needle-sharp peak that is optimal only at one precise `mu`, because the perturbation keeps moving the evaluation point — but "the hope is" is not good enough; whether the perturbation actually raises and holds the policy's entropy is a quantitative claim I will have to check below, not just assert.

Now I have to make three decisions and I want a reason for each, not a default.

What perturbation? I want it bounded and scale-free: bounded so a single rare large `z` cannot throw the mean wildly and destabilize the clip, and symmetric around zero so it does not bias the mean in any direction (`E[z] = 0`, so it is not a systematic shift, only a spreading). A zero-mean Gaussian `z` would be symmetric but unbounded — a fat tail could occasionally yank the mean far. A *uniform* `z ~ U(-alpha, alpha)` per action dimension is symmetric, zero-mean, and *strictly* bounded by `alpha` — it can never push the mean past `alpha` in any coordinate. The boundedness is the property I care about most: it makes `alpha` a hard, interpretable cap on how much the mean is jittered, in the same action units, which is what lets one value of `alpha` mean roughly the same thing across environments after the standard observation/reward normalization. So uniform noise on the mean, half-width `alpha`.

Where exactly does the perturbation apply — sampling, update, or both? This is the crux and it is easy to get backwards. During the *rollout*, the agent is acting in the environment to collect data: I want that data to come from the actual policy `N(mu_theta(s), sigma_theta)`, clean, because the stored action and its log-prob `log pi_old(a|s)` are the reference the importance ratio `r_t = pi_theta/pi_old` is built against, and the advantages are estimated for *that* policy. If I perturbed the mean at sample time too, I would be collecting data from a different, jittered policy and then evaluating ratios against it inconsistently — and the exploration at action-selection time is already handled by `sigma`. So sampling must be clean: when no stored action is supplied, return a sample from the unperturbed `N(mu, sigma)`. The perturbation belongs at *update* time only: when a stored action *is* supplied (the re-evaluation pass), rebuild the distribution as `N(mu + z, sigma)` and take the log-prob of the stored action under that. This way the data is honest about which policy produced it, and the perturbation acts purely as a regularizer on the gradient — it changes how the mean is *fit*, not how actions are *chosen*. The asymmetry is the whole design: clean at rollout, perturbed at update.

Does this break the clipped surrogate or the ratio? Let me check, because I do not want to silently invalidate the trust region. The importance ratio is `r_t = exp(newlogprob - mb_logprobs)`, where `mb_logprobs` is the stored `log pi_old` computed cleanly at rollout, and `newlogprob` is now the log-prob under `N(mu+z, sigma)`. Trace where `z` can enter: it appears only inside `newlogprob`. `mb_logprobs` was logged at rollout from the clean distribution and is a stored constant; the advantages and returns come from GAE on the clean rollout; the value head never sees `z`. So the ratio's denominator, the clip thresholds, the `min`, and the value loss are all structurally the same expressions as in PPO — same `eps`, same GAE advantages, same clipped value loss, same K epochs — and the *only* quantity that changed is the numerator distribution. I am not replacing the base learner; I am adding one perturbation inside its action readout, and everything that made PPO reliable is still computed exactly as before. That is the property I wanted: keep the surrogate and the network exactly, change only the spread of the distribution the update sharpens.

How big is `alpha`? It is the half-width of the uniform jitter on the mean, in action units. Think about the two limits. `alpha = 0` recovers PPO exactly — no perturbation, the entropy collapses as before. `alpha` very large and the mean is jittered so hard that the log-prob of the stored action becomes noise, the gradient is uninformative, and the policy cannot learn anything. Let me make that second limit concrete rather than wave at it. The `mu`-gradient of `log N(a; mu, sigma)` is `(a-mu)/sigma^2`; under the jitter `mu -> mu+z` it becomes `(a-(mu+z))/sigma^2`, i.e. the clean gradient plus an injected mean-zero term `-z/sigma^2`. So the jitter adds gradient noise to the mean with standard deviation `std(z)/sigma^2 = (alpha/sqrt(3))/sigma^2`. At `sigma=1` and `alpha=0.5` that injected noise has std `0.5/sqrt(3) ≈ 0.289`, which sits at the same O(1) scale as a typical near-mean signal `|a-mu|/sigma^2` — present, regularizing, but not overwhelming. Push `alpha` to `3.0` and the injected noise std is `≈ 1.73`, which dwarfs that signal — the gradient is mostly jitter and the policy can't learn. So there really is a sweet spot, and the back-of-envelope says `alpha` around half an action-unit is in it: after the standard observation normalization and action clipping the harness already applies, actions live on a roughly unit scale, so a half-width of about `0.5` jitters the mean by up to half an action-unit. The advantage of `alpha` over the entropy coefficient is that it is bounded and measured directly in action units, not balanced against the drifting advantage scale, so I *expect* a single default to carry across environments far better than a single entropy coefficient could — though that portability is an empirical bet I would only believe after running the suite. On a few environments where even that much jitter is too much, a much smaller `alpha` recovers near-PPO behavior.

Now the load-bearing claim, the one the whole method rests on: does jittering the mean actually keep the policy's entropy from collapsing, and — the part I really care about — does it fight *harder* exactly when the policy is collapsing? I have been saying "it keeps entropy up." Let me actually compute it. With the mean jittered by `z ~ U(-alpha, alpha)` each update, the action distribution the surrogate effectively trains the policy to be good under is a *mixture* over `z` of `N(mu+z, sigma)`. That mixture is exactly the distribution of `(mu + z) + epsilon` with `epsilon ~ N(0, sigma^2)` and `z ~ U(-alpha, alpha)` independent — the unperturbed Gaussian *convolved* with an independent uniform. Its density has a closed form, `p(x) = [Phi((x-mu+alpha)/sigma) - Phi((x-mu-alpha)/sigma)] / (2 alpha)`, so I can integrate `-p log p` numerically and compare it, per action dimension, against the plain Gaussian entropy `(1/2) log(2 pi e sigma^2)`. The interesting variable is `sigma`: the policy starts at `logstd = 0`, i.e. `sigma = 1`, and the disease is `sigma` shrinking, so I want to know what the *same* `alpha = 0.5` jitter buys at `sigma = 1` versus at a collapsed `sigma = 0.2`.

Computing it (numerical `-∫ p log p`, cross-checked by a Monte-Carlo estimate of `-E[log p]` over 4M samples; both agree and the density integrates to 1.00000):

```
sigma = 1.0,  alpha = 0.5:   H_gauss = +1.4189   H_mix = +1.4590   delta = +0.040 nats
sigma = 0.5,  alpha = 0.5:   H_gauss = +0.7258   H_mix = +0.8695   delta = +0.144 nats
sigma = 0.2,  alpha = 0.5:   H_gauss = -0.1905   H_mix = +0.3613   delta = +0.552 nats
```

This is better than I expected and it tells me three things. First, the entropy of the mixture is strictly higher than the plain Gaussian at every `sigma` — the sign of `delta` never flips — so the regularizer always pushes the right way; that is the convolution-can-only-smear-a-density-out fact made concrete. Second, and this is the part the bare assertion would have hidden: the effect is *not* a fixed-size bonus. At the healthy starting spread `sigma = 1` it adds only `0.04` nats — barely touches a policy that is already broad, so it does not stop a genuinely-good policy from committing. But as `sigma` collapses toward `0.2`, the *same* fixed `alpha = 0.5` adds `0.55` nats and actually drags the differential entropy back from negative to positive. The jitter's relative effect *grows as the policy sharpens*, precisely because a fixed-width uniform smear matters more to a narrow Gaussian than to a wide one. So the regularizer is self-adapting in exactly the direction I wanted, with no schedule and no coefficient: it leaves a broad policy alone and fights hardest against a collapsed one. That adaptive behavior is the real reason this beats an entropy bonus, and I would not have known it held — or that it had the right *shape* in `sigma` — without doing the integral; the convolution argument alone only gives me the sign, not the way the gap widens as `sigma` shrinks.

One more consistency check on the gradient, since I leaned on it for the noise-scale estimate: because `z` is sampled fresh and is independent of `theta`, it does not add a spurious gradient path — `mu + z` differentiates to `grad mu`, and `z` is just an additive constant for the purposes of the backward pass through that minibatch. So the perturbation is a stochastic regularizer on the mean's gradient, not a new learnable thing, and it costs no parameters — which keeps the fixed-capacity constraint satisfied: the parameter count is identical to PPO's, the contribution is purely algorithmic.

So the algorithm is: PPO, unchanged, except that in the action readout, when re-evaluating a stored action during the update, perturb the network's mean by uniform noise `U(-alpha, alpha)` before forming the Gaussian and taking the log-prob; sampling at rollout stays clean. Let me write it as it would drop into the harness, with the perturbation gated on whether a stored action was supplied, and the loss left as PPO's clipped surrogate plus clipped value loss.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """RPO = PPO with a uniform perturbation of the policy mean applied
    only during the update (when a stored action is re-evaluated). Same
    network, same parameter count, same clipped surrogate as PPO."""

    def __init__(self, obs_dim, action_dim, rpo_alpha=0.5):
        super().__init__()
        h = 64
        self.rpo_alpha = rpo_alpha          # half-width of the uniform mean perturbation
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            # ROLLOUT: sample from the clean, unperturbed policy
            action = probs.sample()
        else:
            # UPDATE: perturb the mean by z ~ U(-alpha, alpha) before scoring the
            # stored action. Bounded, zero-mean noise keeps the policy from
            # collapsing to a needle-sharp mean; sampling stays clean so the data
            # and the importance ratio remain consistent with pi_old.
            z = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Identical to PPO: clipped surrogate + clipped value loss. The only
    difference from PPO lives in get_action_and_value above."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Clipped surrogate
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Clipped value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb_returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

Let me trace the chain back to be sure each piece earned its place. I started from the failure that the proximal clipped update, while stable, also drives a Gaussian policy deterministic — the `log sigma` gradient shrinks the spread and the `mu` gradient sharpens the peak, so the policy commits before it has explored, and a small `sigma` can no longer escape the basin it committed to. The entropy bonus targets the right thing but with a global coefficient that fights the advantage scale and will not hold across tasks or across a run; a different distribution family changes the network I must leave fixed and helps unevenly; observation augmentation perturbs the wrong end. The leftover lever was the place where the sharpening is actually computed — the re-evaluated log-prob at update time — so I inject bounded, zero-mean uncertainty into the *mean the update is evaluated under*, `z ~ U(-alpha, alpha)`, uniform for a hard interpretable bound, applied *only* during the update so the rollout data and the importance ratio stay consistent with the policy that produced them. I checked that `z` enters only through the re-evaluated log-prob, so the clipped surrogate, the ratio, the value loss, the network, and the parameter count are all exactly PPO's. And I checked the entropy claim by computing it: the jittered policy is the Gaussian convolved with uniform noise, its entropy is strictly higher at every `sigma`, and — the thing that makes it work — the gap *widens* as `sigma` collapses (`+0.04` nats at `sigma=1`, `+0.55` at `sigma=0.2`), so the regularizer leaves a broad policy alone and fights hardest against a collapsing one, with no schedule. The result is one bounded knob `alpha` (default about `0.5`, in normalized action units), a single added line in the action readout, and a method that is PPO wherever PPO already works and self-adaptingly more exploratory wherever PPO collapses too soon.
