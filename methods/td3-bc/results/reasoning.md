Let me start from what actually hurts. I have a fixed dataset of transitions `D = {(s, a, r, s')}` collected by some policy I don't control, and I want a good continuous-control policy out of it without ever touching the environment again. The obvious move is to take the strongest off-policy actor-critic I have — TD3 — and just train it on `D` as if `D` were its replay buffer. It already learns off-policy from a buffer; the buffer is simply frozen now. So why does this fall apart offline?

I know exactly why, and I should be precise about the mechanism because the whole design is going to hinge on it. TD3's critic is fit by TD regression to a bootstrap target `y = r + γ min_i Q_{θ'_i}(s', ã)`, where `ã = π_{φ'}(s') + ε` is the *target policy's* action at the next state. Its actor is moved by the deterministic policy gradient, `∇_φ J = E_s[∇_a Q(s,a)|_{a=π(s)} ∇_φ π(s)]` — it climbs the critic's value surface. Now: where does the actor end up? Wherever the critic reports high value. And the critic, a neural net, will report *something* at every action, including actions that appear nowhere in `D`. At those out-of-distribution actions its output is pure extrapolation — there's no data anchoring it, so it can be arbitrarily wrong, and in practice it's wrong on the high side, because the actor has been actively seeking out wherever the value looks large. Online this is self-healing: the actor takes the over-valued action, the environment hands back the true (low) reward, and the next TD update corrects the critic. Offline there is no environment, so nothing ever refutes the over-valued action. Worse, the error doesn't stay local — the target `y` bootstraps `Q(s', ã)` at the target policy's own next action `ã`, which is also drifting out of distribution, so the inflated estimate gets backed up through the Bellman recursion into earlier states and compounds. The actor chases the inflation; the inflation feeds on the actor. That coupled blow-up is the offline failure: drop a fixed dataset into a strong off-policy learner and it can do no better than learning from scratch, because it is not extracting the data safely.

So the requirement is forced: I have to keep the actor's actions close enough to the data that the critic is only ever asked about state-action pairs it has support for. The question is *how*, and here I want to look hard at what everyone already does, because they all impose this "stay near the data" constraint and they all pay for it in a way that is uniquely bad offline.

The lineage I'd inherit. BCQ named the problem and fixed it by fitting a conditional VAE — a generative model of the behavior policy `π_β` — sampling candidate actions from that VAE, perturbing them a little, and letting the critic pick among them; so the actor is *structurally* confined to VAE-plausible actions. BEAR and BRAC keep an explicit parametric `π̂_β` fit by maximum likelihood and constrain the actor toward it, BRAC as a KL penalty `D_KL(π_θ, π̂_β)`, BEAR as an MMD support constraint. CQL goes the other way and regularizes the *critic* — adds a term that pushes `Q` down on out-of-distribution actions and up on dataset actions, approximated by a logsumexp over a batch of sampled actions, so the value function itself turns pessimistic about unsupported actions. Fisher-BRC, the current best, trains a generative behavior model, reparameterizes the critic as that model's log-density plus a learned offset, and regularizes the offset with a Fisher-divergence gradient penalty plus a reward bonus.

Every one of these works. But step back and tally what they cost, and do it from the offline vantage specifically, because that changes the accounting. BCQ, BEAR, BRAC, Fisher-BRC all require fitting a *second* model of the data distribution — a VAE or a density model — and the constraint is only ever as good as that fit. CQL needs a logsumexp over many sampled actions every step. On top of the named idea, each one also drags in a stack of unannounced implementation changes: CQL's reported numbers lean on an actor pre-training phase, a max-over-sampled-actions evaluation rule, removal of the SAC entropy term, modified architecture and learning rates; Fisher-BRC adds a reward bonus and the same entropy/architecture surgery. And when those implementation extras are stripped back to the base algorithm, the scores fall off a cliff on many datasets — so the headline results are riding on the un-justified extras, not just the headline idea. These also more than double the wall-clock time of the underlying algorithm.

Now here's the thing I keep coming back to. Online, an extra hyperparameter or an extra model is annoying but cheap — I can sweep it, watch the return curve, keep what works. Offline, by definition, I cannot interact, so I cannot validate any of it. Every knob I add is a knob I have to set *blind*, with no environment to tell me whether I set it right. So the cost of complexity is not linear offline; it's punishing. The community keeps adding machinery, and the machinery keeps being the thing that's actually carrying the performance, and none of it can be checked in the setting it's meant for. That reframes the goal for me. I don't want the cleverest constraint. I want the *fewest moving parts* that still keeps the actor inside the data — ideally zero extra models, near-zero extra compute, one hyperparameter — so that whatever performance I get is attributable and I have almost nothing to tune blind.

So let me ask the minimal question. What is the cheapest possible way to say "stay near the data actions"? Every method above answers this with a *learned* `π̂_β` and then a divergence to it. But why do I need to model `π_β` at all? Pause on what the divergence-to-`π̂_β` constructions are really buying. BRAC penalizes `D_KL(π_θ(·|s), π̂_β(·|s))` — but to evaluate a KL I need a *distribution* `π̂_β`, which is the whole reason a generative model gets dragged in. And is there any principled reason KL is the right divergence? When I think about it honestly: no. The goal is just "the policy's action shouldn't stray from the actions in the data." Minimizing a KL is a logical, valid way to encode that, but there's no fundamental argument that KL beats some other distance — it's a choice, and the choice is what forces the behavior model. If I'm willing to drop the insistence on a divergence between *distributions*, I can drop the behavior model with it.

Because my actor is *deterministic*. `π_φ(s)` outputs a single action vector, not a distribution. And the dataset hands me, for each state `s` it contains, the single action `a` that was actually taken there. So "stay near the data" at a sampled `(s, a)` is just: `π_φ(s)` should be near `a`. That's not a divergence between distributions at all — it's a distance between two points in action space. The simplest such distance is squared Euclidean, `(π_φ(s) - a)^2`. And I already know that object: it's exactly the behavior-cloning regression loss. So the cheapest constraint isn't a new constraint at all — it's plain behavior cloning, fit on the same `(s, a)` pairs I'm already sampling for the critic update, with no second model to fit, no density to estimate, one line of code.

But BC alone is the other failure mode — it imitates the dataset wholesale, copies the bad actions with the good, and can never exceed the behavior that generated the data. I don't want to *replace* the value objective with imitation; I want both at once. TD3's actor maximizes `E_{(s,a)~D}[Q(s, π(s))]`. So just add the imitation term as a regularizer onto that same objective:

  `π = argmax_π E_{(s,a)~D}[ Q(s, π(s)) - (π(s) - a)^2 ]`.

The first term pulls the policy toward high value; the second tethers it to the action actually seen at that state. Where they agree — the dataset action already has high value — the policy happily takes it. Where the critic wants to wander to an unsupported action, the BC term pulls it back. The actor can still *improve* on the data within the neighborhood the critic can be trusted, but it can't run off to fantasy actions, because the moment it leaves the data the quadratic penalty grows. And crucially the critic, trained on dataset `(s, a)` pairs, is now only ever asked by the actor about actions near those pairs — exactly the support region — so the extrapolation that blew everything up is starved at its source. No VAE, no `π̂_β`, no divergence machinery. One added term on the loss I already have.

Now I have to set the balance between the two terms, and this is where naively writing them with equal weight bites me. Look at the scales. Assume the actions are normalized to `[-1, 1]`, which is standard. Then the BC term, implemented as mean squared error, is bounded: each action coordinate differs by at most `2`, so the averaged squared error is at most `4` — the BC term sits in a fixed, small numeric range regardless of the task. The `Q` term has no such bound. `Q(s, a)` is an estimate of discounted return, and return scales directly with the reward magnitude, which is wildly different across tasks — a HalfCheetah running reward and a Hopper hopping reward live at completely different numeric scales, and the same task at different data qualities can too. So on one task `Q` might sit around 10 and on another around 1000, while the BC term stays pinned below `4` everywhere. If I weight them equally, then on the high-reward task the `Q` term utterly swamps the BC term — no effective constraint, back to the offline blow-up — and on a low-reward task the BC term dominates and I've got glorified imitation. A single fixed trade-off coefficient `λ` can't be right across tasks, because the thing it's trading against moves by orders of magnitude. I'd have to re-tune `λ` per dataset — and re-tuning per dataset is exactly the blind-knob cost I'm trying to eliminate. Wall.

The fix has to make the trade-off insensitive to the reward scale, which means I need to *measure* the scale of the `Q` term and divide it out. What's a cheap, robust scalar for "how big is `Q` right now"? The mean absolute value over the same batch of actor-update critic values, `(1/N) Σ_i |Q_1(s_i, π(s_i))|`. So instead of a fixed weight, define

  `λ = α / ( (1/N) Σ_i |Q_1(s_i, π(s_i))| )`,

and write the actor objective as

  `π = argmax_π E_{(s,a)~D}[ λ Q(s, π(s)) - (π(s) - a)^2 ]`.

Now follow the units. `λ Q` has magnitude roughly `α · |Q| / mean|Q| ≈ α` — the division cancels the reward scale, so the value term is pinned to about `α` no matter whether `Q` lives near 10 or near 1000. The BC term is already `O(1)`–`O(4)`. So `α` directly sets the dimensionless ratio between exploitation and imitation, and the *same* `α` is meaningful across every task. I've collapsed a per-task `λ` into a single scale-free `α`. That's the whole point of the normalizer: not to make `λ` "bigger" or "smaller," but to make one number transfer across datasets so I don't tune blind.

I have to be careful with one detail in `λ`, and it's a gradient-flow trap. `λ` contains `Q`, and I'm about to backprop through `λ Q(s, π(s))` to update the actor. If I let gradients flow through the `Q` *inside* `λ`, I get a spurious extra term — I'd be differentiating the normalizer itself, which is not what `λ` is. `λ` is a *scalar scaling factor* on the loss; its job is to set magnitude, not to contribute a gradient direction. So `λ` must be detached — computed as a number from the current batch and treated as constant for the backward pass. Only `Q(s, π(s))` (in the value term) and `(π(s) - a)^2` (the BC term) carry gradient. Concretely the actor uses `Q1` (one of the twin critics) for its gradient, the mean absolute `Q` for the detached normalizer.

There's a second, free benefit hiding in this normalization that I didn't design for but should notice, because it tells me I've got the right object. The deterministic policy gradient is `∇_a Q(s,a) · ∇_φ π`. The magnitude of `∇_a Q` *also* scales with the reward scale — a `Q` ten times larger has gradients about ten times larger. So without normalization, the effective learning rate of the value term would itself be reward-scale-dependent, and I'd be implicitly re-tuning the actor's step size every time I changed tasks. Dividing the value term by `mean|Q|` divides `∇_a Q` by `mean|Q|` too, so the gradient magnitude of the value term is normalized across tasks as a side effect. Same scalar, two problems solved: the RL/BC balance and the effective learning rate both become task-invariant. That's a sign the normalizer is the natural one, not a hack.

Now the second change, and I want to derive why it's worth it rather than bolt it on. The dataset is *fixed*. That's usually a curse, but here it's a gift I'm not using: because the data never changes, I can compute exact statistics over it once, up front, for free. Raw observation features in these continuous-control tasks are on very different scales — a joint angle, an angular velocity, a contact force all live in different numeric ranges. An MLP fed raw features has to learn to undo those scale disparities itself, and on a fixed dataset that's wasted capacity and a source of instability. So normalize each state feature to zero mean and unit variance using the dataset statistics:

  `s_i ← (s_i - μ_i) / (σ_i + ε)`,

with `μ_i, σ_i` the per-feature mean and standard deviation over `D`, and a small `ε = 10^{-3}` in the denominator to keep a near-constant feature (tiny `σ_i`) from exploding. I apply this to both `s` and `s'` so the critic's target sees the same normalized inputs, and I keep `μ, σ` so that at evaluation time I can normalize live observations the same way. This is a commonly-used trick in online deep RL, but it's *particularly* apt offline: online the running statistics drift as the data distribution shifts, but offline they're exact and stationary, so normalization is clean and unambiguous. It's not the core of the method, but it's a one-time, near-free stabilizer, so it earns its place.

Let me sanity-check that I haven't quietly changed anything else about TD3, because the entire pitch is *minimal* — the value of this approach is that performance is attributable, and that only holds if everything below the actor objective is untouched TD3. The critic still fits twin `Q_{θ1}, Q_{θ2}` by MSE to the clipped-double-Q target `y = r + γ(1-d)·min_i Q_{θ'_i}(s', ã)`, with target policy smoothing `ã = clip(π_{φ'}(s') + clip(N(0,σ), -c, c), -a_max, a_max)`, `σ = 0.2·a_max`, `c = 0.5·a_max`. The actor and the soft target updates (`τ = 5·10^{-3}`) still fire only every `policy_freq = 2` critic steps — the delayed updates that let the value settle relative to the policy. Discount `γ = 0.99`, Adam at `3·10^{-4}` for both, batch 256, `256×256` ReLU MLPs, twin critics, tanh-squashed deterministic actor. The *only* algorithmic change is the `λ Q - (π-a)^2` actor objective; the *only* implementation change is the state normalization. No generative model, no entropy term to remove, no architecture surgery, no actor pre-training, no reward bonus, no extra forward passes, no second hyperparameter. The BC term reuses the exact `(s, a)` minibatch already drawn for the critic, so its added compute is one elementwise difference and a mean — effectively free.

The single hyperparameter left is `α`. I want to reason about where it should sit. `α` is the dimensionless exploitation-to-imitation ratio after normalization. Push it small and the value term `λQ ≈ α` shrinks relative to the `O(1)` BC term, so the objective collapses toward pure imitation — I get BC, ceilinged by the data. Push it large and the value term dominates, the BC tether goes slack, and I drift back toward the unconstrained offline blow-up. So I want it in a middle band: large enough that the value term can actually improve on the data, small enough that the BC term still tethers the actor inside the support. A value around `α = 2.5` puts the normalized value term at a few times the scale of the BC term — value-led but firmly tethered — and because the normalization already made `α` task-invariant, one such value is meant to hold across the whole benchmark rather than being re-found per dataset.

Now I want to land this as the code I'd actually run, filling the two open slots — the dataset preprocessing and the actor objective — in the TD3 harness, changing nothing else. The objective is written as a maximization; in code I minimize its negative, so `argmax (λQ - (π-a)^2)` becomes a loss `-λ·Q.mean() + mse(π, a)`:

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- state-feature normalization over the fixed dataset (the second change) ----
def normalize_states(states, next_states, eps=1e-3):
    mean = states.mean(0, keepdims=True)
    std = states.std(0, keepdims=True) + eps          # eps guards near-constant features
    # keep (mean, std) to normalize live observations the same way at eval time
    return (states - mean) / std, (next_states - mean) / std, mean, std


class Actor(nn.Module):                                # deterministic policy, unchanged TD3
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):                               # twin critics, unchanged TD3
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256); self.l2 = nn.Linear(256, 256); self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256); self.l5 = nn.Linear(256, 256); self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, state, action):                       # single critic for the actor's gradient
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action = max_action
        self.discount, self.tau = discount, tau
        self.policy_noise = policy_noise * max_action   # smoothing noise scaled by action range
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha                              # the one hyperparameter: RL/BC ratio
        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ---- critic update: untouched TD3 (clipped double-Q + target policy smoothing) ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)         # min over twins: no extra overestimation
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        # ---- delayed actor update: the one algorithmic change ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()       # normalizer; detached -> a scalar scale, no gradient
            # minimize -(lambda*Q) + (pi - a)^2  <=>  maximize  lambda*Q(s, pi(s)) - (pi(s) - a)^2
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

            # soft target updates, every policy_freq steps (untouched TD3)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

The causal chain, start to finish. Off-policy actor-critic collapses offline because the actor climbs the critic toward out-of-distribution actions where the critic extrapolates and overestimates, and on a frozen dataset nothing corrects it, so the error backs up through the Bellman recursion and the actor-critic loop blows up. The accepted cure — keep the policy near the data — is implemented by everyone with a fitted behavior model and a divergence to it, which adds models, compute, and many blind hyperparameters that are uniquely costly offline where nothing can be validated. But a deterministic policy and a dataset that pairs each state with one action turn "stay near the data" from a divergence-between-distributions into a point distance, `(π(s) - a)^2`, which is just behavior cloning — no model, one line. Adding it as a regularizer onto TD3's value-maximizing actor objective keeps the value-led improvement while tethering the actor inside the support, starving the extrapolation that caused the collapse. The value and BC terms live at incomparable scales because `Q` scales with reward while the mean-squared BC term is bounded by `4`, so a fixed weight can't transfer across tasks; dividing the value term by the actor-update batch mean `|Q|` pins it to `O(α)` and collapses the trade-off into one scale-free `α` (≈ 2.5), which as a bonus also normalizes the actor's effective learning rate. Exploiting that the dataset is fixed, exact per-feature state normalization (`ε = 10^{-3}`) is a near-free stabilizer. Everything else is untouched TD3, so the algorithmic delta is exactly those two minimal changes, at essentially zero extra compute and one hyperparameter.
