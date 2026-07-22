Let me start from what actually hurts. I have a fixed dataset of transitions `D = {(s, a, r, s')}` collected by some policy I don't control, and I want a good continuous-control policy out of it without ever touching the environment again. The obvious move is to take the strongest off-policy actor-critic I have — TD3 — and just train it on `D` as if `D` were its replay buffer. It already learns off-policy from a buffer; the buffer is simply frozen now. So why does this fall apart offline?

Let me trace the failure rather than assume it. TD3's critic is fit by TD regression to a bootstrap target `y = r + γ min_i Q_{θ'_i}(s', ã)`, where `ã = π_{φ'}(s') + ε` is the *target policy's* action at the next state. Its actor is moved by the deterministic policy gradient, `∇_φ J = E_s[∇_a Q(s,a)|_{a=π(s)} ∇_φ π(s)]` — it climbs the critic's value surface. Now: where does the actor end up? Wherever the critic reports high value. And the critic, a neural net, will report *something* at every action, including actions that appear nowhere in `D`. At those out-of-distribution actions its output is pure extrapolation — there's no data anchoring it, so it can be arbitrarily wrong, and in practice it's wrong on the high side, because the actor has been actively seeking out wherever the value looks large. Online this is self-healing: the actor takes the over-valued action, the environment hands back the true (low) reward, and the next TD update corrects the critic. Offline there is no environment, so nothing ever refutes the over-valued action. Worse, the error doesn't stay local — the target `y` bootstraps `Q(s', ã)` at the target policy's own next action `ã`, which is also drifting out of distribution, so the inflated estimate gets backed up through the Bellman recursion into earlier states and compounds. The actor chases the inflation; the inflation feeds on the actor. That coupled blow-up is the offline failure: drop a fixed dataset into a strong off-policy learner and it can do no better than learning from scratch, because it is not extracting the data safely.

So the requirement is forced: I have to keep the actor's actions close enough to the data that the critic is only ever asked about state-action pairs it has support for. The question is *how*, and here I want to look hard at what everyone already does, because they all impose this "stay near the data" constraint and they all pay for it in a way that is uniquely bad offline.

The lineage I'd inherit. BCQ named the problem and fixed it by fitting a conditional VAE — a generative model of the behavior policy `π_β` — sampling candidate actions from that VAE, perturbing them a little, and letting the critic pick among them; so the actor is *structurally* confined to VAE-plausible actions. BEAR and BRAC keep an explicit parametric `π̂_β` fit by maximum likelihood and constrain the actor toward it, BRAC as a KL penalty `D_KL(π_θ, π̂_β)`, BEAR as an MMD support constraint. CQL goes the other way and regularizes the *critic* — adds a term that pushes `Q` down on out-of-distribution actions and up on dataset actions, approximated by a logsumexp over a batch of sampled actions, so the value function itself turns pessimistic about unsupported actions. Fisher-BRC, the current best, trains a generative behavior model, reparameterizes the critic as that model's log-density plus a learned offset, and regularizes the offset with a Fisher-divergence gradient penalty plus a reward bonus.

Every one of these works. But step back and tally what they cost, and do it from the offline vantage specifically, because that changes the accounting. BCQ, BEAR, BRAC, Fisher-BRC all require fitting a *second* model of the data distribution — a VAE or a density model — and the constraint is only ever as good as that fit. CQL needs a logsumexp over many sampled actions every step. On top of the named idea, each one also drags in a stack of unannounced implementation changes: CQL's reported numbers lean on an actor pre-training phase, a max-over-sampled-actions evaluation rule, removal of the SAC entropy term, modified architecture and learning rates; Fisher-BRC adds a reward bonus and the same entropy/architecture surgery. And when those implementation extras are stripped back to the base algorithm, the scores fall off a cliff on many datasets — so the headline results are riding on the un-justified extras, not just the headline idea. These also more than double the wall-clock time of the underlying algorithm.

Now here's the thing I keep coming back to. Online, an extra hyperparameter or an extra model is annoying but cheap — I can sweep it, watch the return curve, keep what works. Offline, by definition, I cannot interact, so I cannot validate any of it. Every knob I add is a knob I have to set *blind*, with no environment to tell me whether I set it right. So the cost of complexity is not linear offline; it's punishing. The community keeps adding machinery, and the machinery keeps being the thing that's actually carrying the performance, and none of it can be checked in the setting it's meant for. That reframes the goal for me. I don't want the cleverest constraint. I want the *fewest moving parts* that still keeps the actor inside the data — ideally zero extra models, near-zero extra compute, one hyperparameter — so that whatever performance I get is attributable and I have almost nothing to tune blind.

So let me ask the minimal question. What is the cheapest possible way to say "stay near the data actions"? Every method above answers this with a *learned* `π̂_β` and then a divergence to it. But why do I need to model `π_β` at all? Pause on what the divergence-to-`π̂_β` constructions are really buying. BRAC penalizes `D_KL(π_θ(·|s), π̂_β(·|s))` — but to evaluate a KL I need a *distribution* `π̂_β`, which is the whole reason a generative model gets dragged in. And is there any principled reason KL is the right divergence? When I think about it honestly: no. The goal is just "the policy's action shouldn't stray from the actions in the data." Minimizing a KL is a logical, valid way to encode that, but there's no fundamental argument that KL beats some other distance — it's a choice, and the choice is what forces the behavior model. If I'm willing to drop the insistence on a divergence between *distributions*, I can drop the behavior model with it.

Because my actor is *deterministic*. `π_φ(s)` outputs a single action vector, not a distribution. And the dataset hands me, for each state `s` it contains, the single action `a` that was actually taken there. So "stay near the data" at a sampled `(s, a)` is just: `π_φ(s)` should be near `a`. That's not a divergence between distributions at all — it's a distance between two points in action space. The simplest such distance is squared Euclidean, `(π_φ(s) - a)^2`. And I already know that object: it's exactly the behavior-cloning regression loss. So the cheapest constraint isn't a new constraint at all — it's plain behavior cloning, fit on the same `(s, a)` pairs I'm already sampling for the critic update, with no second model to fit, no density to estimate, one line of code.

But BC alone is the other failure mode — it imitates the dataset wholesale, copies the bad actions with the good, and can never exceed the behavior that generated the data. I don't want to *replace* the value objective with imitation; I want both at once. TD3's actor maximizes `E_{(s,a)~D}[Q(s, π(s))]`. So just add the imitation term as a regularizer onto that same objective:

  `π = argmax_π E_{(s,a)~D}[ Q(s, π(s)) - (π(s) - a)^2 ]`.

The first term pulls the policy toward high value; the second tethers it to the action actually seen at that state. Where they agree — the dataset action already has high value — the policy happily takes it. Where the critic wants to wander to an unsupported action, the BC term pulls it back. The actor can still *improve* on the data within the neighborhood the critic can be trusted, but it can't run off to fantasy actions, because the moment it leaves the data the quadratic penalty grows. And crucially the critic, trained on dataset `(s, a)` pairs, is now only ever asked by the actor about actions near those pairs — exactly the support region — so the extrapolation that blew everything up is starved at its source. No VAE, no `π̂_β`, no divergence machinery. One added term on the loss I already have.

Now I have to set the balance between the two terms, and this is where naively writing them with equal weight bites me. Before I trust my intuition that the scales are mismatched, let me actually put numbers on both terms. Assume the actions are normalized to `[-1, 1]`, which is standard. The BC term is a mean squared error: per coordinate `(π_i - a_i)^2`, and since both `π_i` and `a_i` live in `[-1, 1]` the largest a single coordinate's squared error can be is `(1 - (-1))^2 = 4`, attained only when the policy maxes out at exactly the opposite corner from the data action. Averaged over coordinates the term is therefore in `[0, 4]`, and at a typical mid-training mismatch of, say, `0.3` per coordinate it sits around `0.09` — call it `O(0.1)` to `O(1)` in practice, capped at `4`. So the BC term is bounded and task-independent: it lives in the same small numeric window on every dataset.

The `Q` term has no such bound. `Q(s, a)` estimates discounted return `Σ_t γ^t r_t`. Let me actually compute the order of magnitude this implies. With `γ = 0.99` the effective horizon is `1/(1-γ) = 100` steps, so `Q ≈ 100 · r̄` for a typical per-step reward `r̄`. Now plug in two real D4RL tasks. HalfCheetah's per-step running reward at medium quality is on the order of `r̄ ≈ 5`, giving `Q ≈ 500`; expert-quality data pushes `r̄` higher and `Q` into the thousands. Hopper's per-step reward is roughly `r̄ ≈ 1`–`2` (a small alive bonus plus forward velocity), giving `Q ≈ 100`–`200`; a `random`-data Hopper sits far lower still. So just from `Q ≈ 100·r̄` I get values spanning roughly `Q ~ 10²` to `Q ~ 10³` across the benchmark — two orders of magnitude — while the BC term never leaves `[0, 4]`. That's the mismatch made concrete, not assumed.

Now feed those numbers into an equally-weighted objective `Q - (π-a)^2` and see what each term contributes to the gradient balance. On expert HalfCheetah, `Q`-scale `≈ 1000` against a BC term `≈ 1`: the value term outweighs the tether by a factor of ~1000, so the BC constraint is effectively absent and I'm back to the unconstrained offline blow-up. On a low-return Hopper dataset, `Q`-scale `≈ 50` against a BC term that can reach `4`: now the two are within an order of magnitude and the BC term genuinely competes, so I get glorified imitation. *Same* fixed objective, opposite behavior on the two tasks — which is exactly what a single fixed trade-off coefficient `λ` cannot fix, because the quantity `λ` multiplies moves by 100× between tasks while the thing it trades against stays put. I'd have to re-tune `λ` per dataset — and re-tuning per dataset is exactly the blind-knob cost I'm trying to eliminate. Wall.

The fix has to make the trade-off insensitive to the reward scale, which means I need to *measure* the scale of the `Q` term and divide it out. What's a cheap, robust scalar for "how big is `Q` right now"? The mean absolute value over the same batch of actor-update critic values, `(1/N) Σ_i |Q_1(s_i, π(s_i))|`. So instead of a fixed weight, define

  `λ = α / ( (1/N) Σ_i |Q_1(s_i, π(s_i))| )`,

and write the actor objective as

  `π = argmax_π E_{(s,a)~D}[ λ Q(s, π(s)) - (π(s) - a)^2 ]`.

Let me check that this actually cancels the scale, using the same two tasks I just computed. Take `α = 1` for the check (I'll set its real value later). On expert HalfCheetah the actor-update critic values sit around `|Q| ≈ 1000`, so `mean|Q| ≈ 1000` and `λ = 1/1000`; the value term contributed to the loss is `λ · Q ≈ (1/1000) · 1000 = 1`. On low-return Hopper, `|Q| ≈ 50`, so `λ = 1/50` and `λ · Q ≈ (1/50) · 50 = 1`. Both land at `1` — the same value — even though the raw `Q`'s differed by 20×. That's the cancellation working: `λ · Q ≈ α · (|Q| / mean|Q|)`, and the ratio `|Q|/mean|Q|` is `O(1)` by construction on every task, so the value term is pinned near `α` regardless of whether `Q` lives at `50` or `1000`. The BC term is, as computed above, already `O(0.1)`–`O(4)`. So `α` directly sets the dimensionless ratio between exploitation and imitation, and the *same* `α` carries the same meaning across every task — I've collapsed a per-task `λ` into a single scale-free `α`. That is what the normalizer buys: not a "bigger" or "smaller" `λ`, but one number that transfers across datasets so I don't tune blind.

I have to be careful with one detail in `λ`, and it's a gradient-flow trap. `λ` contains `Q`, and I'm about to backprop through `λ Q(s, π(s))` to update the actor. Let me actually differentiate the term both ways and see how much they differ. Write `m = mean|Q|` for the normalizer's denominator and treat the simplest case where all batch `Q` are positive, so `m = (1/N) Σ_j Q_j`. The term I want to maximize is `α · (1/N) Σ_i Q_i / m`. If I *detach* `m`, the gradient w.r.t. a policy parameter `φ` is `(α/m) · (1/N) Σ_i ∂Q_i/∂φ` — just the normalized value gradient, which is what I intend. If I instead let `m` carry gradient, the quotient rule adds a second piece: `∂/∂φ [ (1/N)Σ_i Q_i / m ] = (1/m)(1/N)Σ_i ∂Q_i/∂φ − ( (1/N)Σ_i Q_i / m² ) · ∂m/∂φ`. The first piece is the intended one; the second is spurious. And it's not negligible — its coefficient `(1/N)Σ_i Q_i / m² = m / m² = 1/m`, the same order as the intended `1/m` coefficient, and it points along `∂m/∂φ`, i.e. it would push the policy to inflate or deflate the *normalizer* `mean|Q|` itself rather than to climb value. That's a real gradient component of comparable magnitude pointing in a meaningless direction. So `λ` must be detached — computed as a number from the current batch and held constant for the backward pass — and then the spurious second term vanishes by construction, leaving exactly the normalized value gradient. Only `Q(s, π(s))` (in the value term) and `(π(s) - a)^2` (the BC term) carry gradient. Concretely the actor uses `Q1` (one of the twin critics) for its gradient, the mean absolute `Q` for the detached normalizer.

There's a second consequence of this normalization that I didn't design for, and I want to check whether it helps or hurts before I call it a benefit. The deterministic policy gradient through the value term is `λ · ∇_a Q(s,a) · ∇_φ π`. Suppose I rescale a task's rewards by a constant `k` — physically the same problem, just different units. Then `Q ↦ k·Q`, so `∇_a Q ↦ k·∇_a Q`, and the normalizer `mean|Q| ↦ k·mean|Q|`, hence `λ = α/mean|Q| ↦ λ/k`. Multiply them: the value-term gradient `λ·∇_a Q ↦ (λ/k)·(k·∇_a Q) = λ·∇_a Q` — unchanged. So the actor's update from the value term is invariant to reward rescaling. Contrast the unnormalized version `∇_a Q`, which would scale by `k` and effectively multiply the actor's learning rate by `k`. So without the normalizer the effective step size of the value term rides the reward scale, and I'd be implicitly re-tuning the actor's learning rate on every task; with it, the step size is fixed. Same scalar, two problems solved: the RL/BC balance and the effective learning rate are both invariant to reward scale. I didn't put the normalizer there for the second reason, but it falls out of the same algebra, which makes me more confident the normalizer is doing something principled rather than papering over a tuning artifact.

Now the second change, and I want to derive why it's worth it rather than bolt it on. The dataset is *fixed*. That's usually a curse, but here it's a gift I'm not using: because the data never changes, I can compute exact statistics over it once, up front, for free. Raw observation features in these continuous-control tasks are on very different scales — a joint angle, an angular velocity, a contact force all live in different numeric ranges. An MLP fed raw features has to learn to undo those scale disparities itself, and on a fixed dataset that's wasted capacity and a source of instability. So normalize each state feature to zero mean and unit variance using the dataset statistics:

  `s_i ← (s_i - μ_i) / (σ_i + ε)`,

with `μ_i, σ_i` the per-feature mean and standard deviation over `D`, and a small `ε = 10^{-3}` in the denominator to keep a near-constant feature (tiny `σ_i`) from exploding. I apply this to both `s` and `s'` so the critic's target sees the same normalized inputs, and I keep `μ, σ` so that at evaluation time I can normalize live observations the same way. This is a commonly-used trick in online deep RL, but it's *particularly* apt offline: online the running statistics drift as the data distribution shifts, but offline they're exact and stationary, so normalization is clean and unambiguous. It's not the core of the method, but it's a one-time, near-free stabilizer, so it earns its place.

Let me sanity-check that I haven't quietly changed anything else about TD3, because the entire pitch is *minimal* — the value of this approach is that performance is attributable, and that only holds if everything below the actor objective is untouched TD3. The critic still fits twin `Q_{θ1}, Q_{θ2}` by MSE to the clipped-double-Q target `y = r + γ(1-d)·min_i Q_{θ'_i}(s', ã)`, with target policy smoothing `ã = clip(π_{φ'}(s') + clip(N(0,σ), -c, c), -a_max, a_max)`, `σ = 0.2·a_max`, `c = 0.5·a_max`. The actor and the soft target updates (`τ = 5·10^{-3}`) still fire only every `policy_freq = 2` critic steps — the delayed updates that let the value settle relative to the policy. Discount `γ = 0.99`, Adam at `3·10^{-4}` for both, batch 256, `256×256` ReLU MLPs, twin critics, tanh-squashed deterministic actor. The *only* algorithmic change is the `λ Q - (π-a)^2` actor objective; the *only* implementation change is the state normalization. No generative model, no entropy term to remove, no architecture surgery, no actor pre-training, no reward bonus, no extra forward passes, no second hyperparameter. The BC term reuses the exact `(s, a)` minibatch already drawn for the critic, so its added compute is one elementwise difference and a mean — effectively free.

The single hyperparameter left is `α`. I want to reason about where it should sit, and the cleanest way to bound it is to take both limits exactly. Set `α = 0`: then `λ = 0`, the loss is `−0·Q.mean() + mse(π, a) = mse(π, a)`, and the actor gradient is precisely `∇_φ mse(π_φ(s), a)` — the value term has vanished and what's left is *exactly* the behavior-cloning objective from the start. So `α → 0` doesn't merely "tend toward" imitation; it reduces to BC term-for-term, a known method whose ceiling is the data. That's a reassuring anchor: my method contains BC as a literal special case. Now the other limit, `α → ∞`: `λ → ∞`, the bounded BC term `≤ 4` becomes negligible against the diverging value term, and the objective is asymptotically `argmax_π Q(s, π(s))` — the unconstrained TD3 actor, i.e. the offline blow-up I started from. So the two endpoints of `α` are precisely the two failure modes (pure imitation; unconstrained value), and the useful regime is strictly interior. Where inside? Recall the value term is pinned near `α` and the BC term sits around `O(0.1)`–`O(1)` in normal training, capped at `4`. To be value-led, I want the value term a few times the typical BC term, so `α` a small single-digit number; to keep the tether from going slack I want `α` well below the regime where it swamps even the `≈ 4` worst-case BC term. `α = 2.5` sits in that band — value-term magnitude a couple of times the typical BC term, still the same order as the BC cap — and because the normalization already made `α` task-invariant, one such value is meant to hold across the whole benchmark rather than being re-found per dataset.

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
