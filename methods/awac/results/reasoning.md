Let me set the problem precisely, because it's a combination that no existing method handles. I have a fixed offline dataset D of transitions (s,a,s',r) — could be demonstrations, could be suboptimal, could be random — and I want to pre-train on it and then fine-tune with a *small* amount of online interaction, continuing to improve. Three things have to hold at once: I must reuse the offline data efficiently online (the online budget is tiny), I must not blow up when training on a static dataset, and I must keep improving once online data arrives. Let me figure out, one at a time, what each requirement forces on the algorithm, by watching the obvious candidates fail.

First, efficiency. The simplest recipe for using demonstrations is to imitate them and then fine-tune with on-policy RL (DAPG-style). Two problems: the data may not be optimal, so pure imitation is the wrong target; and on-policy fine-tuning throws away the prior data during the RL phase — it only learns from freshly collected on-policy rollouts. If I run the comparison on a controlled HalfCheetah setup (a handful of expert demos plus some behavior-cloned suboptimal trajectories), the on-policy methods and the Monte-Carlo / TD(λ)-return methods (AWR, MARWIL) come out roughly an order of magnitude slower than an off-policy actor-critic that bootstraps Q via Bellman backups and reuses the whole buffer. So requirement one forces an off-policy critic: estimate Q^π by bootstrapping, reuse all the data.

So reach for the strongest off-policy actor-critic I have — SAC — and just drop the offline data into its replay buffer. Watch what happens. SAC-with-the-prior-data performs essentially the same as SAC-from-scratch with no prior data at all. The algorithm is *not* extracting anything from the offline data. Worse, if I BC-pretrain SAC's policy first, performance *drops* at the start of RL and then crawls back to scratch-like learning. Why? The improvement step maximizes E_{a~π}[Q(s,a)] and the critic target bootstraps at a' ~ π: y = r + γ Q_{φ̄}(s', a'). When a' is an action the dataset never contains, Q_{φ̄}(s',a') is an unconstrained extrapolation — and on a static dataset there's no fresh data to correct it. The error backs up through the Bellman recursion, and because the actor maximizes the estimate, the policy walks straight toward whichever out-of-distribution action got over-valued. The error feeds itself. So requirement two — offline stability — forces a constraint that keeps the actor's actions inside the data distribution.

That's exactly what the offline-RL methods do: BCQ, BEAR, BRAC, ABM add a constraint D(π_θ, π_β) ≤ ε to the improvement step. And offline they work — their offline starting points are far better than SAC's. But now check requirement three, online improvement, and a new failure appears. Fine-tune one of these (say BEAR) with online data and the curve is nearly flat — it barely improves. Stare at *how* the constraint is implemented: every one of these methods fits an explicit parametric model π̂_β of the behavior policy by maximum likelihood, and constrains π_θ toward that model (as a penalty, or by sampling candidate actions from it). Offline, π̂_β is fit once on a fixed dataset — fine. Online, the data is a *streaming, growing, multi-modal* mixture of the offline data and everything the improving policy collects, and the behavior model has to track it. Streaming density estimation of a shifting multi-modal distribution is genuinely hard; if I plot the behavior model's log-likelihood on the data during fine-tuning it *degrades*. And a constraint that pins the policy to an *inaccurate* π̂_β is worse than useless — it's over-conservative, dragging the policy toward a bad estimate of "the data," which kills online improvement.

So now I can state precisely what I need: an off-policy critic (for efficiency), with a policy constraint (for offline stability), but a constraint that requires **no explicit behavior model** (so online fine-tuning isn't strangled by a model it can't keep accurate). The first two I know how to get. The third is the real design problem — I need the constraint without ever fitting π̂_β.

Let me write the constrained improvement problem and just solve it exactly, and see whether the behavior model can be made to disappear. The advantage A^{π_k}(s,a) = Q^{π_k}(s,a) − V^{π_k}(s) is what I want to push up (maximizing E_π[Q] equals maximizing E_π[A] since V doesn't depend on the action), so at iteration k:

  π_{k+1} = argmax_π E_{a~π(·|s)}[ A^{π_k}(s,a) ]   s.t.   KL(π(·|s) ‖ π_β(·|s)) ≤ ε,   ∫_a π(a|s) da = 1.

Solve it with KKT. The Lagrangian, with multiplier λ on the KL constraint and α on the normalization:

  L(π, λ, α) = E_{a~π}[A(s,a)] + λ( ε − KL(π ‖ π_β) ) + α( 1 − ∫_a π(a|s) da ).

Write the KL out as KL = ∫_a π(log π − log π_β) and differentiate the whole thing with respect to the value π(a|s) at a single action. The derivative of E_π[A] = ∫ π A is A(s,a). The derivative of −λ·KL = −λ∫π(log π − log π_β) is −λ(log π − log π_β + 1) — the "+1" from differentiating the π·log π term. The derivative of −α∫π is −α. Setting the sum to zero:

  A(s,a) − λ( log π(a|s) − log π_β(a|s) + 1 ) − α = 0.

Solve for log π: λ log π = A + λ log π_β − λ − α, i.e.

  log π(a|s) = (1/λ) A(s,a) + log π_β(a|s) − 1 − α/λ.

Exponentiate, and fold the action-independent constants −1 − α/λ into a per-state normalizer 1/Z(s):

  π*(a|s) = (1/Z(s)) · π_β(a|s) · exp( (1/λ) A^{π_k}(s,a) ).

So the optimal constrained policy is the behavior policy reweighted by the exponentiated advantage. λ, the Lagrange multiplier on the KL, is the temperature: small λ sharpens toward the highest-advantage actions (aggressive improvement), large λ flattens toward π_β (cautious, BC-like). Still has π_β in it, though — I haven't beaten the behavior model yet. The projection step is where it dies.

I have to project this non-parametric π* onto my parametric policy π_θ. Project by KL — but which direction? This is the decision that makes or breaks everything, so let me try both. Minimize the *forward* KL, KL(π* ‖ π_θ), averaged over the data states ρ_{π_β}(s):

  argmin_θ E_{ρ}[ KL(π* ‖ π_θ) ] = argmin_θ E_{ρ} E_{a~π*}[ log π* − log π_θ ] = argmin_θ E_{ρ} E_{a~π*}[ −log π_θ(a|s) ],

since only the −log π_θ term depends on θ. Now the move: I need an expectation over a ~ π*, but I can't sample π* directly — except π* is just π_β reweighted, so importance-sample from the buffer β instead:

  E_{a~π*}[ −log π_θ ] = E_{a~π_β}[ (π*/π_β)·(−log π_θ) ] = E_{a~π_β}[ (1/Z(s)) exp(A/λ) · (−log π_θ) ].

The π_β factor *cancels* — π*/π_β = (1/Z) exp(A/λ), no π_β left. So the actor update becomes, maximizing the log-likelihood,

  θ_{k+1} = argmax_θ E_{(s,a)~β}[ log π_θ(a|s) · exp( (1/λ) A^{π_k}(s,a) ) ].

This is a weighted maximum-likelihood — supervised learning — on samples drawn straight from the buffer, with each observed (s,a) weighted by its exponentiated advantage. No behavior model anywhere. The constraint is enforced *implicitly*: by reweighting the buffer's own actions, the update can never put mass on an action the data didn't contain, yet it concentrates that mass on the high-advantage actions. That's the third requirement met.

Now contrast the *reverse* KL, KL(π_θ ‖ π*), to be sure forward was the right call. Reverse KL = E_{a~π_θ}[log π_θ − log π*] = E_{a~π_θ}[log π_θ − log π_β − A/λ + log Z]. This needs two things I'm trying to avoid: it evaluates log π_β — a density model — and it samples actions from π_θ, which when offline are exactly the possibly-out-of-distribution actions that make Q extrapolate. So reverse KL drags both the behavior model and the OOD-Q-query back into the loop, the two things that broke the prior methods. Forward KL needs neither. (And the two aren't unrelated: for a discrete policy bounded below by α_θ, Pinsker gives KL(π*‖π_θ) ≤ (2/α_θ)D_TV² ≤ (1/α_θ)KL(π_θ‖π*), so minimizing the reverse KL also bounds the forward one — they're loosely interchangeable in objective, but only the forward direction lets me sample from the buffer and cancel π_β.)

There's the per-state normalizer Z(s) = ∫_a π_θ(a|s) exp(A/λ) da = E_{a~π_θ}[exp(A/λ)] sitting in the weight. Do I need it? Try to keep it and you have to estimate that expectation — say K=10 samples per batch element — and empirically that makes things *worse*: pen falls from 98% to 84%, door from 95% to 0%, relocate from 54% to 0% when I weight by an estimated Z(s). The estimation error hurts more than the normalization helps. There's also a clean argument for why dropping it is benign: Z(s) is a per-*state* factor, so it only reweights how much different *states* count in the objective, not how different *actions* compete within a state. The buffer's state distribution already differs from the one π_θ will visit, so faithfully preserving it is low-value, while a noisy Z(s) estimate just injects variance like a degenerate importance weight. (If I wanted bounds, Cauchy-Schwarz gives Z(s) ≤ √(∫π² ∫g²) with g = exp(A/λ), and the reverse Cauchy-Schwarz of Polya-Szego gives a matching lower bound, so Z(s) ∈ [C2, C1] — but the bounds are loose and not worth the trouble.) So drop Z(s) and instead just normalize the weights across the minibatch.

That leaves the critic, the policy-evaluation half. Here I want efficiency, so off-policy temporal-difference bootstrapping of the *current* policy's Q^π — not Monte-Carlo, not the behavior policy's value. This is the second key difference from the advantage-weighted-regression methods: AWR estimates V^{π_β}, the *behavior* policy's value, by Monte-Carlo or TD(λ), which is slow and only supports a single step of improvement away from π_β; I bootstrap Q^π of the *current* policy, which reuses off-policy data and improves iteratively. I'll build the critic on twin Q-functions with a min target and a Polyak target network to control the overestimation that bootstrapping induces (TD3/SAC machinery):

  L(φ) = E_D[ (Q_φ(s,a) − y)² ],   y = r + γ E_{s', a'~π}[ Q_{φ̄}(s', a') ],

with two critics and y using the minimum of the twin targets. The advantage the actor needs is then A(s,a) = Q(s,a) − V(s), and since V(s) = E_{a~π}[Q(s,a)] I just estimate it by evaluating the critic at an action sampled from the current policy (taking the min over the twin to stay consistent). The whole thing is a standard actor-critic — evaluate Q^π by TD, improve π by the advantage-weighted MLE — alternated each step, with a Polyak target update.

The decisive structural point: the actor's improvement uses *only buffer actions* reweighted by advantage, so it never queries Q at a policy-proposed (possibly OOD) action during *improvement*, and it never needs a behavior model. That's why this stays stable offline (the implicit constraint) and yet fine-tunes freely online (no behavior model to keep accurate). Pre-train by running this update on the offline buffer for some steps; then go online, adding collected transitions to the same buffer and running the identical update — nothing about the algorithm changes between phases.

Let me write it as real code. Twin critic, a Gaussian policy, the TD critic loss, the advantage-weighted actor loss, and the alternating update:

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__(); self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
    def dist(self, s):
        h = self.trunk(s)
        return Normal(self.mu(h), self.log_std(h).clamp(-6, 0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
    def sample(self, s):
        return self.dist(s).rsample()


def critic_td_loss(batch, critic, target_critic, policy, discount):
    s, a, r, s2, done = (batch["obs"], batch["act"], batch["rew"],
                         batch["obs2"], batch["done"])
    with torch.no_grad():
        a2 = policy.sample(s2)                              # next action from CURRENT policy (off-policy TD)
        tq1, tq2 = target_critic(s2, a2)
        y = r + discount * (1.0 - done) * torch.min(tq1, tq2)   # min of twin targets
    q1, q2 = critic(s, a)
    return F.mse_loss(q1, y) + F.mse_loss(q2, y)


def actor_awac_loss(batch, critic, policy, lam):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = critic(s, a)
        q = torch.min(q1, q2)                               # Q(s, a_data)
        v_a = policy.sample(s)                              # a ~ pi(.|s)
        v1, v2 = critic(s, v_a)
        v = torch.min(v1, v2)                               # V(s) = Q(s, a~pi)
        adv = q - v                                         # advantage A(s, a_data)
        weight = torch.exp(adv / lam)                       # exp(A / lambda); lambda is the KL multiplier
        weight = weight / (weight.mean() + 1e-8)            # normalize over the batch (drop per-state Z(s))
    logp = policy.log_prob(s, a)                            # MLE on BUFFER actions only -> implicit constraint
    return -(weight * logp).mean()


def polyak(critic, target_critic, tau):
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def update(batch, critic, target_critic, policy, opts, hp):
    q_loss = critic_td_loss(batch, critic, target_critic, policy, hp["discount"])
    opts["q"].zero_grad(); q_loss.backward(); opts["q"].step()

    pi_loss = actor_awac_loss(batch, critic, policy, hp["lam"])
    opts["pi"].zero_grad(); pi_loss.backward(); opts["pi"].step()

    polyak(critic, target_critic, hp["tau"])
```

and the offline-pretrain-then-online loop, with the same update throughout:

```python
def train_awac(env, buffer, critic, policy, hp,
               pretrain_steps=25000, online_steps=int(1e6)):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q":  torch.optim.Adam(critic.parameters(), lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-4),
    }
    for _ in range(pretrain_steps):                          # OFFLINE: buffer preloaded with the dataset
        update(buffer.sample(hp["batch_size"]), critic, target_critic, policy, opts, hp)

    o = env.reset()
    for _ in range(online_steps):                            # ONLINE: identical update, same buffer
        a = policy.sample(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
        o2, r, done, _ = env.step(a)
        buffer.add(o, a, r, o2, done)
        o = env.reset() if done else o2
        update(buffer.sample(hp["batch_size"]), critic, target_critic, policy, opts, hp)

# hp: discount=0.99, tau=5e-3, batch_size=1024,
#     lam=0.3 (manipulation) or 1.0 (MuJoCo locomotion)
```

The chain, end to end: pre-training from offline data then fine-tuning online demands one algorithm that is data-efficient, stable on a static dataset, and still improvable online. Off-policy actor-critic is efficient but, applied naively offline, bootstraps its target at out-of-distribution policy actions and accumulates error; the offline-RL fix — constrain to an explicitly-fit behavior model π̂_β — is stable offline but stalls online because the behavior model can't be fit accurately on streaming multi-modal data. Solving the KL-constrained improvement max_π E_π[A] s.t. KL(π‖π_β)≤ε exactly gives π* ∝ π_β·exp(A/λ); projecting it onto the parametric policy by *forward* KL and importance-sampling from the buffer cancels the π_β factor, yielding an advantage-weighted maximum-likelihood actor update θ ← argmax E_{(s,a)~β}[log π_θ(a|s)·exp(A/λ)] that constrains the policy *implicitly* — only buffer actions, no behavior model, never querying Q at a proposed OOD action during improvement. The per-state Z(s) is dropped (estimating it hurt; it only reweights states) in favor of batch normalization. The advantage uses an off-policy bootstrapped Q^π of the *current* policy (twin-Q, min target, Polyak) rather than a Monte-Carlo V^{π_β}, which is what makes it efficient and lets it improve past one step. The same update runs unchanged offline and online, so pre-training flows directly into fine-tuning.
