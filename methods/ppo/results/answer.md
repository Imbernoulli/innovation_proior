# Proximal Policy Optimization (PPO)

PPO is a first-order policy-gradient method that recovers the reliability and data efficiency of a trust-region update while needing nothing beyond ordinary minibatch SGD/Adam and automatic differentiation. It replaces an explicit KL trust region (and its second-order conjugate-gradient / Fisher-vector machinery) with a clipped surrogate objective that is just a term in a differentiable loss, so the same batch of on-policy data can be reused for several epochs of minibatch updates without the policy collapsing.

## The problem it solves

Maximize expected discounted return η(π_θ) = E[Σ_t γ^t R_t] of a stochastic policy. Vanilla policy gradients use each sample for one update and are fragile to step size; running multiple SGD epochs on the same batch would be efficient but the policy-gradient surrogate is only valid while π_θ stays near the data-generating π_old, so naive multi-epoch ascent diverges. Trust-region methods (TRPO) control that drift with a hard KL constraint solved by conjugate gradient on Fisher-vector products plus a line search, but the machinery is heavy, supports only one step per batch (no cheap minibatch reuse), and conflicts with parameter sharing and stochastic network components. PPO wants the trust-region behavior with a first-order loss.

## Key idea

Let r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) be the probability ratio (r = 1 at the start of each update). The conservative-policy-iteration surrogate is L^{CPI}(θ) = Ê_t[ r_t(θ) Â_t ]; maximizing it without a leash drives r_t arbitrarily far — the optimizer cheaply inflates r on positive-advantage samples — and blows up. PPO clips the ratio to [1−ε, 1+ε] (ε = 0.2) and takes the pessimistic minimum of the clipped and unclipped terms:

    L^{CLIP}(θ) = Ê_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ].

The min makes L^{CLIP} a lower bound on L^{CPI}. Case analysis:
- **Â > 0** (good action): for r > 1+ε the term flattens at (1+ε)Â — gradient 0, no reward for pushing the probability higher; for r < 1−ε (a wrong-direction move) the min keeps the unclipped rÂ, so the gradient still pulls the probability back up.
- **Â < 0** (bad action): for r < 1−ε the term flattens at (1−ε)Â — no reward for suppressing further; but for r > 1+ε (a bad-action overshoot) the min keeps the more-negative unclipped term, so an overshoot stays penalized and is corrected.

So clipping removes the incentive to push r past the band *in the direction the advantage favors*, but never discards the gradient that corrects a move in the wrong direction — a *plain* clip (without the min) would freeze an overshoot in the flat region. To first order at θ_old it equals L^{CPI}, so the first epoch is ordinary policy-gradient ascent; the brake engages only as the policy moves. This is a trust region expressed entirely as a flat spot in the loss — no KL term, no Fisher matrix, no line search. ε ≈ 0.2 is large enough to make real progress over K epochs yet small enough that the realized per-update KL stays ~0.01–0.02, where the surrogate bound is tight; because it constrains the unit-free ratio directly, one value is robust across tasks where a fixed KL-penalty coefficient β is not.

## Full objective and algorithm

Advantages use truncated generalized advantage estimation with δ_t = R_t + γ V(s_{t+1}) − V(s_t) and Â_t = Σ_l (γλ)^l δ_{t+l} (λ ≈ 0.95, γ = 0.99); λ trades bias for variance (λ→0: low-variance, biased δ_t; λ→1: unbiased Monte-Carlo advantage), computed as a single reverse scan with a (1−done) mask across episode boundaries. The value V(s) is the baseline that makes the advantage low-variance. When the policy and value share a network, the value error and an entropy bonus are folded into one objective maximized each iteration:

    L^{CLIP+VF+S}(θ) = Ê_t[ L^{CLIP}_t(θ) − c_1 (V_θ(s_t) − V_t^{targ})² + c_2 S[π_θ](s_t) ],

with value target V_t^{targ} = Â_t + V_old(s_t). The value loss is commonly clipped by symmetry with the policy clip (cap how far V_θ moves from V_old per update). The entropy bonus c_2 S[π_θ] sustains exploration (used on Atari; often c_2 = 0 on MuJoCo, where the Gaussian's learned log-std handles exploration).

Each iteration: N actors collect T steps each (NT samples) with π_{θ_old}; compute Â_t; run K epochs of shuffled minibatch SGD/Adam on L^{CLIP+VF+S}; set θ_old ← θ and repeat. The clip is what makes the K-epoch reuse safe.

An alternative variant replaces clipping with an adaptive KL penalty L^{KLPEN} = Ê_t[ r_t Â_t − β KL[π_old, π_θ] ], doubling β when the realized KL exceeds d_targ·1.5 and halving it below d_targ/1.5 — servoing β onto a KL target so the coefficient no longer has to be guessed. Clipping is the default: simpler (no KL in the loss) and at least as good.

Standard MuJoCo settings: T = 2048, Adam 3e-4 (eps 1e-5), K = 10 epochs, minibatch 64, γ = 0.99, λ = 0.95, ε = 0.2. Implementation hygiene that matters: orthogonal init (gain √2 hidden, 0.01 policy head, 1.0 value head), linear LR annealing to 0, per-minibatch advantage normalization, global gradient-norm clip at 0.5, state-independent policy log-std.

## Working code

A faithful single-file continuous-control implementation. The discrete variant swaps the Gaussian head for a `Categorical` over logits and is otherwise identical.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))  # state-independent

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


agent = Agent(envs)
optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

# ---- per iteration: rollout fills obs/actions/logprobs/rewards/dones/values for num_steps x num_envs ----

frac = 1.0 - (iteration - 1.0) / num_iterations
optimizer.param_groups[0]["lr"] = frac * 3e-4

# truncated GAE: A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values

# flatten the batch
b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs = logprobs.reshape(-1)
b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values.reshape(-1)

# K epochs of minibatch updates on the same batch
b_inds = np.arange(batch_size)
for epoch in range(update_epochs):
    np.random.shuffle(b_inds)
    for start in range(0, batch_size, minibatch_size):
        mb_inds = b_inds[start:start + minibatch_size]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()                      # r_t(theta) = pi_theta / pi_old

        mb_advantages = b_advantages[mb_inds]
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # clipped surrogate (minimize the negative => max of two negatives)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # clipped value loss (symmetry with the policy clip)
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds], -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # global norm cap
        optimizer.step()
```

The complete trust region is the three lines computing `pg_loss1`, `pg_loss2`, and `torch.max(...)`. Everything else — GAE, advantage normalization, the value loss, the entropy bonus, gradient clipping, learning-rate annealing, orthogonal init — is standard actor-critic plumbing and hygiene, and the K-epoch minibatch loop over a parallel-actor batch is the data reuse the clipped objective makes safe.
