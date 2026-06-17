# Proximal Policy Optimization (with KL-adaptive learning rate)

## Problem

Improve a neural-network stochastic policy `π_θ(a|s)` by on-policy gradient ascent on expected
return, extracting many gradient steps from each expensive batch of rollouts without the
destructive policy collapse that naive multi-epoch reuse causes — using only first-order SGD, so
it scales to large nets and to architectures that share policy/value parameters, use dropout, or
carry auxiliary losses.

## Key idea

The single-step policy-gradient surrogate is only valid at the sampling point. Re-weight it by the
likelihood ratio `r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)` so it becomes an off-policy estimate that
can be reused for several epochs — then keep `π_θ` close to `π_old` so that estimate stays
faithful. PPO enforces "close" with a **clipped surrogate** that removes any incentive to push the
ratio outside `[1−ε, 1+ε]`:

```
L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ],   ε = 0.2.
```

The `min` makes `L^CLIP` a pessimistic lower bound on the unclipped surrogate `E_t[r_t Â_t]`.
For `Â_t > 0`, the unclipped term is active below `1+ε` and the objective is flat above `1+ε`;
lowering the probability of a good action remains fully penalized. For `Â_t < 0`, the clipped term
is active and flat below `1−ε`, so driving a bad action's probability even lower gives no extra
credit; above `1+ε`, the unclipped term is the smaller one, so raising a bad action's probability
keeps the full penalty. This is the in-objective trust region — no second-order Fisher machinery,
no differentiated KL term needed.

The same trust-region target can also be pursued with an **adaptive KL penalty**: optimize
`E_t[r_t Â_t − β KL[π_old, π_θ]]` and, after each update, measure `d = E_t[KL]` and adjust the
coefficient toward a target `d_targ` — `β←β/2` if `d < d_targ/1.5`, `β←β×2` if `d > d_targ·1.5`.
The factors are heuristic; the loop is self-correcting, so the initial `β` is unimportant.

**This baseline uses the learning-rate form of that controller** (the Roboschool-style variant where
the Adam stepsize is adjusted by the target KL, as in the canonical `rsl_rl`/legged-gym implementation):
with the clipped surrogate already in place, the realized KL of each update drives the **Adam
learning rate** toward the target trust-region size. Because larger `lr` produces larger KL, the
update direction flips relative to `β`:

```
d = E_t[ KL[π_old, π_θ] ]                         # closed-form Gaussian KL
if d > 2.0 · d_targ:        lr ← max(1e-5, lr / 1.5)   # moved too far
if 0 < d < d_targ / 2.0:    lr ← min(1e-2, lr · 1.5)   # room to move
```

The penalty-coefficient controller uses a 1.5 band; this learning-rate controller uses the
implementation's `2.0`/`0.5` thresholds with `/1.5` and `*1.5` learning-rate steps, target
`d_targ ≈ 0.01`, and `lr` clamped to `[1e-5, 1e-2]`. For a diagonal-Gaussian policy the exact KL
is analytic:

```
KL = Σ_i [ log(σ_new,i/σ_old,i) + (σ_old,i² + (μ_old,i − μ_new,i)²) / (2 σ_new,i²) − 1/2 ].
```

The code minimizes the negative objective, so the mathematical `min` becomes `torch.max` over
`-Â_t * ratio` and `-Â_t * clip(ratio, 1−ε, 1+ε)`. The small `1e-5` inside the log ratio below is
the implementation's numerical guard; the formula above is the exact Gaussian KL.

The advantages `Â_t` come from truncated GAE, `Â_t = Σ_l (γλ)^l δ_{t+l}`,
`δ_t = reward_t + γV(s_{t+1}) − V(s_t)`. The full per-iteration objective adds a (clip-mirrored)
value loss and an entropy bonus: `L = L^CLIP − c1 L^VF + c2 S[π_θ]`. Outer loop: each iteration,
`N` parallel actors collect `T` steps, compute GAE, then `K` epochs of minibatch Adam on the
combined loss; then `π_old ← π_θ`.

## Algorithm (actor-critic style)

```
for iteration = 1, 2, ...:
    for actor = 1..N: run π_old for T steps; compute Â_1..Â_T (truncated GAE)
    for epoch = 1..K, over minibatches of the NT transitions:
        measure KL(π_old || π_θ); adapt lr toward d_targ (clamped)
        ascend  L^CLIP − c1 (V_θ − R)^2  + c2 entropy   via Adam (grad-norm clipped)
    π_old ← π_θ
```

## Code (faithful to the canonical implementation)

```python
import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    """PPO with a KL-adaptive learning rate (rsl_rl / legged-gym style)."""

    def __init__(self, actor_critic, num_learning_epochs=5, num_mini_batches=4,
                 clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0,
                 entropy_coef=0.01, learning_rate=1e-3, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
                 device="cpu"):
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma, self.lam = gamma, lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def compute_returns(self, last_values):
        st = self.storage
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 \
                else st.values[step + 1]
            not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + not_terminal * self.gamma * self.lam * advantage     # GAE
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self):
        mean_value_loss = mean_surrogate_loss = 0.0
        for (obs_batch, critic_obs_batch, actions_batch, old_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch) in self.storage.mini_batch_generator(
                 self.num_mini_batches, self.num_learning_epochs):

            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL-adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch)
                           + torch.square(old_mu_batch - mu_batch))
                          / (2.0 * torch.square(sigma_batch)) - 0.5, dim=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate

            # Clipped surrogate
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Clipped value loss
            if self.use_clipped_value_loss:
                value_clipped = old_values_batch + (value_batch - old_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_loss = torch.max((value_batch - returns_batch).pow(2),
                                       (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return mean_value_loss / n, mean_surrogate_loss / n
```

## Hyperparameters

MuJoCo 1M-step benchmark: horizon `T=2048`, Adam `3e-4`, `K=10` epochs, minibatch 64, `γ=0.99`,
`λ=0.95`, `ε=0.2`. Roboschool 3D-humanoid: `T=512`, `K=15`, minibatch 4096, 32 actors (128 for
the harder steering task), log-std annealed, Adam stepsize adjusted by the target KL. Typical
continuous-control controller settings: `desired_kl=0.01`, `lr` clamped to `[1e-5, 1e-2]`, and
the implementation thresholds `kl > 2.0 * desired_kl` and `0 < kl < desired_kl / 2.0`.
