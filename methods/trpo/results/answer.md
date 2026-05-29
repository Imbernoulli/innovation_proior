# Trust Region Policy Optimization (TRPO)

## Problem

Train large, nonlinear stochastic policies (neural networks) so that the true
expected return improves *monotonically and robustly* each update, with little
hyperparameter tuning. Plain policy gradients are brittle because they step in
Euclidean parameter space, where step size has no consistent meaning for the policy
distribution; a single large step can collapse performance.

## Key idea

Measure and bound how far the policy moves in *distribution* space (KL divergence)
per update, not in parameter space. There is an exact identity for the return of a
new policy `π̃` relative to the current `π`:

```
η(π̃) = η(π) + E_{τ∼π̃}[ Σ_t γ^t A_π(s_t, a_t) ].
```

Replacing the new policy's state visitation by the old one's gives a tractable,
first-order-accurate surrogate `L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)`. A
coupling argument bounds the gap and yields a guaranteed-improvement lower bound

```
η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃),    C = 4εγ/(1−γ)^2,  ε = max_{s,a}|A_π(s,a)|,
```

so maximizing the right-hand side gives monotonic improvement (a minorize-maximize
scheme). The theoretical penalty `C` forces microscopic steps, so it is replaced by
a *trust-region constraint* with an interpretable radius `δ`, and the per-state max
KL is softened to the average KL over visited states.

## Final algorithm

At each iteration, solve the sampled problem below, using `ρ_old` as the normalized
discounted visitation distribution; the omitted `1/(1−γ)` factor is positive and
does not change the KL-normalized step.

```
maximize_θ   E_{s∼ρ_old, a∼π_old}[ (π_θ(a|s)/π_old(a|s)) · A_old(s,a) ]
subject to   E_{s∼ρ_old}[ D_KL(π_old(·|s) ‖ π_θ(·|s)) ] ≤ δ.
```

Solve it with local models around `θ_old`: linearize the objective, `L ≈ g^T(θ−θ_old)`
with `g` the policy gradient; quadraticize the constraint, `D̄_KL ≈ ½(θ−θ_old)^T A (θ−θ_old)`
with `A` the average Fisher information matrix. The Lagrangian solution is the natural
gradient direction `x = A^{-1} g`, with step length that saturates the KL budget:

```
β = sqrt( 2δ / (x^T A x) ),   θ = θ_old + β x.
```

Scalability and robustness:
- **Conjugate gradient** solves `A x = g` using only Fisher-vector products, never
  forming `A`. The Fisher-vector product is `A v = ∇_θ((∇_θ D̄_KL)^T v)` (a
  Hessian-vector product), about one gradient in cost; subsampling for `A` makes it
  cheaper, and a small damping `A → A + ηI` keeps CG stable.
- **Backtracking line search**: the linear/quadratic models are only local, so try
  `θ_old + α^j β x` (`j = 0,1,2,…`, `0<α<1`) and accept the first step that improves
  the *true* surrogate and satisfies the *true* `D̄_KL ≤ δ`; otherwise don't move.

Special cases of the same update: natural policy gradient (fixed step instead of
enforcing `δ`), vanilla policy gradient (Euclidean trust region instead of KL),
policy iteration (no constraint, fully maximize `L`).

Policy parameterization: diagonal Gaussian with network mean and state-independent
log-std for continuous control; softmax categorical for discrete actions. Advantages
use a learned value-function baseline fit by regression to returns.

## Working code

A compact PyTorch implementation uses the same pieces as the standard TRPO training
loop: GAE, the ratio·advantage surrogate, analytic KL, conjugate gradient with
Hessian-vector products, a step that saturates `δ`, and backtracking line search.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    """State -> diagonal Gaussian over actions; state-independent log-std."""
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, act_dim)]
        self.mu_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def dist(self, obs):
        return Normal(self.mu_net(obs), torch.exp(self.log_std))

    def logp(self, obs, act):
        return self.dist(obs).log_prob(act).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, 1)]
        self.v = nn.Sequential(*layers)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def set_flat_params(model, flat_params):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat_params[i:i + n].view_as(p)); i += n


def discount_cumsum(x, discount):
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def gae(rewards, values, last_val, gamma=0.99, lam=0.97):
    """Generalized Advantage Estimation; returns advantages and value targets."""
    rews = np.append(rewards, last_val)
    vals = np.append(values, last_val)
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    adv = discount_cumsum(deltas, gamma * lam)
    ret = discount_cumsum(rews, gamma)[:-1]
    return adv, ret


def surrogate(policy, obs, act, adv, logp_old):
    ratio = torch.exp(policy.logp(obs, act) - logp_old)
    return (ratio * adv).mean()


def mean_kl(policy, obs, mu_old, std_old):
    d, d_old = policy.dist(obs), Normal(mu_old, std_old)
    return torch.distributions.kl_divergence(d_old, d).sum(-1).mean()


def fisher_vector_product(policy, obs, mu_old, std_old, v, damping=0.1):
    kl = mean_kl(policy, obs, mu_old, std_old)
    g = flat(torch.autograd.grad(kl, policy.parameters(), create_graph=True))
    gv = (g * v).sum()
    hv = flat(torch.autograd.grad(gv, policy.parameters(), retain_graph=True))
    return hv + damping * v


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone(); p = b.clone(); r_dot = torch.dot(r, r)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = r_dot / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p; r -= alpha * Ap
        r_dot_new = torch.dot(r, r)
        if r_dot_new < tol:
            break
        p = r + (r_dot_new / r_dot) * p; r_dot = r_dot_new
    return x


def trpo_step(policy, obs, act, adv, logp_old, mu_old, std_old,
              delta=0.01, backtrack_coeff=0.8, backtrack_iters=10):
    L_old = surrogate(policy, obs, act, adv, logp_old)
    L_old_value = L_old.detach()
    g = flat(torch.autograd.grad(L_old, policy.parameters(), retain_graph=True))

    Avp = lambda v: fisher_vector_product(policy, obs, mu_old, std_old, v)
    x = conjugate_gradient(Avp, g)                       # x = A^{-1} g

    xAx = torch.dot(x, Avp(x))
    beta = torch.sqrt(2 * delta / (xAx + 1e-8))          # saturate KL budget
    full_step = beta * x

    old_params = flat([p.data for p in policy.parameters()])
    for j in range(backtrack_iters):                     # line search on true L, true KL
        set_flat_params(policy, old_params + (backtrack_coeff ** j) * full_step)
        with torch.no_grad():
            kl = mean_kl(policy, obs, mu_old, std_old)
            L_new = surrogate(policy, obs, act, adv, logp_old)
        if kl <= delta and L_new > L_old_value:
            return
    set_flat_params(policy, old_params)                  # reject: keep old policy


def fit_value(value_net, obs, returns, opt, iters=80):
    for _ in range(iters):
        opt.zero_grad()
        (((value_net(obs) - returns) ** 2).mean()).backward()
        opt.step()


def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out


def train(env, policy, value_net, epochs=50, steps=4000, gamma=0.99, lam=0.97):
    v_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        O, Ac, Rew, LogpOld, MuOld = [], [], [], [], []
        o, ep_rew, ep_val = reset_env(env), [], []
        for t in range(steps):
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                d = policy.dist(ot); a = d.sample()
                logp = d.log_prob(a).sum(-1); val = value_net(ot)
            o2, r, done, _ = step_env(env, a.numpy())
            O.append(o); Ac.append(a.numpy()); ep_rew.append(r); ep_val.append(val.item())
            LogpOld.append(logp.item()); MuOld.append(d.mean.numpy()); o = o2
            if done or t == steps - 1:
                with torch.no_grad():
                    last_val = 0.0 if done else value_net(torch.as_tensor(o, dtype=torch.float32)).item()
                adv, ret = gae(np.array(ep_rew), np.array(ep_val), last_val, gamma, lam)
                Rew.append((adv, ret)); o, ep_rew, ep_val = reset_env(env), [], []
        adv = np.concatenate([a for a, _ in Rew]); ret = np.concatenate([r for _, r in Rew])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(np.array(O), dtype=torch.float32)
        act = torch.as_tensor(np.array(Ac), dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        logp_old = torch.as_tensor(np.array(LogpOld), dtype=torch.float32)
        mu_old = torch.as_tensor(np.array(MuOld), dtype=torch.float32)
        std_old = torch.exp(policy.log_std).detach().expand_as(mu_old)

        trpo_step(policy, obs, act, adv_t, logp_old, mu_old, std_old, delta=0.01)
        fit_value(value_net, obs, ret_t, v_opt, iters=80)
```
