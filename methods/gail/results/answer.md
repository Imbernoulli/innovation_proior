# Generative Adversarial Imitation Learning (GAIL)

## Problem

Learn a policy from a fixed set of expert trajectories (state-action pairs)
with no reward signal, no expert queries during training, and no access to the
expert's actions in the states the learner visits. The two standard routes each
pay a price: behavioral cloning fits single-timestep decisions on the expert's
state distribution and compounds errors under covariate shift at test time;
inverse RL recovers a cost function but runs reinforcement learning in an inner
loop (slow) and yields a cost rather than the actions actually wanted. GAIL
extracts a policy directly, as if by RL-following-IRL, without the nested loop.

## Key idea

**Occupancy-measure duality.** For a policy $\pi$, the occupancy measure is
$\rho_\pi(s,a)=\sum_{t}\gamma^t P(s_t=s,a_t=a\mid\pi)$, so
$\mathbb E_\pi[c]=\sum_{s,a}\rho_\pi(s,a)c(s,a)$. Policies and occupancy measures
are in bijection over the convex set $\mathcal D$ of Bellman-flow-feasible
measures. Running maximum-causal-entropy IRL with a closed proper convex cost
regularizer $\psi$ over *all* cost functions, then running RL on the recovered
cost, is exactly

$$\mathrm{RL}\circ\mathrm{IRL}_\psi(\pi_E)=\arg\min_{\pi}\ -H(\pi)+\psi^*(\rho_\pi-\rho_{\pi_E}),$$

where $H$ is causal entropy and $\psi^*$ is the convex conjugate. IRL is the
*dual* of occupancy matching; the cost is the dual variable, and RL recovers the
primal. So imitation reduces to matching $\rho_\pi$ to $\rho_{\pi_E}$ under the
discrepancy $\psi^*$, with no explicit cost ever materialized. The choice of
$\psi$ selects the matching distance: a constant $\psi$ forces exact (but
intractable) matching; the indicator $\delta_{\mathcal C}$ of a linear cost class
gives apprenticeship learning, which cannot imitate exactly because $\mathcal C$
is a small subspace.

**The GAIL regularizer and the GAN reduction.** Choosing

$$\psi_{\mathrm{GA}}(c)=\begin{cases}\mathbb E_{\pi_E}[g(c(s,a))] & c<0\\ +\infty &\text{else}\end{cases},\qquad
g(x)=\begin{cases}-x-\log(1-e^x)& x<0\\ +\infty&\text{else}\end{cases}$$

(the surrogate-loss construction with the logistic loss $\phi(x)=\log(1+e^{-x})$)
gives

$$\psi_{\mathrm{GA}}^*(\rho_\pi-\rho_{\pi_E})=\max_{D:\mathcal S\times\mathcal A\to(0,1)}\ \mathbb E_\pi[\log D(s,a)]+\mathbb E_{\pi_E}[\log(1-D(s,a))],$$

the maximized classifier log-likelihood (negative logistic risk) between policy
and expert state-actions, equal up to a constant to the Jensen-Shannon divergence
$D_{\mathrm{JS}}(\rho_\pi,\rho_{\pi_E})$.
The imitation objective is therefore

$$\min_\pi\ D_{\mathrm{JS}}(\rho_\pi,\rho_{\pi_E})-\lambda H(\pi),$$

a divergence whose square root is a metric, so it is zero only at exact
occupancy matching. It is solved as a generative-adversarial game: a
discriminator $D$ classifies policy vs expert $(s,a)$ pairs, and the policy is
the generator that fools it.

## Algorithm

Find a saddle point $(\pi_\theta, D_w)$ of
$\mathbb E_\pi[\log D]+\mathbb E_{\pi_E}[\log(1-D)]-\lambda H(\pi)$, where
$D$ is the probability that a transition came from the current policy, by
alternating:

1. **Discriminator step.** Adam step on $w$ to increase
   $\hat{\mathbb E}_{\tau_i}[\nabla_w\log D_w(s,a)]+\hat{\mathbb E}_{\tau_E}[\nabla_w\log(1-D_w(s,a))]$
   — binary cross-entropy with policy samples labeled 1 and expert samples labeled 0.
   If the implementation instead uses an expert-probability logit $s$, this is
   the same as minimizing BCE with expert label 1 and policy label 0, with
   $D=1-\sigma(s)$.
2. **Policy step.** TRPO (KL-constrained natural-gradient) step on $\theta$ with
   cost $c(s,a)=\log D_w(s,a)$, implemented by maximizing the reward
   $-\log D_w$ so the policy moves toward expert-classified regions. With an
   expert-probability logit $s$, the direct reward is
   $-\log(1-\sigma(s))=\mathrm{softplus}(s)$. The cost-gradient to descend is
   $\hat{\mathbb E}_{\tau_i}[\nabla_\theta\log\pi_\theta(a\mid s)\,Q(s,a)]-\lambda\nabla_\theta H(\pi_\theta)$,
   $Q(\bar s,\bar a)=\hat{\mathbb E}_{\tau_i}[\log D_w(s,a)\mid s_0,a_0]$, with a value
   baseline and generalized advantage estimation for variance reduction.

The causal-entropy gradient folds into the policy gradient:
$\nabla_\theta H(\pi_\theta)=\mathbb E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(a\mid s)\,Q_{\log}]$,
$Q_{\log}(\bar s,\bar a)=\mathbb E_{\pi_\theta}[-\log\pi_\theta(a\mid s)\mid s_0,a_0]$.
The TRPO step prevents policy divergence from the noisy gradient; the
discriminator is an adaptive cost refit each iteration, so there is no nested RL
loop and no compounding-error single-step fit.

## Code

```python
import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F

class LearningSignal(nn.Module):
    """Binary transition classifier; score > 0 means expert-like."""
    def __init__(self, obs_dim, act_dim, hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))                       # logit / "score"

    def score(self, obs, act):
        return self.net(torch.cat([obs, act], dim=1)).squeeze(1)

    def reward(self, obs, act, favor_zero_expert=False):
        # The derivation's D is policy-probability. Here sigmoid(score) is
        # expert-probability, so direct reward is -log D = softplus(score).
        with torch.no_grad():
            s = self.score(obs, act)
            if favor_zero_expert:                       # alternate shaping: <= 0
                return F.logsigmoid(s)
            return F.softplus(s)                        # -log(1 - sigmoid(s))

    def fit(self, opt, pi_obs, pi_act, ex_obs, ex_act, steps=1, ent_reg=1e-3):
        obs = torch.cat([pi_obs, ex_obs]); act = torch.cat([pi_act, ex_act])
        B, Ball = len(pi_obs), len(pi_obs) + len(ex_obs)
        labels = torch.zeros(Ball, device=obs.device); labels[B:] = 1.0
        weights = torch.empty(Ball, device=obs.device)  # evenly weight both halves
        weights[:B] = 1.0 / B; weights[B:] = 1.0 / (Ball - B)
        for _ in range(steps):
            s = self.score(obs, act)
            bce = F.binary_cross_entropy_with_logits(s, labels, reduction='none')
            ent = F.softplus(-s) + (1.0 - torch.sigmoid(s)) * s  # logit-Bernoulli entropy
            loss = ((bce - ent_reg * ent) * weights).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

def imitation_loop(env, expert_obs, expert_act, policy, value_fn,
                   obs_dim, act_dim, iters=500, gamma=0.995, gae_lam=0.97,
                   lam_ent=0.0, max_kl=0.01):
    signal = LearningSignal(obs_dim, act_dim)
    signal_opt = torch.optim.Adam(signal.parameters(), lr=1e-2)
    for _ in range(iters):
        # 1. roll out current policy
        obs, act, ep_lens = sample_trajectories(env, policy)
        # 2. imitation reward from the discriminator (the adaptive cost)
        rew = signal.reward(obs, act)
        if lam_ent:                                     # causal-entropy bonus
            rew = rew + lam_ent * (-policy.log_prob(obs, act).detach())
        # 3. variance-reduced advantages, then one KL-constrained policy step
        adv = gae(rew, value_fn(obs).detach(), ep_lens, gamma, gae_lam)
        trpo_step(policy, obs, act, adv, max_kl=max_kl)
        # 4. discriminator step: subsample expert to match policy batch size
        idx = np.random.choice(len(expert_obs), size=len(obs))
        signal.fit(signal_opt, obs, act, expert_obs[idx], expert_act[idx])
        # 5. fit the value function to returns under the refit signal
        rew_for_value = signal.reward(obs, act)
        if lam_ent:
            rew_for_value = rew_for_value + lam_ent * (-policy.log_prob(obs, act).detach())
        value_fn.fit(obs, returns_from(rew_for_value, ep_lens, gamma))
    return policy
```

The policy is a Gaussian (continuous) or categorical (discrete) network with two
hidden layers of 100 $\tanh$ units; `trpo_step` is a KL-constrained
natural-gradient step (conjugate-gradient Fisher-vector products plus line
search); `gae` and `value_fn` implement generalized advantage estimation with a
fitted baseline. Optionally, behavioral cloning can warm-start the policy
parameters to save environment interaction.
