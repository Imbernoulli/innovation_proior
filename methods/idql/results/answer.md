# IDQL: Implicit Diffusion Q-Learning

## Problem

Offline RL must learn a policy from a fixed dataset `D = {(s,a,r,s')}` from a behavior policy
`mu`, without querying value functions at out-of-distribution actions (which causes
overestimation and divergence). Implicit Q-Learning (IQL) solves the *value* side in-sample, but
(a) it is unclear which policy its implicitly trained value function corresponds to, and (b) it
extracts that policy with a unimodal Gaussian via advantage-weighted regression — a class too
weak to represent the (multimodal) policy the critic actually evaluates.

## Key idea

1. **IQL is an actor-critic method.** Generalize IQL's value loss to an arbitrary convex `f`
   with `f'(0)=0`, `V*(s) = argmin_V E_{a~mu}[f(Q(s,a) - V(s))]`. Its optimum satisfies
   `V*(s) = E_{a~pi_imp}[Q(s,a)]`, i.e. `V*` is the value of an **implicit actor**
   `pi_imp(a|s) ∝ mu(a|s) * |f'(Q-V*)| / |Q-V*|` — a reweighting of the behavior policy whose
   skew is set by `f`. The expectile loss gives `w_2^tau=|tau - 1(Q < V*)|`; the quantile
   loss gives `w_1^tau=|tau - 1(Q < V*)|/|Q-V*|`; the exponential/linex loss
   `f(u)=exp(alpha*u)-alpha*u` gives
   `w_exp=alpha*|exp(alpha*(Q-V_exp))-1|/|Q-V_exp|`, with
   `V_exp=(1/alpha)log sum_a exp(alpha Q(s,a)+log mu(a|s))`.
   That `V_exp` is the log partition of the exponential/AWR policy
   `pi_exp(a|s)=mu(a|s)exp(alpha(Q(s,a)-V_exp(s)))`, yielding
   `KL(mu||pi_exp)=E[alpha(V_exp-Q)]`.
2. **Decoupled, expressive policy extraction.** Because `pi_imp = mu * w / Z`, train an
   expressive behavior model `mu_phi` by **pure diffusion behavior cloning** (no weights, no
   critic), then realize `pi_imp` at inference by **importance resampling**: draw `N` candidates
   `a_i ~ mu_phi(.|s)`, weight by the chosen implicit-actor weights, and resample. The
   CleanDiffuser MuJoCo implementation uses the exponential-style finite-sample rule
   `softmax((Q-V)*weight_temperature)`; a greedy `argmax_i Q(s,a_i)` is the deterministic
   evaluation variant. Training the expressive model *with* importance weights fails
   (capacity washes out the weighting), so the weighting is applied at sampling time only.
3. **Outlier-robust score network.** A naive MLP diffusion model emits OOD outlier actions the
   critic over-scores; a high-capacity, LayerNorm-regularized residual MLP keeps candidates
   in-support, making selection robust to `N`.

The behavior model never touches the critic during training, preserving IQL's stability and
hyperparameter robustness; only the expectile `tau` materially deviates `pi_imp` from `mu`.

## Final algorithm

Critic (every other step):
- `V` expectile regression: `L_V = E[ |tau - 1(Q_targ(s,a) - V(s) < 0)| (Q_targ(s,a) - V(s))^2 ]`.
- `Q` SARSA-TD on `V(s')`: `L_Q = E[ (r + gamma (1-done) V(s') - Q(s,a))^2 ]` over twin heads.
- Polyak-update the `Q` target.

Actor (every step): diffusion BC `L_mu = E[ || eps - mu_phi(sqrt(abar_t) a + sqrt(1-abar_t) eps, s, t) ||^2 ]`.

Inference: `N` candidates from `mu_phi`, advantage `A = Q - V`, resample by
`softmax(A*weight_temperature)` in the CleanDiffuser MuJoCo pipeline (or select argmax `Q` for
the deterministic variant). The exact theorem actor would instead use the derivative-ratio
weights for the selected critic loss.

## Code (faithful to a CleanDiffuser-style implementation)

```python
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from torch.optim.lr_scheduler import CosineAnnealingLR


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4), nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x):
        return x + self.net(x)


class IDQLMlp(BaseNNDiffusion):
    """Epsilon-prediction score network for mu_phi(a|s): high capacity + LayerNorm."""
    def __init__(self, obs_dim, act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1,
                 timestep_emb_type="positional", timestep_emb_params: Optional[dict] = None):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.obs_dim = obs_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.Mish(),
                                      nn.Linear(emb_dim * 2, emb_dim))
        self.affine_in = nn.Linear(obs_dim + act_dim + emb_dim, hidden_dim)
        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.affine_out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, noise, condition):
        if condition is None:
            condition = torch.zeros(x.shape[0], self.obs_dim, device=x.device)
        t = self.time_mlp(self.map_noise(noise))      # positional timestep embedding
        x = torch.cat([x, t, condition], -1)
        return self.affine_out(self.ln_resnet(self.affine_in(x)))


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, 1))
        self.Q1, self.Q2 = head(), head()

    def both(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.Q1(x), self.Q2(x)

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class V(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        return self.V(obs)


def train(actor, dataloader, obs_dim, act_dim, args):
    # actor: a CleanDiffuser DiscreteDiffusionSDE wrapping IDQLMlp.
    #        .update(act, obs) takes one mean-squared epsilon-prediction BC step.
    #        .sample(...) returns (actions, log).
    q_net  = TwinQ(obs_dim, act_dim, args.critic_hidden_dim).to(args.device)
    q_targ = deepcopy(q_net).requires_grad_(False).eval()
    v_net  = V(obs_dim, args.critic_hidden_dim).to(args.device)
    q_optim = torch.optim.Adam(q_net.parameters(), lr=args.critic_learning_rate)
    v_optim = torch.optim.Adam(v_net.parameters(), lr=args.critic_learning_rate)
    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=args.gradient_steps)
    v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=args.gradient_steps)

    n_step = 0
    for batch in dataloader:
        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act, rew, tml = (batch["act"].to(args.device), batch["rew"].to(args.device),
                         batch["tml"].to(args.device))

        if n_step % 2 == 0:                                    # in-sample IQL critic
            q = q_targ(obs, act)
            v = v_net(obs)
            u = q - v
            v_loss = (torch.abs(args.iql_tau - (u < 0).float()) * u ** 2).mean()
            v_optim.zero_grad(); v_loss.backward(); v_optim.step()

            with torch.no_grad():
                td_target = rew + args.discount * (1 - tml) * v_net(next_obs)
            q1, q2 = q_net.both(obs, act)
            q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
            q_optim.zero_grad(); q_loss.backward(); q_optim.step()
            q_lr_scheduler.step(); v_lr_scheduler.step()

            for p, pt in zip(q_net.parameters(), q_targ.parameters()):
                pt.data.copy_(0.995 * p.data + 0.005 * pt.data)

        actor.update(act, obs)                                 # weight-free diffusion BC
        actor_lr_scheduler.step()
        n_step += 1
        if n_step >= args.gradient_steps:
            break
    return actor, q_targ, v_net


@torch.no_grad()
def select_action(actor, q_targ, v_net, obs, obs_dim, act_dim, args):
    n = obs.shape[0]
    obs_rep = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
    prior = torch.zeros((n * args.num_candidates, act_dim), device=obs.device)
    act, _ = actor.sample(prior, solver=args.solver, sample_steps=args.sampling_steps,
                          n_samples=n * args.num_candidates, condition_cfg=obs_rep,
                          w_cfg=1.0, use_ema=args.use_ema, temperature=args.temperature)

    q = q_targ(obs_rep, act)
    v = v_net(obs_rep)
    adv = (q - v).view(-1, args.num_candidates, 1)
    w = torch.softmax(adv * args.weight_temperature, dim=1)    # practical advantage reweight
    act = act.view(-1, args.num_candidates, act_dim)
    p = (w / w.sum(1, keepdim=True)).squeeze(-1)
    idx = torch.multinomial(p, 1).squeeze(-1)
    return act[torch.arange(act.shape[0]), idx].cpu().numpy()
    # greedy variant: idx = q.view(-1, args.num_candidates).argmax(1)
```

Defaults in the CleanDiffuser MuJoCo path: `tau = 0.7`, `discount = 0.99`, critic LR `3e-4`,
actor LR `3e-4`, `diffusion_steps = sampling_steps = 5`, actor hidden size `256`, `3` residual
blocks, dropout `0.1`, actor EMA `0.9999`, `num_candidates = 256`, sampling `temperature = 0.5`,
and cosine LR schedulers. The critic uses Adam; the diffusion wrapper owns the actor optimizer.
