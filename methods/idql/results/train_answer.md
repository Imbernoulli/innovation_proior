Offline reinforcement learning has to learn a good policy from a fixed dataset of transitions collected by some unknown behavior policy, with no further environment interaction. The central failure mode is distribution shift: any bootstrap update that targets `r + gamma max_{a'} Q(s', a')` must evaluate actions `a'` proposed by the learned policy, and those actions quickly leave the support of the data. The function approximator then returns over-optimistic values for these out-of-distribution actions, the backup propagates the optimism, and training diverges. Existing fixes either constrain the learned policy to stay near the behavior policy, penalize OOD Q-values, or avoid the OOD query entirely with in-sample backups. The in-sample approach is the cleanest, but it splits into two sub-problems: learning a stable value function, and then extracting a policy that actually matches what that value function evaluates. Most methods handle one side well and the other poorly.

Implicit Q-Learning (IQL) solves the value-learning side elegantly. It fits a separate value network `V(s)` to a `tau`-expectile of the target Q-values over dataset actions, then trains `Q` with a SARSA-style backup against `V(s')`. This never queries an OOD action and still performs multi-step dynamic programming. But IQL extracts the final policy with advantage-weighted regression onto a unimodal Gaussian, and the policy that the expectile critic implicitly evaluates is generally multimodal. A Gaussian cannot represent that policy faithfully, so the careful value learning is partly wasted at the extraction step. Meanwhile, purely training a diffusion model to clone the behavior distribution can represent multimodality, but it has no reward signal and cannot improve over the behavior policy. Diffusion Q-learning couples the diffusion actor and the critic tightly, which is slow and reintroduces the hyperparameter sensitivity that in-sample methods avoid.

The method I propose is IDQL, Implicit Diffusion Q-Learning. It keeps IQL's decoupled, in-sample critic but replaces the Gaussian actor with an expressive diffusion behavior model, and it applies the critic's guidance only at inference time through importance resampling. The result is a method that is stable like IQL, expressive enough to represent the true implicit policy, and cheap to train because the actor and critic never interact during training.

The first step is to understand what policy IQL's critic actually evaluates. Generalize the value loss to any convex `f` with `f'(0) = 0`, so `V*(s) = argmin_V E_{a~mu}[f(Q(s,a) - V(s))]`. At the optimum, the derivative with respect to `V` vanishes: `E_{a~mu}[f'(Q - V*)] = 0`. Because `f'` is nondecreasing and zero at zero, we can write `f'(x) = |f'(x)| * x/|x|`. Folding the nonnegative scalar weight `w(s,a) = |f'(Q - V*)| / |Q - V*|` into the behavior policy gives an implicit actor `pi_imp(a|s) ∝ mu(a|s) * w(s,a)`, and the optimality condition becomes `E_{a~pi_imp}[Q(s,a) - V*(s)] = 0`. Therefore `V*` is exactly the value function of the policy `pi_imp`. For the expectile loss used by IQL, the weight simplifies to `w(s,a) = |tau - 1(Q(s,a) < V*(s))|` — a sign-dependent reweighting that spreads mass over all above-value actions. That policy is multimodal whenever the behavior policy is multimodal, which is why a unimodal Gaussian extraction is inadequate.

The second step is to realize `pi_imp` without training the expressive model on importance-weighted data. Expressive models trained with importance-weighted likelihood tend to raise the likelihood of all training points and wash out the weighting signal. Instead, IDQL trains the diffusion model with pure behavior-cloning on dataset actions only, so it simply learns `mu(a|s)` as accurately as possible. At inference, it draws `N` candidate actions from this behavior model, computes their advantages `A(s,a_i) = Q(s,a_i) - V(s)`, and resamples according to `softmax(A * weight_temperature)`. In the large-`N` limit this realizes the advantage-reweighted policy. A deterministic evaluation variant simply selects the candidate with the highest `Q`. Because the critic is applied only to in-support candidates from the behavior model, there is no OOD action query at training or inference.

The third step is to make the behavior model's samples stay in support. A naive MLP score network on low-dimensional continuous actions fits the modes loosely and emits outlier samples, which the critic can over-score. IDQL uses a high-capacity residual MLP with LayerNorm inside each block; the residual structure and normalization tame these outliers while the width preserves the ability to fit sharp modes. The critic side stays standard: twin Q networks, a target network, Adam at `3e-4`, Polyak target updates, and cosine learning-rate decay. Training interleaves the two independent objectives, updating the critic every other step and the behavior model every step.

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
    """Epsilon-prediction score network for the behavior model mu_phi(a|s)."""
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
        t = self.time_mlp(self.map_noise(noise))
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
        obs = batch["obs"]["state"].to(args.device)
        next_obs = batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        if n_step % 2 == 0:
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

        actor.update(act, obs)  # weight-free diffusion behavior cloning
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
    w = torch.softmax(adv * args.weight_temperature, dim=1)
    act = act.view(-1, args.num_candidates, act_dim)
    p = (w / w.sum(1, keepdim=True)).squeeze(-1)
    idx = torch.multinomial(p, 1).squeeze(-1)
    return act[torch.arange(act.shape[0]), idx].cpu().numpy()
```
