## Problem (from step 1)

Diffusion BC clones the `medium` buffer faithfully (tight seeds) but its ceiling *is* the behavior
policy `mu` — it has no reward signal, so it cannot prefer the better-than-average actions the buffer
contains. The next move is to let value in. But bootstrapping `Q` toward `r + gamma max_{a'} Q(s',a')`
queries out-of-distribution actions, which overestimate and diverge.

## Key idea

Learn value **in-sample** and extract the actor **at inference**, keeping the floor's diffusion-BC
actor untouched.

1. **In-sample critic (IQL).** Never form `max_{a'} Q`. Estimate the value of the best in-support
   action by expectile regression of `Q` over dataset actions on a separate value net `V`:
   `L_V = E[ |tau - 1(Q_targ-V<0)| (Q_targ-V)^2 ]`, then SARSA-back `Q` against `V(s')`:
   `L_Q = E[ (r + gamma(1-done) V(s') - Q)^2 ]` over twin heads. Interpolates SARSA (`tau=0.5`) to
   support-constrained Q-learning (`tau->1`); never touches an OOD action.
2. **IQL is an actor-critic.** Generalizing the value loss to convex `f` with `f'(0)=0`, its optimum
   gives `V*(s) = E_{pi_imp}[Q(s,a)]` where `pi_imp(a|s) ∝ mu(a|s) |f'(Q-V*)|/|Q-V*|` — a *reweighting*
   of `mu`. On a multimodal `medium` buffer this implicit actor is multimodal, so a Gaussian extraction
   (AWR) cannot represent it.
3. **Decoupled, expressive extraction.** Train the actor as pure diffusion BC of `mu` (same as step 1 —
   importance-weighting an expressive model washes the skew out), then realize `pi_imp` at inference by
   importance resampling: draw `N` candidates, `adv = Q - V`, `w = softmax(adv * weight_temperature)`,
   resample one. Value enters only at inference.

## Why it works

Same actor as the floor, plus a value-based selection that can only help if `Q`/`V` are informative — so
it must beat diffusion BC on every environment. The critic never sees the actor during training and the
actor never sees `Q`/`V`, so the floor's stable, reproducible BC is preserved and only the *selection*
chases value. That decoupling is also its ceiling: the actor's own distribution is never pushed above
`mu`, so IDQL can only pick the best of `N` samples from `mu` — it should land above the floor but below
DQL, which maximizes `Q` *during* actor training.

## Hyperparameters

Actor backbone `IDQLMlp` (emb_dim=64, actor hidden / n_blocks / dropout from args), wrapper
`DiscreteDiffusionSDE` with `diffusion_steps = sampling_steps = 5`. Critic `IDQLQNet` (twin-Q) +
`IDQLVNet`, hidden = `critic_hidden_dim`, Adam at `critic_learning_rate` (3e-4), cosine schedulers.
Expectile `iql_tau` (≈0.7), `discount = 0.99`, Polyak `0.995`, critic updated every other step, actor
(BC) every step. Inference: `num_candidates = 50`, `weight_temperature` from args, `use_ema=True`. No
`eta`/Q-maximization term on the actor.

## Scaffold edit

```python
import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import report_parameters, IDQLQNet, IDQLVNet
from utils import set_seed

# ============================================================================
# idql baseline: full IDQL pipeline (decoupled actor / IQL critic)
# ============================================================================

# --------------- Network Architecture -----------------
nn_diffusion = IDQLMlp(
    obs_dim, act_dim, emb_dim=64,
    hidden_dim=args.actor_hidden_dim, n_blocks=args.actor_n_blocks, dropout=args.actor_dropout,
    timestep_emb_type="positional")
nn_condition = IdentityCondition(dropout=0.0)

# --------------- Diffusion Model Actor --------------------
actor = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    x_max=+1. * torch.ones((1, act_dim)),
    x_min=-1. * torch.ones((1, act_dim)),
    diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

# ------------------ Critic (twin-Q + value net) ---------------------
iql_q = IDQLQNet(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
iql_v = IDQLVNet(obs_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
q_optim = torch.optim.Adam(iql_q.parameters(), lr=args.critic_learning_rate)
v_optim = torch.optim.Adam(iql_v.parameters(), lr=args.critic_learning_rate)

# ---------------------- Training ----------------------
if args.mode == "train":

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=args.gradient_steps)
    v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=args.gradient_steps)

    actor.train(); iql_q.train(); iql_v.train()
    n_gradient_step = 0

    for batch in loop_dataloader(dataloader):

        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # -- IQL critic (every other step)
        if n_gradient_step % 2 == 0:
            q = iql_q_target(obs, act)
            v = iql_v(obs)
            v_loss = (torch.abs(args.iql_tau - ((q - v) < 0).float()) * (q - v) ** 2).mean()
            v_optim.zero_grad(); v_loss.backward(); v_optim.step()

            with torch.no_grad():
                td_target = rew + args.discount * (1 - tml) * iql_v(next_obs)
            q1, q2 = iql_q.both(obs, act)
            q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
            q_optim.zero_grad(); q_loss.backward(); q_optim.step()
            q_lr_scheduler.step(); v_lr_scheduler.step()

            for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
                target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

        # -- Policy (every step): weight-free diffusion BC, no eta * q_loss
        bc_loss = actor.update(act, obs)["loss"]
        actor_lr_scheduler.step()

        n_gradient_step += 1
        if n_gradient_step >= args.gradient_steps:
            break

# ---------------------- Inference ----------------------
# (checkpoint load) load actor + iql_q / iql_q_target / iql_v; .eval() all
# (action selection) advantage reranking over num_candidates:
#     obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
#     obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
#     act, _ = actor.sample(
#         prior, solver=args.solver, n_samples=args.num_envs * args.num_candidates,
#         sample_steps=args.sampling_steps, condition_cfg=obs, w_cfg=1.0,
#         use_ema=args.use_ema, temperature=args.temperature)
#     with torch.no_grad():
#         adv = (iql_q_target(obs, act) - iql_v(obs)).view(-1, args.num_candidates, 1)
#         w = torch.softmax(adv * args.task.weight_temperature, 1)
#         act = act.view(-1, args.num_candidates, act_dim)
#         p = w / w.sum(1, keepdim=True)
#         indices = torch.multinomial(p.squeeze(-1), 1).squeeze(-1)
#         sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
```
</content>
