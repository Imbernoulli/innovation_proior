## Research question

Offline reinforcement learning on D4RL MuJoCo control. I am handed a fixed buffer of transitions
`(s, a, r, s')` collected by some behavior policy `mu`, with no further environment interaction, and I
must learn a **Markov policy** that maps the current observation to one action. The design object is the
**policy algorithm core** — the actor, the optional critic, the training objective, and the inference-time
action-selection rule. Everything outside the policy core — the D4RL dataset construction, the environment
names, the evaluation loop, the seeds, the episode count, the vectorized-environment count, and the
checkpoint names — is frozen.

## Prior art / Background / Baselines

- **Behavior cloning by L2 regression.** Fits `f(s) -> a` by minimizing `E||a - f(s)||^2`, which converges
  to the conditional mean `E[a|s]` — a unimodal Gaussian head.
- **Conservative offline value learning (CQL).** Bootstraps a value function while penalizing Q-values of
  out-of-distribution actions, pushing them below in-support values so the policy cannot exploit phantom
  optima. The penalty is a coarse, dataset-wide pressure and the extracted policy remains a Gaussian.
- **Advantage-weighted regression (AWR / AWAC).** Extracts a policy by
  `E[exp(beta * A(s,a)) log pi(a|s)]`, a maximum-likelihood fit to the KL-constrained reward-maximizing
  distribution `pi ∝ mu * exp(beta * A)`. This stays near data by construction, with a unimodal Gaussian
  fit to the reweighted target.
- **In-sample value learning (IQL).** Avoids writing `max_{a'} Q(s',a')` by estimating the value of the best
  in-support action through expectile regression of `Q` over dataset actions on a separate value net `V`, then
  SARSA-backing `Q` against `V(s')`. The value side is stable and in-sample.
- **Score-based generative modeling / DDPM.** Models a distribution by learning the gradient field of its
  log-density through a denoising regression objective, then samples via a stochastic reverse chain from
  Gaussian noise. This gives an expressive, multimodal conditional density `p(a|s)` trained by plain MSE with
  no intractable normalizer.

## Fixed substrate / Code framework

A CleanDiffuser offline-RL pipeline (`CleanDiffuser/pipelines/custom_policy.py`) is frozen outside the
editable region: it builds the D4RL MuJoCo TD dataset (`D4RLMuJoCoTDDataset` over
`d4rl.qlearning_dataset(env)`, with optional reward normalization), a shuffled `DataLoader`, and reads
`obs_dim, act_dim` off the dataset. After training it runs a vectorized evaluation loop
(`gym.vector.make(env_name, num_envs)`), normalizes observations with `dataset.get_normalizer()`, collects
episodic returns over `num_episodes`, converts them to D4RL normalized scores via
`env.get_normalized_score`, and prints `EVAL_METRICS normalized_score=... episode_reward=...`.

The loop provides the actor wrapper `DiscreteDiffusionSDE` (a discrete VP-SDE / DDPM diffusion model exposing
`.loss(act, obs)`, `.update(act, obs)["loss"]` for a denoising-BC step, `.sample(...)`, `.ema_update()`,
`.save/.load`, and its own optimizer), the `IdentityCondition` conditioner (the state is fed in raw), the
score backbones (`DQLMlp`, `IDQLMlp`), the critics (`DQLCritic`, `IDQLQNet`, `IDQLVNet`), and
`report_parameters`. The diffusion actor uses `diffusion_steps = sampling_steps = 5` and a cosine noise
schedule; actions live in the box `[-1, 1]`.

## Editable interface

Exactly one region is editable — the policy-algorithm core inside `pipeline(args)` in `custom_policy.py`
(the imports block, the network/critic construction, the training loop, the inference-time checkpoint load,
the eval `prior`, and the action-selection block). A method fills this contract by: constructing a diffusion
**actor** (and optionally a **critic**); in the training loop, taking an actor objective (a denoising-BC
`loss`/`update` step, possibly with a Q-maximization term) and optionally a critic objective; at inference,
loading the checkpoint(s), sampling candidate action(s) from the actor, and selecting the action to execute
(single sample, or a reranking over `num_candidates` using the critic).

The default fill below instantiates a diffusion actor and a twin-Q critic, trains the actor with a BC term
plus a Q-maximization term, and reranks sampled candidates by softmax over `Q` at inference.

```python
import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from utils import set_seed

# ---- EDITABLE REGION (default fill: diffusion actor + twin-Q critic) ----

# --------------- Network Architecture -----------------
nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
nn_condition = IdentityCondition(dropout=0.0).to(args.device)

# --------------- Diffusion Model Actor --------------------
actor = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    x_max=+1. * torch.ones((1, act_dim), device=args.device),
    x_min=-1. * torch.ones((1, act_dim), device=args.device),
    diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

# ------------------ Critic ---------------------
critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
critic_target = deepcopy(critic).requires_grad_(False).eval()
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

# ---------------------- Training ----------------------
if args.mode == "train":
    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor.train(); critic.train()
    prior = torch.zeros((args.batch_size, act_dim), device=args.device)

    for batch in loop_dataloader(dataloader):
        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # Critic Training: SARSA-style TD with next action drawn from the diffusion actor
        current_q1, current_q2 = critic(obs, act)
        next_act, _ = actor.sample(
            prior, solver=args.solver, n_samples=args.batch_size, sample_steps=args.sampling_steps,
            use_ema=True, temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
        target_q = torch.min(*critic_target(next_obs, next_act))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_optim.zero_grad(); critic_loss.backward(); critic_optim.step()

        # Policy Training: BC loss + eta * Q-maximization on freshly sampled actions
        bc_loss = actor.loss(act, obs)
        new_act, _ = actor.sample(
            prior, solver=args.solver, n_samples=args.batch_size, sample_steps=args.sampling_steps,
            use_ema=False, temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
        with FreezeModules([critic, ]):
            q1_new_action, q2_new_action = critic(obs, new_act)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = bc_loss + args.task.eta * q_loss
        actor.optimizer.zero_grad(); actor_loss.backward(); actor.optimizer.step()

        actor_lr_scheduler.step(); critic_lr_scheduler.step()
        if n_gradient_step % args.ema_update_interval == 0:
            if n_gradient_step >= 1000:
                actor.ema_update()
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.copy_(0.995 * p.data + 0.005 * tp.data)
        # ... logging / checkpointing / break at gradient_steps ...

# ---------------------- Inference ----------------------
elif args.mode == "inference":
    actor.load(...); critic.load_state_dict(...); critic_target.load_state_dict(...)
    actor.eval(); critic.eval(); critic_target.eval()
    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    # for each env step: sample num_candidates actions per env, then
    #   q = critic_target.q_min(obs, act).view(-1, num_candidates, 1)
    #   w = softmax(q * weight_temperature, dim=1); idx = multinomial(w); pick act[idx]
```

## Evaluation settings

Three D4RL MuJoCo environments — **hopper-medium-v2**, **walker2d-medium-v2**, **halfcheetah-medium-v2**
— evaluated with `num_envs = 50`, `num_episodes = 3`, `use_ema = True`. Training is fixed at
`gradient_steps = 1,000,000` for every method (a method may shorten its own training but never lengthen
it). For methods that rerank, `num_candidates = 50` at inference; single-sample methods ignore it. Metrics:
`normalized_score` (D4RL normalized score, higher is better), `episode_reward` (raw return), and
`training_time` (seconds, lower is better). The final score is the **geometric mean** over the three
environment-specific normalized scores. Seed 42 is primary; seeds 123 and 456 are run for the strongest
baseline.
