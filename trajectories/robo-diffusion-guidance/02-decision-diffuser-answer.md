**Problem.** Unconditional trajectory diffusion samples *plausible* plans, which on mixed
`*-medium` data means mediocre plans — worst on the unforgiving hopper (0.4362). I need the sampler
to prefer *high-return* behavior, ideally without re-importing an offline value function and its
out-of-distribution over-estimation.

**Key idea (state-only diffusion + classifier-free return guidance + inverse dynamics).** Diffuse
only the smooth state sequence (drop the jerky action channel), recover actions with a supervised
`MlpInvDynamic` `a_t=f_φ(s_t,s_{t+1})`. Condition on normalized return with classifier-free
guidance: one network trained with `label_dropout` is both `ε(x,y,k)` and `ε(x,∅,k)`, and sampling
extrapolates `ε̂=ε(∅)+ω·(ε(y)-ε(∅))` toward `target_return` with `ω=w_cfg`. Low-temperature
sampling (0.5) concentrates draws on the high-likelihood (post-conditioning: high-return) mode. No
classifier, no Bellman backup.

**Why.** Maximum-likelihood conditional generation stays on the data manifold by construction and
has no value function, so it avoids the deadly triad. CFG keeps the conditional model shaped by the
return throughout training, so training and sampling objectives agree — unlike classifier guidance,
which steers a separately-trained unconditional model at test time only.

**Step-2 edit (departs from the substrate backbone).** Swap `JannerUNet1d`-on-`obs+act` for a
state-only `DiT1d` Transformer + `ContinuousDiffusionSDE`; add `MLPCondition(in_dim=1)` and
`MlpInvDynamic`; train diffusion on `obs` with `val/DD_RETURN_SCALE` plus the inverse-dynamics
regression; at inference condition on `target_return` with `w_cfg` and recover the action from the
first two predicted states. Single-attribute return conditioning only — the harness wires no
constraint/skill composition.

**Hyperparameters.** State-only diffusion; `label_dropout=0.25`; `w_cfg`=4.4/6.0/3.2 (hopper/walker2d/halfcheetah);
`target_return`=0.7/0.75/1.1; `temperature=0.5`; `noise_schedule="linear"`; `MlpInvDynamic` hidden 512;
`diffusion_gradient_steps=100000`, `batch_size=256`; no candidate re-ranking.

```python
# CleanDiffuser/pipelines/custom_guidance.py — decision_diffuser fill (DD port)
import os

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed


@hydra.main(config_path="../configs/dd/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset (FIXED) ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ============================================================================
    # Network + Agent Setup (DiT1d state-only + CFG condition + inverse dynamics)
    # ============================================================================
    nn_diffusion = DiT1d(
        obs_dim, emb_dim=args.emb_dim,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    fix_mask[0] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    loss_weight[1] = args.next_obs_loss_weight

    agent = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # ============================================================================
    # Training (conditional diffusion + inverse dynamics)
    # ============================================================================
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        agent.train()
        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale

            log["avg_loss_diffusion"] += agent.update(obs, val)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.invdyn_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "finetune":
        pass

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        # ============================================================================
        # Inference Setup
        # ============================================================================
        agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
        agent.eval()
        invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)   # FIXED
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # ============================================================================
        # Prior + Condition Initialization (return-conditioned)
        # ============================================================================
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
        condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return

        for i in range(args.num_episodes):

            env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # ============================================================================
                # Action Sampling (CFG, inverse dynamics, no re-ranking)
                # ============================================================================
                prior[:, 0] = obs
                traj, log = agent.sample(
                    prior, solver=args.solver,
                    n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                    condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)

                with torch.no_grad():
                    act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

                # FIXED post-sample print references logp/idx; define harmless placeholders.
                logp = torch.zeros((1, args.num_envs), device=args.device)
                idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)

                obs, rew, done, info = env_eval.step(act)   # FIXED

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

            episode_rewards.append(ep_reward)

        raw_episode_rewards = episode_rewards
        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        mean_score = float(np.mean(episode_rewards))
        std_score = float(np.std(episode_rewards))
        print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f}")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
```
