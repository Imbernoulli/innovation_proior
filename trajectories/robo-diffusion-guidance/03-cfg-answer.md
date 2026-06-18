**Problem.** Return-conditioning rescued hopper, but decision-diffuser changed the guidance
mechanism *and* the whole backbone at once, so I cannot attribute the soft walker2d (0.6972) and
halfcheetah (0.4355) numbers to either. I need the controlled comparison: classifier-free guidance
on the *original* substrate backbone, so the only difference from the eventual CG default is the
guidance pathway.

**Key idea (CFG on the original `JannerUNet1d`-on-`obs+act` backbone).** Keep every architectural
choice of the default — same backbone, same `DiscreteDiffusionSDE`, same masking, same `obs+act`
trajectory — and swap only the guidance. Remove the `CumRewClassifier`; add an `MLPCondition` over
the normalized return trained with `label_dropout`; sample with `condition_cfg=target_return`,
`w_cfg` from config, `w_cg=0`, and no candidate re-ranking (read the action straight off the
conditional sample).

**Why.** By Bayes the classifier hidden in any conditional+unconditional generator gives
`ε̃=(1+w)ε(z,c)-w·ε(z)` — a linear mix of two epsilon predictions, no second network, no classifier
gradient, no offline value function. One net trained with condition dropout serves as both. This is
the apples-to-apples point: it isolates "CFG vs CG, same backbone" from "the DD architecture
upgrade."

**Step-3 edit.** Identical `JannerUNet1d` + masking as the default; `classifier=None`,
`nn_condition=MLPCondition(in_dim=1, dropout=label_dropout)`; training calls `agent.update(x, val)`
with `val/return_scale`; inference samples once per env with `w_cfg`, `w_cg=0`, reads
`traj[:,0,obs_dim:]`. Skip `report_parameters` on the tiny condition net (its top-K indexing
crashes).

**Hyperparameters.** `model_dim=32`, `kernel_size=5`, `attention=False`, `diffusion_steps=20`,
`predict_noise=False`, `ema_rate=0.9999`, `action_loss_weight=10`; `label_dropout=0.25`;
`w_cfg`=4.4/6.0/3.2 (hopper/walker2d/halfcheetah); `target_return`=0.7/0.75/1.1; `temperature=0.5`;
single sample per env (no `num_candidates`).

```python
# CleanDiffuser/pipelines/custom_guidance.py — cfg fill (CFG on original backbone)
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
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed


@hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

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
    # cfg: Network + Agent Setup (JannerUNet1d + MLPCondition, no classifier)
    # ============================================================================
    return_scale = DD_RETURN_SCALE[args.task.env_name]

    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)

    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.model_dim,
        hidden_dims=[args.model_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Condition Network: MLPCondition =======================")
    print(f"Total parameters: {sum(p.numel() for p in nn_condition.parameters())}")
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ============================================================================
    # cfg: Training (diffusion only with return-conditioning, no classifier)
    # ============================================================================
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        agent.train()
        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale
            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x, val)['loss']
            diffusion_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "finetune":
        pass

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        # ============================================================================
        # cfg: Inference Setup (diffusion only)
        # ============================================================================
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)   # FIXED
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # ============================================================================
        # cfg: Prior + Condition Initialization (return-conditioned via MLPCondition)
        # ============================================================================
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return

        for i in range(args.num_episodes):

            env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # ============================================================================
                # cfg: Action Sampling (CFG, no candidate re-ranking)
                # ============================================================================
                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    condition_cfg=condition,
                    w_cfg=args.task.w_cfg,
                    w_cg=0.0,
                    temperature=args.temperature)

                act = traj[:, 0, obs_dim:]
                act = act.clip(-1., 1.).cpu().numpy()

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
