**Problem.** A trajectory-level diffusion planner can sample dynamically plausible plans, but
"plausible" is not "good": on `*-medium-v2` data the unconditional plan distribution is dominated
by the mediocre behavior policy. Before designing any guidance, I need the steering-off reference —
the score a guidance method must beat — so I measure the planner with the entire reward pathway
removed.

**Key idea (unconditional trajectory diffusion).** Diffuse the whole state-action array
`τ=[s_0…s_{H-1}; a_0…a_{H-1}]` at once with a `JannerUNet1d` temporal U-Net wrapped by a
`DiscreteDiffusionSDE`; anchor the plan at the current observation by inpainting (`fix_mask` clamps
`τ[0,:obs_dim]` every reverse step, its loss zeroed); up-weight the executed first action. No
classifier, no condition network, no return — sampling is Langevin-like ascent on the trajectory
density, so plans stay on the data manifold. Draw one plan per env-step and read off the first
action; no candidate re-ranking.

**Why.** This isolates the manifold model from every steering choice. Its score is exactly the
quantity the guidance study is about — the gap above it measures how much steering buys. It is the
floor by construction: it samples a *typical* plausible plan, with no preference for the high-return
plans the data also contains.

**Step-1 edit.** Strip the default's classifier pathway and nothing else: drop `nn_classifier` /
`CumRewClassifier` and pass `classifier=None`; in training call only `agent.update(x)`; at inference
load only the diffusion checkpoint; sample once per env with `w_cg=0.0, w_cfg=0.0`; read
`act = traj[:,0,obs_dim:]`. Define zero `logp`/`idx` placeholders for the fixed post-sample print.

**Hyperparameters.** `model_dim=32`, `kernel_size=5`, `attention=False`, `diffusion_steps=20`,
`sampling_steps=20`, `solver=ddpm`, `predict_noise=False`, `ema_rate=0.9999`,
`action_loss_weight=10`, `temperature=0.5`, `diffusion_gradient_steps=100000`, `batch_size=256`;
one sample per env (no `num_candidates`); `w_cg=w_cfg=0`.

```python
# CleanDiffuser/pipelines/custom_guidance.py — no_guidance fill (unconditional)
import os

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
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
    # no_guidance: Network + Agent Setup (no classifier)
    # ============================================================================
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ============================================================================
    # no_guidance: Training (diffusion only, no classifier)
    # ============================================================================
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        agent.train()
        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)['loss']
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
        # no_guidance: Inference Setup (diffusion only)
        # ============================================================================
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)   # FIXED
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # ============================================================================
        # no_guidance: Prior Initialization (no condition, no target return)
        # ============================================================================
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)

        for i in range(args.num_episodes):

            env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # ============================================================================
                # no_guidance: Action Sampling (w_cg=0, no re-ranking)
                # ============================================================================
                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    w_cg=0.0,
                    w_cfg=0.0,
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
