The steering-off floor told me exactly what is missing, in three numbers: unconditional trajectory diffusion scored 0.4362 on hopper, 0.6019 on walker2d, 0.4115 on halfcheetah, geometric mean 0.4763. The split is the one I expected — halfcheetah barely moved off the dense-reward medium policy, walker2d held up, and hopper collapsed to 0.44 because hopper falls over and the medium dataset is full of sub-optimal windows whose *plausible* continuations include falling. This is not a learning or capacity problem; the manifold model is fine. It is that "plausible" is not "good": the dataset contains good trajectories and bad ones, and sampling faithfully reproduces the mixture instead of selecting the good part. The entire job of this rung is to make the sampler *prefer high return* — and to do it without re-importing an offline value function and its out-of-distribution over-estimation, because the floor result argues *for* the generative route. The manifold model already stays on-distribution by construction, so all I need to add is selection, not a whole value-estimation apparatus that re-imports the OOD pathology.

I propose the Decision Diffuser: state-only diffusion conditioned on return by classifier-free guidance, with actions recovered by a separate inverse-dynamics model. Three pieces, each chosen against a concrete worse alternative. First, *what* to diffuse over. The floor diffused the whole $\text{obs}+\text{act}$ trajectory, but states and actions are not the same animal: states are continuous and smooth along a trajectory, while actions are joint torques — high-frequency, jerky, much less structured. A denoiser regresses the clean signal out of noise, and a non-smooth target is genuinely harder to fit, so jointly diffusing both makes the same machinery chase the hardest, least-structured part of the trajectory and the action channel drags it down. The move is to not diffuse the actions at all: diffuse only the state sequence $x^0=(s_t,\dots,s_{t+H-1})$ — clean, smooth, continuous — and recover each action from consecutive states with an inverse-dynamics model $a_t=f_\phi(s_t,s_{t+1})$, a small supervised MLP trained by plain regression on the same offline transitions. The jerky signal never enters the diffusion model. On the edit surface this makes the diffusion target $\text{obs\_dim}$-wide, the masking clamp the leading *state* column, and a separate `MlpInvDynamic` train and answer at inference — the substrate's `JannerUNet1d`-on-$\text{obs}+\text{act}$ is replaced wholesale.

Second, *how* to condition. The obvious route, classifier guidance, trains a predictor $p_\phi(\text{return}\mid x_k)$ on noised trajectories and steers with its gradient — but a predictor of "the return achieved by this noisy trajectory" *is* an offline value function, with the same OOD over-estimation, and its gradient then guides the sampler *toward* off-distribution, actually-bad regions. There is also a structural mismatch: the diffusion model would be trained unconditionally and only steered at test time by a separately-learned object, so the thing I sample is not the thing I modeled. Classifier-free guidance is built for exactly this. The classifier hidden in any conditional-plus-unconditional generator is, by Bayes, $\nabla_x\log p(y\mid x)=\nabla_x\log p(x\mid y)-\nabla_x\log p(x)$ — the classifier gradient I wanted is just the conditional score minus the unconditional one, both producible by the generator. So I train one network to be both a conditional denoiser $\varepsilon_\theta(x_k,y,k)$ and an unconditional one $\varepsilon_\theta(x_k,\varnothing,k)$ by replacing the condition $y$ with a null token with probability $p$ during training (condition dropout), and at sample time use
$$\hat\varepsilon=\varepsilon_\theta(x_k,\varnothing,k)+\omega\,\big(\varepsilon_\theta(x_k,y,k)-\varepsilon_\theta(x_k,\varnothing,k)\big).$$
At $\omega=1$ this is plain conditional sampling; $\omega>1$ extrapolates past the conditional, pushing harder toward $y$. No value function, so no deadly triad and no OOD over-estimation, and the network is shaped by the condition throughout training, so training and sampling objectives finally agree. Here $y$ is the dataset-normalized return — I divide `val` by a per-env `DD_RETURN_SCALE` so it lands in roughly $[0,1]$ — and at sample time I condition on the task's `target_return` (0.7 / 0.75 / 1.1 for hopper / walker2d / halfcheetah) with `w_cfg` from config (4.4 / 6.0 / 3.2). Those weights sit well above 1 because a scalar return is a weak, low-dimensional handle, so I lean hard on the extrapolation; correspondingly I set `label_dropout=0.25` rather than the $\sim$0.1 image work uses, so the unconditional branch — which I lean on heavily — is trained on a healthy fraction of batches and stays a reliable baseline for the $\varepsilon(y)-\varepsilon(\varnothing)$ difference.

Third, a subtlety the floor makes me address: conditioning gets me the right *region* of the distribution but not its high-likelihood core. The medium data is mostly sub-optimal, so even conditioned on high return the model's idea of "a trajectory" is a noisy multimodal cloud, and I will draw mediocre members as often as good ones unless I concentrate the draw. The lever is the sampling temperature: shrinking the per-step reverse noise concentrates samples on the high-probability mode, which after return-conditioning are the genuinely high-return ones. Zero noise collapses to one deterministic mode and kills the diversity that lets the model recombine fragments; full noise keeps drawing mediocre members; so I want an intermediate temperature, and the substrate's `temperature=0.5` is exactly that std-scale. Tied to the state-only formulation is a backbone choice: the smooth state-only sequence is a natural fit for a Transformer's global self-attention over the time axis, so I take the canonical `DiT1d(obs_dim, emb_dim, d_model, n_heads, depth, timestep_emb_type="fourier")` wrapped by `ContinuousDiffusionSDE(... noise_schedule="linear")`, with an `MLPCondition(in_dim=1)` for the scalar return.

I should be precise that this lands the single-attribute core, not the full method. The general formulation also offers attribute *composition* — combining several conditioning variables additively in score space, including a NOT operator that flips a score difference — and constraint/skill conditioning via one-hot indicators. None of that is wired here: this task conditions on a single scalar return, one `MLPCondition` over `in_dim=1`, one `w_cfg`, one `target_return` per env. So I build the $n=1$ case of the composition — exactly the consistency check of that general derivation — and explicitly do not build the multi-attribute machinery, because the harness has no place for it. The history-clamping is also the simple version: I clamp only the leading state column to the current observation (`{0: obs}` inpainting), not a length-$C$ history queue. The pieces I do build are state-only diffusion, classifier-free return conditioning with `label_dropout=0.25`, `MlpInvDynamic` action recovery, low-temperature sampling at 0.5, and receding-horizon planning that clamps the first state and reads $a_t=f_\phi(s_t,s_{t+1})$. The delta from the floor is therefore large and specific: where no-guidance diffused $\text{obs}+\text{act}$ jointly and read the first action straight off, this rung diffuses *states only* with a Transformer backbone, recovers actions with a separate inverse-dynamics regressor, and conditions the sampler on high return with no value function — which I expect to rescue the hopper number most, lift walker2d moderately, and leave halfcheetah near its already-saturated 0.41.

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
