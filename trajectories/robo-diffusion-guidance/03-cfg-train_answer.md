Return-conditioning did what I hoped on hopper and raised a clean confound everywhere else. The Decision Diffuser scored 0.9345 on hopper, 0.6972 on walker2d, 0.4355 on halfcheetah, geometric mean 0.6571, up from the floor's 0.4763 — hopper jumped from 0.4362 to 0.9345, exactly the rescue I predicted. But the geometric mean is dragged down by the soft walker2d and halfcheetah, and I cannot tell *why*, because that rung changed two things at once: it did not just add classifier-free guidance, it swapped the entire backbone — state-only diffusion instead of $\text{obs}+\text{act}$, a `DiT1d` Transformer instead of `JannerUNet1d`, a `ContinuousDiffusionSDE` instead of the discrete one, plus an inverse-dynamics net. Any of those could be costing walker2d and halfcheetah. The architecture change is a confound, and resolving it is the whole point of this rung.

So the design is constrained before I derive anything: keep every architectural choice of the original substrate — the same `JannerUNet1d` backbone on $\text{obs}+\text{act}$ trajectories, the same `DiscreteDiffusionSDE`, the same diffusion steps, horizons, masking, and `action_loss_weight` — and change *only* the guidance pathway, from classifier guidance to classifier-free. That makes this the apples-to-apples comparison: it differs from the eventual CG default by the guidance mechanism alone, and it differs from the Decision Diffuser by holding the backbone fixed. If it lands near the Decision Diffuser, the architecture swap was doing little; if it lands well above on walker2d or halfcheetah, the swap was *costing* something.

I propose classifier-free guidance on the original backbone, and the derivation is what justifies it as the right mechanism rather than just a different knob. The substrate's default is classifier guidance: train a separate predictor $p_\phi(c\mid z_\lambda)$ on the noised latents and nudge each reverse step toward higher class-probability, $\hat\varepsilon=\varepsilon_\theta(z_\lambda,c)-g\,\sigma_\lambda\,\nabla_z\log p_\phi(c\mid z_\lambda)$. Collecting gradients, this follows the score of $p(z\mid c)\,p_\phi(c\mid z)^g$, and raising the classifier to the power $g$ sharpens it onto its high-confidence modes — a real selection knob. But three things bother me. It is a *second trained model*, and because it sees noisy inputs at sample time it must be trained on noised latents, so I cannot reuse a clean predictor — I have to build a noise-aware reward predictor (the default's `CumRewClassifier`). Worse for a return predictor specifically, a predictor of "the return this noisy trajectory will achieve" *is* an offline value function, with all the OOD over-estimation that implies; its gradient can guide the sampler toward off-support, actually-bad regions. And the guidance direction is the input-gradient of a classifier, which is structurally an adversarial perturbation.

The way out is to notice the classifier already implicit in any conditional-plus-unconditional generator. By Bayes $p(c\mid z)\propto p(z\mid c)/p(z)$, so $\nabla_z\log p(c\mid z)=\nabla_z\log p(z\mid c)-\nabla_z\log p(z)$ — the $\log p(c)$ term has no $z$-gradient, but the $p(z)$ denominator is the unconditional score and must not be dropped. Both gradients are things the generator already has: $\nabla_z\log p(z\mid c)$ is, up to $-1/\sigma_\lambda$, the conditional epsilon, and $\nabla_z\log p(z)$ is, up to $-1/\sigma_\lambda$, the unconditional epsilon. So the classifier gradient I was about to train a whole network to compute is just $-(1/\sigma_\lambda)\,[\varepsilon(z,c)-\varepsilon(z)]$. Substituting that implicit classifier into the classifier-guidance formula, the machinery falls away:
$$\tilde\varepsilon(z,c)=\varepsilon(z,c)-w\,\sigma_\lambda\,\nabla_z\log p(c\mid z)=\varepsilon(z,c)+w\,[\varepsilon(z,c)-\varepsilon(z)]=(1+w)\,\varepsilon(z,c)-w\,\varepsilon(z).$$
The $\sigma_\lambda$ and $1/\sigma_\lambda$ cancel exactly, and what is left has no classifier in it — just a linear mix of conditional and unconditional epsilon with $w$ the dial. At $w=0$ it is the plain conditional model; crank $w$ and I sharpen toward the condition. Crucially, because $\varepsilon(z,c)$ and $\varepsilon(z)$ come from unconstrained networks, their difference is in general *not* the gradient of any classifier, so the adversarial-attack objection dissolves: the implicit classifier was the inspiration for the formula, not a thing the sampler differentiates. And I get the unconditional epsilon for free: train *one* network to be both by replacing the real condition with a null token with probability `label_dropout` during training — the MSE-optimal output given $(z,\varnothing)$ is the marginalized denoiser, i.e. the unconditional score, while given $(z,c)$ it is the conditional score. The whole method is two small changes to the original pipeline: drop the condition with probability `label_dropout` in training, and mix conditional and unconditional epsilon at sample time. No `CumRewClassifier`, no classifier gradient, no second network.

Landing it on the edit surface while holding the backbone fixed: in the network setup I keep the identical `JannerUNet1d(obs_dim+act_dim, …, kernel_size=5, attention=False)` and remove the `HalfJannerUNet1d` / `CumRewClassifier` entirely. In its place I add an `MLPCondition(in_dim=1, out_dim=model_dim, hidden_dims=[model_dim], act=SiLU, dropout=label_dropout)` over the normalized return, passed to the `DiscreteDiffusionSDE` as `nn_condition` with `classifier=None`. The masking — `fix_mask` clamping $\tau[0,:\text{obs\_dim}]$, `loss_weight` up-weighting the first action — is byte-for-byte the default's, because start-state inpainting and executed-first-action weighting have nothing to do with the guidance choice. (One harness wrinkle: `report_parameters` crashes on the tiny `MLPCondition`, its hardcoded top-K indexing running off the end of the parameter list, so I skip that call for the condition net and just print its total parameter count.) In training the classifier branch disappears: where the default ran `agent.update(x)` plus `agent.update_classifier(x, val)`, here I call only `agent.update(x, val)` — the same denoising step, now passed the normalized return as the condition, with the `label_dropout` inside the condition net randomly zeroing it to create the unconditional branch. I normalize the return by the per-env `DD_RETURN_SCALE`, and there is no separate classifier LR scheduler. At inference I keep the start-state-anchored prior, add `condition = ones(num_envs,1) * target_return`, draw a *single* sample per env — no `num_candidates` fan-out and no re-ranking, because CFG has no classifier log-probability to re-rank by — and call `agent.sample(prior, …, condition_cfg=condition, w_cfg=task.w_cfg, w_cg=0.0, temperature=temperature)`. The `w_cg=0.0` turns the classifier-guidance pathway fully off; `w_cfg` is the dial $w$ derived above (4.4 / 6.0 / 3.2 for hopper / walker2d / halfcheetah, well above 1 because a scalar return is a weak handle so I lean on the extrapolation). I read the action straight off the conditional sample, `act = traj[:,0,obs_dim:]`, clipped to $[-1,1]$, with the usual zero placeholders for the fixed print's `logp`/`idx`.

What this rung does *not* carry over from the general method, because the harness omits it: there is no candidate re-ranking — that is the default's CG move, and CFG's standard inference reads the conditional sample directly; there is no attribute composition, no constraint or skill conditioning, only a single `MLPCondition` over a scalar return, one `w_cfg`, one `target_return`. And unlike the Decision Diffuser, the diffusion target is the full $\text{obs}+\text{act}$ trajectory and actions are read straight off the array rather than recovered by inverse dynamics — that is the whole point of holding the backbone fixed. The delta from the previous rung is therefore a controlled subtraction: the same classifier-free conditioning mechanism, on the original `JannerUNet1d`-on-$\text{obs}+\text{act}$ `DiscreteDiffusionSDE` backbone instead of the `DiT1d`/state-only/inverse-dynamics pipeline. I expect hopper to land near the Decision Diffuser's 0.9345, walker2d to recover toward the high-0.7s if the architecture was the confound, and halfcheetah to stay flat near 0.41 — and whatever gap remains after that isolates the guidance mechanism itself for the final comparison against classifier guidance with re-ranking.

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
