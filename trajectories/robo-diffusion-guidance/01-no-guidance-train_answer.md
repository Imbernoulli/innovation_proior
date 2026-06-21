The research question is narrow on purpose — design the way the reverse diffusion process is steered toward good behavior — and the only honest way to learn whether *any* steering helps is to first measure the planner with steering switched entirely off. So I begin at the floor: an unconditional trajectory diffusion model that learns what plausible trajectories of this system look like and samples one, with no classifier, no return conditioning, no candidate re-ranking. It is the reference point a meaningful guidance strategy must beat, and the size of the gap above it is the entire quantity this task is about.

The thing worth being precise about is *why* a diffusion planner is the right substrate to put a floor under, because that justifies measuring the unguided sampler as an object in its own right. The textbook offline recipe is to fit a single-step dynamics model $\hat s_{t+1}=f(s_t,a_t)$ and hand it to a trajectory optimizer that searches for the action sequence maximizing summed reward. That decomposition behaves badly with learned models in a specific, structural way: to plan over a horizon I roll the model forward autoregressively, every prediction carries error, each prediction is the input to the next, so errors compound and a long rollout drifts somewhere the real system would never go. Worse, a strong differentiable optimizer walks the plan straight into the regions where $f$ is confidently wrong, because that is where the model reports high reward — the "optimal" plan comes back as an adversarial example. The reflex fixes all cost something I want to keep: weakening the planner throws away the long-horizon reasoning; a fully model-free conservative value function discards trajectory structure and makes future-goal conditioning unnatural; a left-to-right autoregressive Transformer re-imports compounding error and pushes conditioning the wrong way, since decision-making is anti-causal — the action now depends on where I am trying to end up. What I actually want, before reaching for machinery, is for producing a plan to *be* sampling from a model of trajectories (so there is no separate exploitable search), for the whole plan to be produced at once (so neither rollout error nor causal ordering bites), and for the model to learn *first* — before any notion of reward enters — just "what do plausible trajectories of this system look like," a model that stays on the data manifold by construction.

The method that delivers exactly this is unconditional denoising diffusion over whole trajectories. Take a clean trajectory window $x^0$ and define a forward process that slowly destroys it, $q(x^i\mid x^{i-1})=\mathcal N\!\big(x^i;\sqrt{1-\beta_i}\,x^{i-1},\beta_i I\big)$, for a small prespecified schedule $\beta_i$. The shrink-by-$\sqrt{1-\beta_i}$ keeps the per-step variance controlled, so the signal stays at unit scale all the way down. Writing $\alpha_i=1-\beta_i$ and $\bar\alpha_i=\prod_{j\le i}\alpha_j$, composing steps telescopes to a closed form, $q(x^i\mid x^0)=\mathcal N\!\big(x^i;\sqrt{\bar\alpha_i}\,x^0,(1-\bar\alpha_i)I\big)$, so I can jump to any noise level in one shot, $x^i=\sqrt{\bar\alpha_i}\,x^0+\sqrt{1-\bar\alpha_i}\,\varepsilon$ — and that is what makes training cheap. The model is the reverse chain $p_\theta(x^{i-1}\mid x^i)=\mathcal N\!\big(x^{i-1};\mu_\theta(x^i,i),\Sigma^i\big)$ started from $\mathcal N(0,I)$, fit by maximizing a variational bound. Conditioning the forward posterior on $x^0$ reduces that bound to a sum of Gaussian KLs, each matching $p_\theta$ to the analytic posterior $q(x^{i-1}\mid x^i,x^0)$; with the reverse variance fixed the KL is just the squared distance of the means, and substituting $x^0=(x^i-\sqrt{1-\bar\alpha_i}\,\varepsilon)/\sqrt{\bar\alpha_i}$ makes the posterior mean a function of $x^i$ and $\varepsilon$ alone. So instead of predicting the mean I have the network predict the noise $\varepsilon_\theta(x^i,i)$, and the loss collapses to plain denoising regression,
$$L_{\text{simple}}=\mathbb E_{i,x^0,\varepsilon}\big\|\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_i}\,x^0+\sqrt{1-\bar\alpha_i}\,\varepsilon,\,i)\big\|^2,$$
with $i$ uniform. The deeper reason $\varepsilon$ is the clean target is that $\varepsilon_\theta$ estimates the gradient of the log-density of the noised data up to scale — the score — so the reverse chain is Langevin-like ascent on the data density, and it lands on high-density, in-distribution trajectories. That is precisely why the unguided sampler is a meaningful object: it is a sampler of dynamically plausible plans.

What makes this a *planner* rather than a generic generator is the choice of $x$. A trajectory is a sequence over planning time of states $s_t$ and actions $a_t$. Modeling states and predicting actions separately would reproduce the model-then-controller split I am trying to kill, so I stack them into one object and generate them jointly — $\tau=[s_0\,\dots\,s_{H-1};\,a_0\,\dots\,a_{H-1}]$, width $\text{obs\_dim}+\text{act\_dim}$, length $H$ — and diffuse the whole array at once. The denoising chain refines all timesteps together, so anti-causal conditioning is a non-issue and the controller is trained under the same objective as the state prediction. The architecture respects that the two axes are not symmetric: along the horizon axis the data is a translation-equivariant, local time series, so I convolve there; along the feature axis the coordinates are heterogeneous, so I treat them as channels. That gives a 1-D temporal convolution over time, stacked into a U-Net that downsamples and upsamples along the horizon so deep blocks see the whole plan — and because the sampler composes many local denoising steps, local consistency applied repeatedly becomes global coherence. This is the `JannerUNet1d` the substrate fixes: kernel size 5, no attention (the trajectories are low-dimensional enough that convolutional locality plus iteration suffices), with $H=32$ a power of two for the repeated downsampling. The chain is short — 20 steps, far fewer than the 1000 used for images, since a locomotion window is low-dimensional and smooth — so the schedule matters more, and a cosine schedule spreads the signal-to-noise ratio gracefully over it.

There is one piece of conditioning even an *unconditional* planner cannot do without, and it must not be confused with guidance. Every plan must start at the current state: standing at observation $s$, the sampled plan must have $s_0=s$. This is a hard constraint on a subset of the array's coordinates — exactly inpainting — so I clamp $\tau[0,:\text{obs\_dim}]$ to the observation at *every* reverse step (the `fix_mask`) and zero its training loss, since the network is never asked to predict what is given. This is a start-state constraint, not reward — no value, no return, no preference for high-return plans enters here. I also up-weight the loss on the first action $\tau[0,\text{obs\_dim}:]$ by a factor of 10, because in receding-horizon control only the first action is executed and the rest of the plan is lookahead (`action_loss_weight=10`). To match the fixed harness I take the $x^0$-prediction route (`predict_noise=False`), recovering $\varepsilon$ from the predicted clean trajectory in closed form at each reverse update, and an EMA of the weights (decay 0.9999) gives a stable sampler.

Concretely, this rung strips the substrate's default classifier-guidance pathway and nothing else. In the network setup I drop `nn_classifier` / `CumRewClassifier` and pass `classifier=None` to the `DiscreteDiffusionSDE`, keeping the identical `JannerUNet1d` backbone, `fix_mask`, `loss_weight`, and SDE. In training I remove the classifier-update branch and call only `agent.update(x)` — the diffusion denoising step, nothing else, since `val` is no longer consumed. At inference I load only the diffusion checkpoint, keep the start-state-anchored Gaussian prior with no condition, draw a *single* sample per env (no `num_candidates` fan-out) with `w_cg=0.0, w_cfg=0.0`, and read the first action straight off the trajectory, `act = traj[:,0,obs_dim:]`, clipped to $[-1,1]$. The fixed post-sample print line still references `logp`/`idx` from the re-ranking path that no longer exists, so I define harmless zero placeholders. The result is a single dynamically plausible plan, read off, executed, replanned — the same manifold model as the default with the entire reward pathway removed. What I expect, and what forces step 2, is that the sampler lands on plausible plans, not good ones: on `*-medium-v2` data the unconditional distribution is dominated by mediocre behavior, so a *typical* plausible plan is mediocre, most visibly on the unforgiving hopper.

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
