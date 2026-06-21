The controlled CFG run resolved the confound and pointed straight at the last move. Classifier-free guidance on the original `JannerUNet1d`-on-$\text{obs}+\text{act}$ backbone scored 0.9207 on hopper, 0.7620 on walker2d, 0.4129 on halfcheetah, geometric mean 0.6616, just above the Decision Diffuser's 0.6571. Walker2d is the answer I was after: 0.7620 versus 0.6972, a clean six-point gain from doing nothing but holding the backbone fixed — so the soft walker2d under the Decision Diffuser was the *architecture*, not the guidance. Halfcheetah sat at 0.4129, essentially the floor's 0.4115, confirming that on the dense-reward medium policy guidance of any flavor leaves almost no headroom. But each step from the last two has been small, and the geometric mean has climbed only 0.4763 → 0.6571 → 0.6616. The thing every CFG rung lacks is a way to *reject a bad draw*: CFG reads a *single* conditional sample per env-step and executes its first action, and on `*-medium` data that conditional distribution is still a noisy multimodal cloud even after conditioning. A single draw has no scalar to compare candidates by, and the place single-draw variance should bite is precisely where the headroom looks zero.

So I propose returning to the substrate's own default — classifier guidance with candidate re-ranking — and I want to derive the steering term from scratch, because the derivation is what makes the re-ranking asset visible. The diffusion model gives a sampler of plausible trajectories; I want to bias it toward high-return ones. Cast "high return" as conditioning on an optimality event $O$ and ask for $p(z\mid O)$. By Bayes the conditional reverse transition factors as $p(z_{t-1}\mid z_t,O)\propto p(z_{t-1}\mid z_t)\,p_\phi(O\mid z_{t-1})$ — the unconditional transition times a predictor likelihood. The unconditional transition is the Gaussian $\mathcal N(\mu,\Sigma)$ my reverse step already produces. Take the log of the predictor and Taylor-expand around the transition mean $\mu$, $\log p_\phi(O\mid z)\approx\log p_\phi(O\mid\mu)+(z-\mu)^\top g$ with $g=\nabla_z\log p_\phi(O\mid z)\big|_{z=\mu}$ (the curvature is small relative to $\Sigma$ when the step is small). Then the product $\mathcal N(z;\mu,\Sigma)\,\exp\big((z-\mu)^\top g\big)$ is, completing the square, again a Gaussian — same covariance, mean shifted to $\mu+\Sigma g$. So conditioning on optimality shifts each reverse step by $\Sigma\,\nabla_z\log p_\phi(O\mid z)$: sample from the unconditional step but nudge its mean up the predictor's log-gradient. In the epsilon parameterization the SDE uses (epsilon is the negative score up to $\sigma$), this is
$$\hat\varepsilon=\varepsilon_\theta(z)-w_{cg}\,\sigma\,\nabla_z\log p_\phi(\text{return}\mid z),$$
and the whole guided chain follows the score of $p(z)\,p_\phi(\text{return}\mid z)^{w_{cg}}$, where $w_{cg}$ raises the predictor to a power and sharpens onto its high-return modes. So classifier guidance trains a separate predictor on noised trajectories — here a cumulative-reward predictor, `CumRewClassifier` with a `HalfJannerUNet1d` head — and tilts each reverse step by exactly that shifted mean.

I spent the CFG derivation listing why I dislike this — a second trained network, an offline value function with OOD over-estimation, a guidance direction that is literally a classifier gradient — but the numbers force me to weigh those objections against what classifier guidance *buys* that CFG structurally cannot. The classifier produces a scalar $\log p_\phi(\text{return}\mid z)$ for any trajectory: not just a steering gradient, but a *score I can rank by*. That is the asset. CFG has no such scalar — its conditional sample is what it is, with nothing to compare candidates against. So classifier guidance uniquely enables the inference protocol the CFG rungs could not: draw many candidates and keep the one the predictor scores highest. The planning-as-inpainting structure makes the re-ranking even more natural. The model is anchored at the current state by clamping $\tau[0,:\text{obs\_dim}]$ throughout the chain, so every candidate starts at the same true observation and differs only in the future it imagines — they are genuinely comparable plans from the same state, exactly the condition under which ranking by a single scalar is meaningful. And because the executed action is only $\tau[0,\text{obs\_dim}:]$, the first action of the winning plan, re-ranking does not need the whole trajectory to be globally optimal — only the *first action* of the highest-scored plan to be good — and receding-horizon replanning corrects any later drift. Selection over candidates and receding-horizon control compose cleanly: draw many futures from the anchored state, keep the best-scored one, execute its first action, replan.

Concretely the protocol is this. At each env-step I replicate the start-state-anchored prior `num_candidates=64` times, run the guided reverse chain to get 64 trajectories per env, and the sampler returns `log["log_p"]` — the classifier's accumulated log-probability per candidate. I sum it over the trajectory, take `argmax` over the 64 candidates per env, and execute the first action of the winning plan. So even where the guided sampler's *mean* draw is mediocre, the max over 64 draws lands on a genuinely high-return plan, and the predictor's ranking is doing exactly the selection a single CFG draw cannot. This rung *is* the substrate's unmodified default — the only rung that edits nothing. The diffusion model is the same `JannerUNet1d` on $\text{obs}+\text{act}$ the CFG rung used (so the same-backbone comparison that made the CFG gain interpretable is preserved), trained the same way; the only additions over CFG are the `CumRewClassifier` (the `HalfJannerUNet1d` head trained alongside the diffusion model by `agent.update_classifier(x, val)`) and the 64-candidate fan-out with `argmax` re-ranking at inference. The guidance weights are the per-env `w_cg` the config ships — 0.3 / 0.007 / 0.0001 — tiny compared to the CFG `w_cfg` values (4.4 / 6.0 / 3.2), which is itself revealing: classifier guidance leans only lightly on the gradient nudge because the heavy lifting is done by the *re-ranking*, not by sharpening the score. The halfcheetah `w_cg` of 0.0001 is almost off — on the dense task the guidance barely steers and the candidate selection carries it.

What this rung does *not* do is the contrast with everything above: there is no return conditioning on the diffusion model itself — `nn_condition=None`, `w_cfg` unused — so the diffusion model is the same unconditional manifold model the floor used, and the *only* place reward enters is the classifier (steering) and the re-ranking (selection). That is the cleanest statement of the substrate's bet: keep the manifold model reward-agnostic and bolt reward on entirely through a separate predictor used both to steer and to rank. The cost is the second network and 64× the inference compute per env-step — this is the slowest of the four rungs — but the CFG ceiling says that cost is buying the thing CFG could not: per-step selection. The delta from the previous rung is precisely the re-ranking that was missing: where CFG drew one conditional sample and executed it, this rung draws 64 classifier-guided candidates and executes the best, with a trained reward predictor supplying both the steering gradient and the ranking score. I expect hopper to push into the mid-0.96s above both prior rungs, walker2d to hold near 0.76, and — the falsifiable bet — halfcheetah to finally move off the ~0.41 every other rung was stuck at, selection extracting the last available headroom where steering could not. If this rung is the strongest, the verdict is that on this planner the right way to guide is not a cleverer score tilt but *generate-many-and-select*, and the scalar a trained reward predictor gives for ranking is worth more than the gradient it gives for steering.

```python
# CleanDiffuser/pipelines/custom_guidance.py — default fill (classifier guidance + re-ranking)
import os

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
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
    # Network + Agent Setup (JannerUNet1d + CumRewClassifier)
    # ============================================================================
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ============================================================================
    # Training (diffusion + reward classifier)
    # ============================================================================
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)
            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                classifier_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "finetune":
        pass

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")
        agent.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)   # FIXED
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)

        for i in range(args.num_episodes):

            env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # ----- Action Sampling: 64 candidates, classifier guidance + re-ranking -----
                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior.repeat(args.num_candidates, 1, 1),
                    solver=args.solver,
                    n_samples=args.num_candidates * args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)

                logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                idx = logp.argmax(0)
                act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                      idx, torch.arange(args.num_envs), 0, obs_dim:]
                act = act.clip(-1., 1.).cpu().numpy()

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
