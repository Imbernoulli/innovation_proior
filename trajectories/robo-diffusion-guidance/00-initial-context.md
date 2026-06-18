## Research question

A trajectory-level diffusion planner for offline control is fixed. It diffuses whole
state-action trajectories at once, anchors the plan at the current observation by inpainting,
and reads off the first action in a receding-horizon loop. The one thing being designed is the
**guidance mechanism** — how the reverse diffusion process is steered toward high-return
behavior. Everything else (dataset, environment loop, backbone width, diffusion-step budget,
evaluation) is frozen. The question is narrow on purpose: not "how do I build a planner" but
"given this planner, what is the right way to condition or guide its sampling so the plans it
draws are good plans, not merely plausible ones."

## Prior art before the first rung (trajectory-generation lineage)

The substrate the first rung edits — a temporal-U-Net diffusion model over trajectories — is
itself the resolution of a line of offline-control methods. These precede the ladder; the fixed
substrate below is what they converged to.

- **Autoregressive single-step dynamics models for planning (PETS, Chua et al. 2018; ensemble
  MB-RL, Wang et al. 2019).** Fit `ŝ_{t+1}=f(s_t,a_t)`, then plan by rolling it forward under
  candidate action sequences and scoring them (CEM / random shooting). Gap: trained for
  *single-step* accuracy but used for *multi-step* rollouts, so error compounds over the horizon,
  and a strong optimizer exploits exactly the off-manifold regions where the model is confidently
  wrong — the "optimal" plan comes back as an adversarial example, not a trajectory.
- **Offline TD value learning (CQL, IQL, BCQ; circa 2019–2021).** Estimate `Q*` with a Bellman
  backup whose `max_{a'}` propagates value backward and stitches sub-optimal segments into
  something better than any single logged run. Gap: sits on the deadly triad (function
  approximation + bootstrapping + off-policy data) and, offline, over-estimates value on
  out-of-distribution actions with no environment to correct it, forcing a per-task in-distribution
  penalty.
- **Sequence-model decision making (Decision Transformer / Trajectory Transformer, 2021).**
  Tokenize a trajectory and fit an autoregressive Transformer over interleaved states and actions,
  generate left-to-right conditioned on a target return. Gap: decision-making is *anti-causal* —
  the action now depends on where you are trying to end up — so a strictly forward decoder pushes
  conditioning the wrong way, and the left-to-right factorization re-imports compounding error.
- **Denoising diffusion (Sohl-Dickstein et al. 2015; Ho, Jain & Abbeel 2020; cosine schedule,
  Nichol & Dhariwal 2021).** A fixed forward process corrupts data to noise; a learned reverse
  process denoises it back, training collapses to predicting the added noise at a random level,
  and sampling is iterative — a sequence of denoising steps rather than one forward pass. The
  reverse transitions are Gaussian and *modifiable*: the sampling distribution can be tilted by
  external information at each step, and constraints imposed by clamping known coordinates
  (inpainting). This is the engine the substrate is built on.

## The fixed substrate

A trajectory-level diffusion planner is frozen and must not be touched in spirit. It diffuses an
array `τ = [s_0 … s_{H-1}; a_0 … a_{H-1}]` of width `obs_dim + act_dim` and length `H`: a
`JannerUNet1d` 1-D temporal U-Net (`model_dim=32`, `kernel_size=5`, no attention) wrapped by a
`DiscreteDiffusionSDE` (`diffusion_steps=20`, `solver=ddpm`, `predict_noise=False`,
`ema_rate=0.9999`). The start state is imposed as inpainting: a `fix_mask` clamps `τ[0,:obs_dim]`
to the current observation at every reverse step and zeros its training loss; the first action
`τ[0,obs_dim:]` is up-weighted (`action_loss_weight=10`) because only it is executed. Training
runs `diffusion_gradient_steps=100000` at `batch_size=256` with a cosine-annealed LR;
inference runs `sampling_steps=20` over `num_envs=10` envs × `num_episodes=10`. The dataset,
D4RL normalization, environment loop, seeds, and final D4RL scoring are all fixed.

The loop also exposes the conditioning hooks every guidance method fills: a `CumRewClassifier`
head (`HalfJannerUNet1d`) for classifier guidance; an `nn_condition` slot for classifier-free
conditioning; the per-task config scalars `w_cg`, `w_cfg`, `target_return`; and the
`agent.sample(..., w_cg=, w_cfg=, condition_cfg=)` call where the guided epsilon is formed.

## The editable interface

Exactly one file is editable — `CleanDiffuser/pipelines/custom_guidance.py`, created from
`edits/custom_template.py`. The editable regions cover the imports, the network + agent setup
(backbone, classifier/condition, masking, the `DiscreteDiffusionSDE`), the training loop (how
`agent.update(...)` is called, return normalization, label dropout), the inference setup, the
prior + condition initialization, and the action-sampling block (`w_cg`, `w_cfg`,
`condition_cfg`, candidate re-ranking). Every method on the ladder is a fill of this same file.

The starting point is the scaffold default — classifier guidance: a `CumRewClassifier` is trained
alongside the diffusion model, and at sample time `num_candidates=64` trajectories are drawn per
env-step and re-ranked by the classifier's log-probability. Each later method replaces exactly
the relevant editable regions and nothing else.

```python
# DEFAULT fill of CleanDiffuser/pipelines/custom_guidance.py — classifier guidance (CG)
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
    os.makedirs(save_path, exist_ok=True)

    # ---------------------- Dataset (FIXED) ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ---------------- Network + Agent (EDITABLE) -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult, timestep_emb_type="positional",
        attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier,
        ema_rate=args.ema_rate, device=args.device,
        diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ---------------------- Training (EDITABLE) ----------------------
    if args.mode == "train":
        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)
        agent.train()
        n_gradient_step = 0
        for batch in loop_dataloader(dataloader):
            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)
            x = torch.cat([obs, act], -1)
            agent.update(x)                       # diffusion step
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                agent.update_classifier(x, val)   # reward-predictor step
                classifier_lr_scheduler.step()
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")
            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Inference (EDITABLE) ----------------------
    elif args.mode == "inference":
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")
        agent.eval()
        env_eval = gym.vector.make(args.task.env_name, args.num_envs)   # FIXED
        normalizer = dataset.get_normalizer()
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        for i in range(args.num_episodes):
            obs, cum_done, t = env_eval.reset(), 0., 0
            while not np.all(cum_done) and t < 1001:
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior.repeat(args.num_candidates, 1, 1), solver=args.solver,
                    n_samples=args.num_candidates * args.num_envs, sample_steps=args.sampling_steps,
                    use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)
                logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                idx = logp.argmax(0)              # re-rank candidates by classifier log-prob
                act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                      idx, torch.arange(args.num_envs), 0, obs_dim:]
                obs, rew, done, info = env_eval.step(act.clip(-1., 1.).cpu().numpy())   # FIXED
                t += 1
```

## Evaluation settings

Three D4RL MuJoCo environments — **hopper-medium-v2**, **walker2d-medium-v2**,
**halfcheetah-medium-v2** — each at seed 42, 10 envs × 10 episodes. The primary metric is the
**D4RL normalized score** per environment (higher is better); the task aggregator is the
**geometric mean** across the three environments. Per-environment training wall-clock is reported
alongside. Top-level hyperparameters (`diffusion_gradient_steps=100000`, `model_dim=32`,
`diffusion_steps=20`, `sampling_steps=20`, `solver=ddpm`) are fixed for every method.
