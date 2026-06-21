The diffusion-BC floor came in exactly where the theory said it would: pure cloning landed at $0.49$ on hopper, $0.66$ on walker2d, $0.42$ on halfcheetah (means over seeds 42/123/456), and the seeds are tight, so this is not a variance story but a *ceiling* story. The actor does its one job well — it faithfully and reproducibly reproduces the behavior policy with no mode-averaging collapse — but the behavior policy is `medium`, a half-trained agent, and cloning it well just reproduces its mediocrity. The actor was given no reward signal at all, so its ceiling is $\mu$ by construction, and $0.49$ on hopper *is* $\mu$'s ceiling. The obvious next move is to let reward in and push the actor above $\mu$ toward the better-than-average actions the buffer demonstrably contains. The instant I reach for a value function, though, I hit the wall that defines offline RL: the buffer is fixed, so the moment I bootstrap $Q(s,a)\to r+\gamma\,Q(s',a')$ I must choose which $a'$ to plug in at the next state. If I take $a'$ from the policy I am improving — which by construction wants to deviate from $\mu$ — then $a'$ slides off the data support, the approximator returns a value for an action it has never seen, and those values come out too high far more often than too low. The backup carries that phantom optimism backward, the policy chases it, and the whole thing diverges. The real enemy is querying the value of out-of-distribution actions; I have to add reward *without* ever writing down $\max_{a'}Q(s',a')$, because that max is exactly the operator that reaches outside the data.

I propose **IDQL** — Implicit Diffusion Q-Learning — which learns value strictly in-sample and extracts the actor at inference, keeping the floor's diffusion-BC actor untouched. The cleanest way to refuse the OOD query is never to form it. Instead of maximizing over $a'$, estimate "the value of the best in-support action" from dataset actions alone by **expectile regression**. The $\tau$-expectile of a random variable is $\arg\min_m\,\mathbb{E}[\,|\tau-\mathbf{1}(x<m)|\,(x-m)^2\,]$: an asymmetric squared loss that for $\tau>0.5$ punishes $x<m$ less than $x>m$, so the minimizer is pulled above the mean, and as $\tau\to1$ it climbs toward the supremum. Apply it to the distribution of $Q(s,a)$ as $a$ ranges over $\mu(\cdot\mid s)$, and put it on a *separate* value net $V(s)$:
$$L_V(\psi)=\mathbb{E}_{(s,a)\sim D}\big[\,|\tau-\mathbf{1}(Q_{\text{targ}}(s,a)-V_\psi(s)<0)|\,(Q_{\text{targ}}(s,a)-V_\psi(s))^2\,\big],$$
$$L_Q(\theta)=\mathbb{E}_{(s,a,s')\sim D}\big[\,(r+\gamma(1-\text{done})\,V_\psi(s')-Q_\theta(s,a))^2\,\big].$$
A high expectile of that distribution is the value of the best action the data supports, recovered purely from in-sample actions, with the max done implicitly by the loss asymmetry — I never name the maximizing action, so I never query an OOD one. The split onto a separate $V$ is load-bearing: if I expectile-regressed $r+\gamma Q(s',a')$ directly, the asymmetry would also reward lucky *transitions*, confusing "this action is reliably good" with "this sample got lucky." Letting $V$ take the action-expectile and having $Q$ do an ordinary SARSA TD backup against $V(s')$ isolates the max over actions from the dynamics noise. This interpolates from SARSA at $\tau=0.5$ to support-constrained Q-learning as $\tau\to1$ — stable, multi-step, never touching an OOD action.

The half that actually drives the score is the extraction, and the subtlety is that I have a critic with no policy in sight. Which policy does this in-sample critic evaluate? I do not want to guess, so I derive it. Generalize the value loss to an arbitrary convex $f$ with $f'(0)=0$ (expectile is the case $f=|\tau-\mathbf{1}(u<0)|\,u^2$), and define $V^*(s)=\arg\min_V\mathbb{E}_{a\sim\mu}[f(Q(s,a)-V(s))]$. At the optimum the derivative in $V$ vanishes: $\mathbb{E}_{a\sim\mu}[f'(Q-V^*)]=0$. Convexity with $f'(0)=0$ means $f'$ has the sign of its argument, so $f'(x)=|f'(x)|\,x/|x|$, and substituting gives $\mathbb{E}_{a\sim\mu}\!\big[\frac{|f'(Q-V^*)|}{|Q-V^*|}(Q-V^*)\big]=0$. The factor $|f'(Q-V^*)|/|Q-V^*|$ is a nonnegative scalar weight; fold it into the sampling distribution as $\pi_{\text{imp}}(a\mid s)\propto\mu(a\mid s)\,\frac{|f'(Q-V^*)|}{|Q-V^*|}$ and the condition collapses to $\mathbb{E}_{a\sim\pi_{\text{imp}}}[Q(s,a)-V^*(s)]=0$, i.e. $V^*(s)=\mathbb{E}_{a\sim\pi_{\text{imp}}}[Q(s,a)]$. So $V^*$ is the *value of an implicit actor* $\pi_{\text{imp}}$ — a reweighting of the behavior policy. The in-sample critic is secretly an actor-critic, and for the expectile loss the weight is the strikingly simple two-valued $|\tau-\mathbf{1}(Q<V^*)|$: $\tau$ above the value, $1-\tau$ below, so the implicit actor broadens the good half of $\mu$ and shrinks the bad half.

That derivation hands me the fix almost directly, and tells me what *not* to do. The implicit actor is $\mu$ reweighted, and the reweighting of a multimodal $\mu$ is itself multimodal — which a unimodal Gaussian extraction (the classic AWR move of Peng et al. 2019; Nair et al. 2020) cannot represent; it would smear across the modes and put mass in the low-density valley where the OOD, over-valued actions live, throwing away the careful in-sample value learning at the last step. So I keep the expressive diffusion actor I already have. The naive instinct — retrain it with the advantage weights baked into the loss — is a known dead end: a high-capacity model trained with importance-weighted maximum likelihood raises the likelihood of all training points regardless of weight, so the $\exp(\beta A)$ skew washes out and I lose the very reweighting I wanted. The escape sits in the form $\pi_{\text{imp}}=\mu\,w/Z$: I do not have to bake the weighting into training. I train the actor to represent $\mu$ alone — pure diffusion BC, the exact thing I validated at the floor — and apply the critic weights at *inference* by importance resampling. At a state, draw $N$ candidates from the behavior model, score each with the critic, and resample one with probability proportional to the advantage weight. This is why IDQL is the natural rung above diffusion BC and not a different animal: it *reuses the floor's actor verbatim* and adds exactly the missing piece — a value function and an inference-time reweighting — without touching the actor's training. The decoupling (the critic never sees the actor during training; the actor never sees $Q$/$V$ during training) is the whole source of stability, and it is precisely what lets me keep the floor's clean, reproducible BC and just bolt value on top. Concretely the inference rule I adopt is the practical exponential-style one: draw $N$ candidates, compute advantage $A=Q-V$ for each, form $w=\text{softmax}(A\cdot\text{weight\_temperature})$ over the candidates, and resample one. This is the finite-sample realization of the exponential/KL implicit actor (the linex loss $f(u)=\exp(\alpha u)-\alpha u$, whose value is the log-partition of $\pi\propto\mu\exp(\alpha A)$), skewing toward high-advantage candidates while staying inside the support the behavior model sampled from.

Against the harness the literal edit is precise. The substrate's default builds a single `DQLCritic` (twin-Q) and trains the actor with `bc_loss + eta * q_loss`; IDQL needs a different critic shape — twin-Q *plus* a value net, with the expectile rule. The harness exposes exactly these (`IDQLQNet`, with `.both(obs,act)` and a `q_min`, and `IDQLVNet`) and a matching actor backbone `IDQLMlp`, so I import them rather than hand-rolling. I swap the actor backbone from `DQLMlp` to `IDQLMlp`, construct `iql_q`, a frozen `iql_q_target = deepcopy(iql_q)`, and `iql_v`, each with its own Adam optimizer and cosine scheduler. The training loop reads `obs, next_obs, act, rew, tml`. I update the critic on *every other* step — the behavior model is the harder fit and deserves at least as many updates as the in-sample critic, which converges quickly — doing first the expectile $V$ step (`v_loss = (|tau - 1((q-v)<0)| * (q-v)^2).mean()` with `q = iql_q_target(obs,act)`), then the $Q$ step against the bootstrapped `td_target = rew + discount*(1-tml)*iql_v(next_obs)` over both twin heads, then a Polyak update of the $Q$ target at $0.995$. On *every* step I take the weight-free diffusion BC step `actor.update(act, obs)["loss"]` — identical to the floor's actor update. Critically, IDQL drops the default's `eta * q_loss` term entirely: there is no $Q$-maximization gradient flowing into the actor; the actor is pure BC, and all value information enters only at inference. At inference the reranking comes back (the floor deleted it): I load the actor plus the critic, repeat each observation `num_candidates` times, sample that many actions, compute `adv = iql_q_target(obs,act) - iql_v(obs)` reshaped to `(-1, num_candidates, 1)`, `w = softmax(adv * weight_temperature, dim=1)`, normalize, and resample one per env with `torch.multinomial`, with `num_candidates = 50` fixed so reranking methods see equal compute. I expect IDQL to beat the floor on every environment, with the largest gain where $\mu$ has the most exploitable spread, but to land clearly *below* the next rung: its actor is pure BC, so the critic can only pick the best of $N$ samples from $\mu$ — it can never surface an action $\mu$ would not draw. The full scaffold module — the IDQL critic construction, the decoupled expectile/BC training loop, and the advantage-reranking inference block — follows.

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
