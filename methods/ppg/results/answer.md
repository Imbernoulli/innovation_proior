# PPG — Phasic Policy Gradient

## Problem

In on-policy actor-critic, sharing parameters between the policy and value function lets them share
useful features (important on high-dimensional inputs) but causes two problems: the value loss can
**interfere** with the policy through the shared trunk, and a shared network forces policy and value to
be trained on the **same data with the same sample reuse**. Separate networks fix both but lose the
feature sharing. The goal: keep feature sharing while decoupling the two trainings — independent
objectives, independent sample reuse — exploiting the fact that value optimization tolerates much
higher sample reuse than policy optimization.

## Key idea

Separate policy and value training **in time** into two alternating phases.

- **Policy phase** (`N_π` PPO iterations, on *disjoint* policy and value networks): optimize the PPO
  clipped surrogate (plus entropy bonus) on the policy net for `E_π` epochs and the value MSE on the
  true value net for `E_V` epochs, with GAE advantages/targets; because the nets are disjoint, no value
  gradient touches the policy and `E_π`, `E_V` are independent (default `E_π = 1`). Stash all
  `(s_t, V̂^targ_t)` into a buffer `B`.
- **Auxiliary phase** (every `N_π` updates, `E_aux` epochs over `B`): the policy network carries an
  **auxiliary value head** sharing its trunk. Optimize
  `L^joint = L^aux + β_clone·Ê[KL(π_old(·|s), π_θ(·|s))]`, where `L^aux = ½·Ê[(V_θπ(s) − V̂^targ)²]`
  distills value-predictive features into the policy trunk and the KL **behavioral-cloning** term holds
  the policy's outputs fixed (so the representation changes but the policy does not). Also re-fit the
  true value net on `B` (`L^value`), which is where the value gets its high sample reuse (raise `E_aux`,
  not `E_V`). Targets `V̂^targ` are fixed throughout the auxiliary phase.

The auxiliary phase runs infrequently (`N_π = 32`) because each one mildly perturbs the policy. The
policy objective is swappable (PPO clip or a fixed-weight KL penalty `L^KL = Ê[−Â_t r_t + β_π KL]`), and
a single-network variant ("detach") reduces the parameter cost by detaching the true-value path from
the shared policy features; in the archived implementation, shared features in this variant are still
trained during the auxiliary phase through the auxiliary value head.

## Algorithm

Policy phase losses (GAE: `Â_t = Σ_l (γλ)^l δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`,
`V̂^targ = V + Â`):

  `L^clip = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε)Â_t)]`,  optimize `L^clip + β_S·S[π]` (policy net),
  `L^value = Ê_t[½(V(s_t) − V̂^targ_t)²]` (true value net).

Auxiliary phase: `L^joint = ½Ê[(V_θπ(s) − V̂^targ)²] + β_clone·Ê[KL(π_old, π_θ)]` (policy net) and
`L^value` (true value net). With the default dual architecture these losses have disjoint parameter
paths; the implementation can sum compatible losses in one optimizer step when the epoch counts match.

Hyperparameters: `N_π = 32`, `E_π = 1`, `E_V = 1`, `E_aux = 6`, `β_clone = 1`; `γ = .999`, `λ = .95`,
`ε = .2`, `β_S = .01`, vfcoef `.5`; rollout 256 steps, 8 minibatches/epoch (policy phase), 16
minibatches/aux-epoch; Adam lr `5e-4`; reward normalization on; arch `dual` (default).

## Code

```python
import torch as th
import torch.nn as nn
from torch import distributions as td


class PhasicValueModel(nn.Module):
    def __init__(self, ob_space, ac_space, enc_fn, arch="dual"):       # "dual" | "detach" | "shared"
        super().__init__()
        self.arch = arch
        self.detach_value_head = (arch == "detach")
        self.pi_enc = enc_fn(ob_space)
        if arch == "dual":
            self.vf_enc = enc_fn(ob_space)
        feat = self.pi_enc.outsize
        self.pi_head = nn.Linear(feat, ac_space.size)
        self.true_vf_head = nn.Linear(self._vf_feat(), 1)
        self.aux_vf_head = nn.Linear(feat, 1)                          # on the policy trunk
        self.make_distr = lambda logits: td.Categorical(logits=logits)

    def _vf_feat(self):
        return self.vf_enc.outsize if self.arch == "dual" else self.pi_enc.outsize

    def forward(self, ob):
        pi_x = self.pi_enc(ob)
        pd = self.make_distr(self.pi_head(pi_x))
        if self.arch == "dual":
            vf_x = self.vf_enc(ob)
        else:
            vf_x = pi_x.detach() if self.detach_value_head else pi_x
        vpred_true = self.true_vf_head(vf_x)[..., 0]
        vpred_aux = self.aux_vf_head(pi_x)[..., 0]
        return pd, vpred_true, vpred_aux

    def compute_aux_loss(self, vpred_aux, vpred_true, vtarg):
        return {
            "vf_aux":  0.5 * ((vpred_aux  - vtarg) ** 2).mean(),
            "vf_true": 0.5 * ((vpred_true - vtarg) ** 2).mean(),
        }


def compute_gae(reward, vpred, first, gamma, lam):
    nenv, nstep = reward.shape
    adv = th.zeros(nenv, nstep); lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        delta = reward[:, t] + notlast * gamma * vpred[:, t + 1] - vpred[:, t]
        adv[:, t] = lastgaelam = delta + notlast * gamma * lam * lastgaelam
    vtarg = vpred[:, :-1] + adv
    return adv, vtarg


def normalize_adv(adv, eps=1e-8):
    return (adv - adv.mean()) / (adv.var(unbiased=False).sqrt() + eps)


def ppo_losses(model, mb, clip_param, ent_coef, vf_coef=0.5, kl_penalty=0.0):
    pd, vpred_true, _aux = model(mb["ob"])
    newlogp = pd.log_prob(mb["ac"])
    logratio = newlogp - mb["logp"]
    ratio = th.exp(logratio)
    pg = th.max(-mb["adv"] * ratio,
                -mb["adv"] * th.clamp(ratio, 1 - clip_param, 1 + clip_param)).mean()
    approx_kl_penalty = kl_penalty * 0.5 * (logratio ** 2).mean()
    pi_loss = pg - ent_coef * pd.entropy().mean() + approx_kl_penalty
    vf_loss = vf_coef * ((vpred_true - mb["vtarg"]) ** 2).mean()
    return pi_loss, vf_loss


def aux_train(model, buffer, opt, beta_clone, vf_true_weight):
    for mb in buffer.minibatches():
        pd, vpred_true, vpred_aux = model(mb["ob"])
        losses = model.compute_aux_loss(vpred_aux, vpred_true, mb["vtarg"])
        pol_distance = td.kl_divergence(mb["oldpd"], pd).mean()
        loss = losses["vf_aux"] + beta_clone * pol_distance + vf_true_weight * losses["vf_true"]
        opt.zero_grad(); loss.backward(); opt.step()


def learn(venv, model, hp):
    ppo_opt = th.optim.Adam(model.parameters(), lr=hp["lr"])
    vf_opt = ppo_opt if hp["E_pi"] == hp["E_v"] else th.optim.Adam(model.parameters(), lr=hp["lr"])
    aux_opt = th.optim.Adam(model.parameters(), lr=hp["aux_lr"])
    while True:
        buffer = []
        for _ in range(hp["n_pi"]):                                    # policy phase
            rollouts = collect_rollouts(venv, model, hp["nstep"])
            adv, vtarg = compute_gae(rollouts["reward"], rollouts["vpred"],
                                     rollouts["first"], hp["gamma"], hp["lam"])
            rollouts["adv"], rollouts["vtarg"] = normalize_adv(adv), vtarg
            if hp["E_pi"] == hp["E_v"]:
                for _ in range(hp["E_pi"]):
                    for mb in minibatches(rollouts, hp["nminibatch"]):
                        pi_loss, vf_loss = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"],
                                                      hp["vf_coef"], hp.get("kl_penalty", 0.0))
                        ppo_opt.zero_grad(); (pi_loss + vf_loss).backward(); ppo_opt.step()
            else:
                for _ in range(hp["E_v"]):
                    for mb in minibatches(rollouts, hp["nminibatch"]):
                        _, vf_loss = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"],
                                                hp["vf_coef"], hp.get("kl_penalty", 0.0))
                        vf_opt.zero_grad(); vf_loss.backward(); vf_opt.step()
                for _ in range(hp["E_pi"]):
                    for mb in minibatches(rollouts, hp["nminibatch"]):
                        pi_loss, _ = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"],
                                                hp["vf_coef"], hp.get("kl_penalty", 0.0))
                        ppo_opt.zero_grad(); pi_loss.backward(); ppo_opt.step()
            buffer.append({k: rollouts[k] for k in ("ob", "vtarg")})

        for seg in buffer:                                             # snapshot π_old
            with th.no_grad():
                seg["oldpd"], _, _ = model(seg["ob"])
        for _ in range(hp["n_aux_epochs"]):                            # auxiliary phase
            aux_train(model, BufferView(buffer), aux_opt,
                      hp["beta_clone"], hp["vf_true_weight"])

# hp = dict(n_pi=32, E_pi=1, E_v=1, n_aux_epochs=6, beta_clone=1, vf_true_weight=1,
#           gamma=.999, lam=.95, nstep=256, nminibatch=8, ent_coef=.01,
#           clip_param=.2, vf_coef=.5, kl_penalty=0.0, lr=5e-4, aux_lr=5e-4)
```
