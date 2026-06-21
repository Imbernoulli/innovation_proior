Every on-policy actor-critic algorithm has to make one unglamorous architectural decision — does the policy $\pi$ and the value function $V$ share a network, or do they live in two separate networks? — and both answers are unsatisfying. A shared trunk is attractive because on high-dimensional inputs like images the features that predict the value are largely the same features that select a good action, so one representation serving both is far more efficient than learning two; on visually rich benchmarks sharing is not merely convenient but necessary, and separate networks simply do worse there. Yet sharing forces two compromises. First, a single set of parameters is now pulled by two different gradients — the policy-gradient and the value-regression gradient — so a relative weight $\text{vfcoef}\cdot L^{\text{value}}$ must be chosen against $L^{\text{clip}}$, and however that weight is tuned the two objectives can *interfere*: a step that improves value prediction can degrade the policy, and they fight over the shared trunk. Second, and more subtly, a shared trunk all but forces $\pi$ and $V$ to be trained on the *same data with the same number of epochs* — if I do three optimization passes over a batch, I do three for both, an arbitrary coupling I never chose. Separate networks remove both problems — the value gradient never touches the policy's parameters, and each can be given whatever sample reuse I want — but they discard exactly the feature sharing that was doing real work. So the choice is between feature sharing on one side and non-interference-plus-independent-reuse on the other, and I want all three at once.

Before chasing that, I want to settle whether independent reuse even matters, because if value and policy want the same number of epochs anyway the whole complaint evaporates. The nagging fact is that extra epochs over a rollout help in PPO, but in a shared net an "epoch" does two things at once — extra *policy* optimization and extra *value* optimization, confounded — so I cannot even tell which one wants the extra passes. That confounding is itself the tell: the architecture is hiding the measurement. My bet is that the policy wants very *few* epochs and the value wants *many*. The PPO policy update is a first-order, trust-region-like surrogate that is only valid near the current policy, so hammering it for many epochs marches $\pi$ into a region where the clipped surrogate lies — one policy pass should be near-optimal. The value function, by contrast, is plain regression onto returns, which tolerates being fit hard for many passes with no policy-collapse risk. If that is right, PPO's apparent need for three epochs is really the *value* being under-trained, with the extra epochs smuggling in value fitting through the shared loss.

I propose Phasic Policy Gradient (PPG). The core move is to stop treating this as a single joint optimization and instead separate the two trainings *in time*, alternating a policy phase and an auxiliary phase. In the policy phase I run $N_\pi$ iterations of PPO on *disjoint* policy and value networks: collect rollouts under the current $\pi$, form GAE advantages and targets with $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, and $\hat V^{\text{targ}}_t = V(s_t) + \hat A_t$, then optimize the policy net for $E_\pi$ epochs on the clipped surrogate plus an entropy bonus,
$$L^{\text{clip}} = \hat{\mathbb{E}}_t\big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat A_t\big)\big],\qquad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)},$$
and optimize the *true* value net for $E_V$ epochs on $L^{\text{value}} = \hat{\mathbb{E}}_t[\tfrac12(V(s_t) - \hat V^{\text{targ}}_t)^2]$. Because the two nets are disjoint here, no value gradient ever touches the policy's parameters and $E_\pi$ and $E_V$ are finally independent — I set $E_\pi = 1$ to take one conservative trust-region pass and stop before the surrogate is overused. Along the way I stash every $(s_t, \hat V^{\text{targ}}_t)$ into a buffer $B$.

But if policy and value are disjoint during the policy phase, the feature sharing has gone missing — what I have so far is just separate-net PPO with decoupled epochs, which has the non-interference and the independent reuse but none of the sharing. The sharing is the job of the second phase, and it needs somewhere to land. So I give the policy network an extra head, an auxiliary value head $V_{\theta_\pi}$ that shares all of the policy trunk's parameters except its own final linear layer. During the policy phase this head does nothing; it simply waits, a hook through which value information can later be poured into the policy's trunk. The auxiliary phase, run once every $N_\pi$ policy updates over all of $B$, then trains that head to predict the value targets — and because the head sits on the policy trunk, lowering its regression error forces the trunk to learn value-predictive features, which *is* the feature sharing I wanted. The auxiliary objective starts as $L^{\text{aux}} = \tfrac12\,\hat{\mathbb{E}}[(V_{\theta_\pi}(s) - \hat V^{\text{targ}})^2]$.

The immediate danger is that optimizing the policy network to predict value changes the shared trunk, and the policy head sitting on that trunk will drift — I would wreck the very policy I just carefully improved. I need to change the trunk's *features* without changing the policy's *outputs*, and that is exactly what a behavioral-cloning / distillation term does. I snapshot the policy right before the auxiliary phase as $\pi_{\text{old}}$ and pin the current policy to it, so the joint auxiliary objective on the policy net's parameters $\theta_\pi$ becomes
$$L^{\text{joint}} = L^{\text{aux}} + \beta_{\text{clone}}\cdot\hat{\mathbb{E}}\big[\,\mathrm{KL}\big(\pi_{\text{old}}(\cdot|s),\ \pi_\theta(\cdot|s)\big)\big].$$
The KL clone lets the trunk absorb value features (driving $L^{\text{aux}}$ down) while holding the action distribution fixed (the KL pulls it back to $\pi_{\text{old}}$), with $\beta_{\text{clone}} = 1$ trading the two off. This is what makes "share features without disturbing the policy" literally true: the features move, the policy does not. The targets $\hat V^{\text{targ}}$ regressed against here are the same ones computed during the policy phase and stay fixed throughout the auxiliary phase, so this phase is pure supervised regression onto stationary targets — stable, and precisely the kind of optimization that tolerates many passes. Since I am already replaying all of $B$, I also re-fit the *true* value net on the same buffer with $L^{\text{value}}$, which is the natural place to give the value function its high sample reuse — by cranking the number of auxiliary epochs $E_{\text{aux}}$ rather than $E_V$. Because $L^{\text{joint}}$ (on the policy net) and $L^{\text{value}}$ (on the true value net) share no parameters under the default dual architecture, they are optimized separately.

The sample-reuse story is now fully decoupled: $E_\pi$ controls policy reuse in the policy phase and $E_{\text{aux}}$ controls value-and-feature reuse in the auxiliary phase. The conservative choice $E_\pi = 1$ takes one trust-region-like pass near the data-generating policy; the supervised-regression side gets the extra passes, with $E_{\text{aux}} \approx 6$ — enough to train features and the true value function without overfitting a recent buffer. One more knob is how often the auxiliary phase runs, set by $N_\pi$: even with the KL clone each auxiliary phase mildly perturbs the policy trunk, so running it frequently compounds that perturbation, and I make it infrequent with $N_\pi = 32$. The dual design costs roughly twice the parameters of a shared baseline; for that there is a single-network "detach" variant that keeps one trunk but detaches the true-value path from the shared representation so the true-value loss cannot push the policy features during the policy update — while in the auxiliary phase the shared features are still trained through the auxiliary value head, the branch whose whole purpose is to make the policy trunk value-predictive. The policy objective itself is swappable: $L^{\text{clip}}$ can be replaced by a fixed-weight KL penalty $L^{\text{KL}} = \hat{\mathbb{E}}[-\hat A_t\, r_t(\theta) + \beta_\pi\,\mathrm{KL}(\pi_{\text{old}}, \pi_\theta)]$ with $\beta_\pi = 1$, which is reasonable because rewards are normalized so discounted returns have roughly unit variance and clipping matters most when rewards are poorly scaled. More broadly, the framework is a way to do *any* auxiliary optimization alongside RL stably; value-function error is just the auxiliary objective I happen to use.

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
