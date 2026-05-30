Let me start from a nagging implementation decision in on-policy actor-critic and try to actually resolve it instead of just picking a side. I have a policy π and a value function V, and I have to decide: one shared network with two heads, or two separate networks? Both choices irritate me, and I want to name exactly why.

Take the shared network first. The appeal is obvious: on a high-dimensional input like images, the features that let you predict the value are largely the same features that let you choose an action, so a shared trunk learns one good representation that serves both — and empirically, on visually rich benchmarks, sharing is not optional; separate networks just do worse there. But sharing forces two compromises. First, I now have one set of parameters being pulled by two different gradients, the policy gradient and the value-regression gradient, so I have to pick a relative weight — a vfcoef multiplying L^value against L^clip — and no matter how carefully I tune it, the two objectives can *interfere*: a step that's good for value prediction can be bad for the policy, and they fight over the shared trunk. Second, and this one is subtler, a shared trunk all but forces π and V to be trained on the *same data* with the *same number of epochs*. If I do three optimization passes over a batch, I do three for both. There's no reason that should be the right setting for both — it's an artificial coupling I never chose, it just falls out of having one network.

Now the separate networks. Two trunks, two optimizers, no shared parameters. The interference is gone — the value gradient never touches the policy's parameters — and I can train each with whatever sample reuse I want. But I've thrown away the feature sharing, and on the inputs I care about that sharing was doing real work. So separate-net actor-critic underperforms exactly where it matters.

So I'm stuck choosing between feature sharing (shared net) and non-interference-plus-independent-sample-reuse (separate net), and I want all three. Before I try to get them, let me pin down whether independent sample reuse even matters, because if value and policy want the same reuse anyway, the whole coupling complaint evaporates. So I run the thought experiment of sweeping sample reuse — epochs per batch — in a shared PPO. More epochs helps, up to about three, then tapers. But "epochs" in a shared net does two things at once: it does extra *policy* optimization AND extra *value* optimization, confounded. I can't tell which one wants the extra passes. To disentangle I'd need to vary them independently — which I can't do in a shared net. That itch — "I can't even measure the thing because the architecture confounds it" — is the tell that the coupling is the real problem.

Here's the hypothesis I'd bet on: the policy wants very *few* epochs and the value wants *many*. The policy update is a first-order trust-region-ish surrogate (PPO's clipped objective) that's only valid near the current policy, so beating on it for many epochs marches π into a region where the surrogate lies — a single policy epoch should be near-optimal. The value function, by contrast, is just regression onto returns; regression tolerates being fit hard, many passes, with no policy-collapse risk. If that's right, then PPO's apparent need for three epochs isn't the policy benefiting — it's the *value* being under-trained, and the extra epochs are sneaking in extra value fitting through the shared loss. To test that cleanly I'd need to crank value reuse while holding policy reuse at one. Which, again, I can't do while they share a network and a batch.

So the requirement crystallizes: I need to train the policy with low sample reuse and the value with high sample reuse, *and* keep them sharing features, *and* stop the value loss from interfering with the policy. Sharing-without-interference and same-network-but-different-sample-reuse sound contradictory — within a single joint optimization they are. The way out is to stop doing it as a single joint optimization. Separate the two trainings *in time*: alternate between a phase that trains the policy and a phase that handles the value's feature-sharing. If the value's gradient never touches the policy *during the policy phase*, there's no interference there; and if there's a separate phase whose whole job is to push value-derived features into the policy's representation, the sharing survives — just relocated to its own phase where it can't disturb the policy update.

Let me design the policy phase first, because it's almost just PPO. Use *separate* networks for policy and value during this phase — a policy net and a true value net — so that while I'm optimizing the policy there is zero value-gradient on the policy's parameters. Run N_π iterations of: collect rollouts under the current π, compute the value targets V̂^targ and advantages Â with GAE, then optimize the policy net for E_π epochs on the PPO clipped surrogate plus an entropy bonus, L^clip + β_S·S[π], and optimize the true value net for E_V epochs on L^value = Ê[½(V(s) − V̂^targ)²]. Because the two nets are disjoint here, E_π and E_V are finally independent — I can set E_π = 1 (the policy only wants one pass) and give value its own reuse. And along the way, stash every (s_t, V̂^targ_t) into a buffer B.

But wait — if the policy and value live in *separate* networks during the policy phase, where did the feature sharing go? I haven't shared anything yet; I've just built separate-net PPO with decoupled epochs, which has the non-interference and the independent reuse but *not* the sharing. The sharing is the job of the second phase, and I need a place for it to land. The policy network needs to *learn* value-predictive features in its own trunk. So give the policy network an extra head — an auxiliary value head V_θπ that shares all of the policy net's parameters except its own final linear layer. During the policy phase this head does nothing; it just exists, waiting. Its entire purpose is to be a hook through which value information can be poured into the policy's trunk later.

Now the auxiliary phase, run once every N_π policy updates over all the data accumulated in B. The goal: train the policy net's auxiliary value head to predict the value targets — which, because the head shares the policy trunk, forces the trunk to learn value-predictive features → the feature sharing I wanted. The loss for that is just the auxiliary head's value regression, L^aux = ½·Ê[(V_θπ(s) − V̂^targ)²]. But there's an immediate danger: if I optimize the policy *network* to predict value, I'm changing the shared trunk, and the policy head sitting on top of that trunk will drift — I'll wreck the policy I just spent the policy phase carefully improving. I need to change the trunk's *features* without changing the policy's *outputs*. That's exactly what a distillation / behavioral-cloning term does: snapshot the policy right before the auxiliary phase as π_old, and add a clone term that pins the current policy to it. So the auxiliary objective is

  L^joint = L^aux + β_clone · Ê[ KL( π_old(·|s), π_θ(·|s) ) ],

optimized with respect to the policy net's parameters θ_π. The KL term lets the trunk absorb value features (lowering L^aux) while holding the policy's action distribution fixed (the KL pulls it back to π_old); β_clone trades these off. This is the mechanism that makes "share features without disturbing the policy" literally true: features move, policy doesn't.

A few details fall out. The value targets V̂^targ I regress against in the auxiliary phase are the *same* ones computed during the policy phase, and they stay fixed throughout the auxiliary phase — so the auxiliary phase is pure supervised regression onto stationary targets, which is stable and exactly the kind of thing that tolerates many epochs. And since I'm already replaying all of B in the auxiliary phase, I'll take the opportunity to also further train the *true* value net here, optimizing L^value on B as well — this is now the natural place to give the value function its high sample reuse, by cranking the number of auxiliary epochs E_aux rather than E_V. Note L^joint (acting on the policy net) and L^value (acting on the true value net) share no parameters, so I optimize them separately within the auxiliary phase.

So the sample-reuse story is fully decoupled now: E_π controls policy reuse in the policy phase (set to 1), and E_aux controls value/feature reuse in the auxiliary phase. Sweeping E_π confirms the hypothesis — a single policy epoch is near-optimal once value training is isolated, which means PPO's old "3 epochs" really was under-trained value leaking through the shared loss. Sweeping E_aux shows extra auxiliary epochs help, tapering around six (too many starts overfitting the recent buffer). So E_aux ≈ 6 is the main knob for value sample reuse.

One more knob: how often to run the auxiliary phase, controlled by N_π. The auxiliary phase, even with the KL clone, still perturbs the policy network's trunk a little, so running it too frequently compounds that perturbation and interferes with policy optimization. Running it rarely keeps the disturbance small. Sweeping N_π confirms frequent auxiliary phases hurt; infrequent ones (N_π = 32) are critical to success.

Let me also sanity-check the memory cost and an alternative. The dual-network design uses roughly twice the parameters of a shared-net baseline (separate policy and value trunks). If that matters, there's a single-network variant that mimics the same dynamics: keep one shared trunk, but during the *policy phase* detach the value-head gradient at the last shared layer, so the value loss can't flow into the shared trunk and can't interfere with the policy; during the *auxiliary phase* take the value gradient with respect to all parameters, including the shared ones, so the trunk still learns value features. That recovers most of the benefit at one-times the parameters. (My initial worry was that the value function couldn't train its shared features at all during the policy phase under the detach — but it gets the full gradient back in the auxiliary phase, so in practice it's fine.)

And the policy objective itself is swappable: I've used the PPO clipped surrogate, but I could replace L^clip with a fixed-weight KL-penalty form, L^KL = Ê[ −Â_t·r_t(θ) + β_π·KL(π_old, π_θ) ] with β_π = 1, and it performs about the same in this framework — clipping matters most when rewards are poorly scaled, and I normalize rewards so discounted returns have roughly unit variance, which defuses that concern. The whole framework is, in fact, a way to do *any* auxiliary optimization alongside RL stably; value-function error is just the auxiliary objective I happen to use.

Let me write it as real code. The model is a dual encoder with a policy head, a true value head, and the auxiliary value head on the policy trunk:

```python
import torch as th
import torch.nn as nn
from torch import distributions as td


class PhasicValueModel(nn.Module):
    def __init__(self, ob_space, ac_space, enc_fn, arch="dual"):   # "dual" | "detach" | "shared"
        super().__init__()
        self.arch = arch
        self.detach_value_head = (arch == "detach")
        self.true_vf_key = "vf" if arch == "dual" else "pi"        # dual: separate value trunk
        self.pi_enc = enc_fn(ob_space)                             # policy encoder (trunk)
        if arch == "dual":
            self.vf_enc = enc_fn(ob_space)                         # separate true-value encoder
        feat = self.pi_enc.outsize
        self.pi_head = nn.Linear(feat, ac_space.size)             # policy head
        self.true_vf_head = nn.Linear(self._vf_feat(), 1)         # TRUE value head
        self.aux_vf_head = nn.Linear(feat, 1)                     # AUX value head, on the POLICY trunk
        self.make_distr = lambda logits: td.Categorical(logits=logits)

    def _vf_feat(self):
        return (self.vf_enc.outsize if self.arch == "dual" else self.pi_enc.outsize)

    def forward(self, ob):
        pi_x = self.pi_enc(ob)
        pd = self.make_distr(self.pi_head(pi_x))
        if self.arch == "dual":
            vf_x = self.vf_enc(ob)
        else:
            vf_x = pi_x.detach() if self.detach_value_head else pi_x   # detach variant
        vpred_true = self.true_vf_head(vf_x)[..., 0]
        vpred_aux = self.aux_vf_head(pi_x)[..., 0]                # shares policy trunk -> trains its features
        return pd, vpred_true, vpred_aux

    def compute_aux_loss(self, vpred_aux, vpred_true, vtarg):
        return {
            "vf_aux":  0.5 * ((vpred_aux  - vtarg) ** 2).mean(),  # distill value features into policy trunk
            "vf_true": 0.5 * ((vpred_true - vtarg) ** 2).mean(),  # extra training of the true value net
        }
```

GAE and the PPO policy-phase losses:

```python
def compute_gae(reward, vpred, first, gamma, lam):
    nenv, nstep = reward.shape
    adv = th.zeros(nenv, nstep)
    lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        delta = reward[:, t] + notlast * gamma * vpred[:, t + 1] - vpred[:, t]   # TD residual
        adv[:, t] = lastgaelam = delta + notlast * gamma * lam * lastgaelam      # Σ(γλ)^l δ
    vtarg = vpred[:, :-1] + adv                                                  # V̂^targ = V + Â
    return adv, vtarg


def ppo_losses(model, mb, clip_param, ent_coef):
    pd, vpred_true, _aux = model(mb["ob"])
    newlogp = pd.log_prob(mb["ac"])
    ratio = th.exp(newlogp - mb["logp"])                                # π_θ / π_old
    pg = th.max(-mb["adv"] * ratio,
                -mb["adv"] * th.clamp(ratio, 1 - clip_param, 1 + clip_param)).mean()   # clipped surrogate
    entropy = pd.entropy().mean()
    pi_loss = pg - ent_coef * entropy                                   # L^clip + β_S S[π]
    vf_loss = ((vpred_true - mb["vtarg"]) ** 2).mean()                  # L^value (true value net)
    return pi_loss, vf_loss
```

The auxiliary-phase step — value-feature distillation plus the policy clone term, plus the true-value refit:

```python
def aux_train(model, buffer, opt, beta_clone, vf_true_weight):
    for mb in buffer.minibatches():
        pd, vpred_true, vpred_aux = model(mb["ob"])
        losses = model.compute_aux_loss(vpred_aux, vpred_true, mb["vtarg"])  # fixed targets from policy phase
        pol_distance = td.kl_divergence(mb["oldpd"], pd).mean()              # KL(π_old, π_θ): preserve the policy
        loss = (losses["vf_aux"]                                             # L^aux
                + beta_clone * pol_distance                                  # β_clone · KL clone
                + vf_true_weight * losses["vf_true"])                        # extra true-value training
        opt.zero_grad(); loss.backward(); opt.step()
```

The outer alternation: N_π policy-phase iterations, then one auxiliary phase of E_aux epochs over the buffer:

```python
def learn(venv, model, hp):
    pi_opt  = th.optim.Adam(model.parameters(), lr=hp["lr"])
    aux_opt = th.optim.Adam(model.parameters(), lr=hp["aux_lr"])
    while True:
        buffer = []
        for _ in range(hp["n_pi"]):                                     # POLICY PHASE (N_π PPO iterations)
            rollouts = collect_rollouts(venv, model, hp["nstep"])
            adv, vtarg = compute_gae(rollouts["reward"], rollouts["vpred"],
                                     rollouts["first"], hp["gamma"], hp["lam"])
            rollouts["adv"], rollouts["vtarg"] = adv, vtarg
            for _ in range(hp["E_pi"]):                                  # E_π policy epochs (=1)
                for mb in minibatches(rollouts, hp["nminibatch"]):
                    pi_loss, _ = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"])
                    pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()
            for _ in range(hp["E_v"]):                                  # E_V value epochs (=1)
                for mb in minibatches(rollouts, hp["nminibatch"]):
                    _, vf_loss = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"])
                    pi_opt.zero_grad(); vf_loss.backward(); pi_opt.step()
            buffer.append({k: rollouts[k] for k in ("ob", "vtarg")})    # stash (s, V̂^targ)

        for seg in buffer:                                              # snapshot π_old before the aux phase
            with th.no_grad():
                seg["oldpd"], _, _ = model(seg["ob"])
        for _ in range(hp["n_aux_epochs"]):                             # AUXILIARY PHASE (E_aux epochs over B)
            aux_train(model, BufferView(buffer), aux_opt,
                      hp["beta_clone"], hp["vf_true_weight"])

# hp: n_pi=32, E_pi=1, E_v=1, n_aux_epochs=6, beta_clone=1, vf_true_weight=1,
#     gamma=.999, lam=.95, nstep=256, nminibatch=8, ent_coef=.01, clip_param=.2,
#     lr=aux_lr=5e-4, reward normalization on.
```

The chain, end to end: a shared actor-critic net gets feature sharing but suffers policy↔value interference and forces a single sample-reuse on both objectives; a separate-net version kills the interference and decouples reuse but loses the sharing — and you can't even measure whether policy and value want different reuse because the shared loss confounds them. Separating the two trainings *in time* breaks the deadlock: a policy phase runs PPO on disjoint policy and value nets (so no value gradient touches the policy, and E_π and E_V are independent — E_π = 1 turns out near-optimal), accumulating (s, V̂^targ) into a buffer; an auxiliary phase then distills value-predictive features into the policy net's trunk through an auxiliary value head (L^aux = ½(V_θπ − V̂^targ)²) while a behavioral-cloning term β_clone·KL(π_old, π_θ) holds the policy's outputs fixed — so the representation is shared but the policy is undisturbed — and re-fits the true value net on the same buffer, giving value its high sample reuse via E_aux ≈ 6 rather than E_V. The auxiliary phase is run only every N_π = 32 policy updates because more frequent phases re-perturb the policy. Targets stay fixed across the auxiliary phase (stable regression), the policy objective is swappable (PPO clip or a fixed KL penalty), and a single-net detach variant recovers most of the benefit at half the parameters.
