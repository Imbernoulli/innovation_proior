Let me start from a nagging implementation decision in on-policy actor-critic and try to actually resolve it instead of just picking a side. I have a policy π and a value function V, and I have to decide: one shared network with two heads, or two separate networks? Both choices irritate me, and I want to name exactly why.

Take the shared network first. The appeal is obvious: on a high-dimensional input like images, the features that let you predict the value are largely the same features that let you choose an action, so a shared trunk learns one good representation that serves both — and empirically, on visually rich benchmarks, sharing is not optional; separate networks just do worse there. But sharing forces two compromises. First, I now have one set of parameters being pulled by two different gradients, the policy gradient and the value-regression gradient, so I have to pick a relative weight — a vfcoef multiplying L^value against L^clip — and no matter how carefully I tune it, the two objectives can *interfere*: a step that's good for value prediction can be bad for the policy, and they fight over the shared trunk. Second, and this one is subtler, a shared trunk all but forces π and V to be trained on the *same data* with the *same number of epochs*. If I do three optimization passes over a batch, I do three for both. There's no reason that should be the right setting for both — it's an artificial coupling I never chose, it just falls out of having one network.

Now the separate networks. Two trunks, two optimizers, no shared parameters. The interference is gone — the value gradient never touches the policy's parameters — and I can train each with whatever sample reuse I want. But I've thrown away the feature sharing, and on the inputs I care about that sharing was doing real work. So separate-net actor-critic underperforms exactly where it matters.

So I'm stuck choosing between feature sharing (shared net) and non-interference-plus-independent-sample-reuse (separate net), and I want all three. Before I try to get them, let me pin down whether independent sample reuse even matters, because if value and policy want the same reuse anyway, the whole coupling complaint evaporates and I should just pick a side and move on. The PPO tuning fact that keeps bothering me is that extra epochs over a rollout can help, but in a shared net "epochs" does two things at once: it does extra *policy* optimization AND extra *value* optimization, confounded. I can't tell which one wants the extra passes. To disentangle I'd need to vary them independently — which I can't do in a shared net, where one epoch counter drives both. The discomfort I keep returning to is less "sharing is bad" than "I can't even measure whether the two objectives want the same reuse, because the architecture fuses the measurement." Let me at least argue, before committing to anything, that they plausibly want *different* reuse — because if I can't make that case the whole exercise is unmotivated.

My working guess is that the policy wants very *few* epochs and the value wants *many*, and I can give a reason rather than just a feeling. The policy update is a first-order trust-region-ish surrogate (PPO's clipped objective) that's only valid near the current policy, and the clip mechanism is explicitly built to stop contributing once the ratio leaves the band [1−ε, 1+ε]. Let me check that the surrogate really does go flat there, because that is the whole argument for "one epoch is almost enough." Take a single sample with advantage Â > 0. The per-sample objective is min(r·Â, clip(r, 1−ε, 1+ε)·Â). For r below 1+ε the first term governs and ∂/∂r = Â > 0 — there's gradient, the policy still moves. Once r climbs past 1+ε the clipped term governs, it equals (1+ε)·Â which has no r-dependence, so ∂/∂r = 0: the sample stops pushing. With ε = .2 and Â > 0 the contribution is pinned the instant the action's probability has risen ~20% over π_old. So within a single epoch many samples are already at their clip edge and contribute nothing; a second and third epoch keep optimizing only the not-yet-clipped tail while dragging the clipped ones further from π_old, where the surrogate is no longer a faithful local model of the true objective. That's a concrete reason one policy pass should be close to optimal and more passes are at best wasted and at worst harmful. The value function, by contrast, is just regression onto returns; regression tolerates being fit hard, many passes, with no policy-collapse risk. If that's right, then PPO's apparent need for three epochs isn't the policy benefiting — it's the *value* being under-trained, and the extra epochs are sneaking in extra value fitting through the shared loss. To test that cleanly I'd need to crank value reuse while holding policy reuse at one. Which, again, I can't do while they share a network and a batch.

So the requirement crystallizes: I need to train the policy with low sample reuse and the value with high sample reuse, *and* keep them sharing features, *and* stop the value loss from interfering with the policy. Sharing-without-interference and same-network-but-different-sample-reuse sound contradictory — within a single joint optimization they are. The way out is to stop doing it as a single joint optimization. Separate the two trainings *in time*: alternate between a phase that trains the policy and a phase that handles the value's feature-sharing. If the value's gradient never touches the policy *during the policy phase*, there's no interference there; and if there's a separate phase whose whole job is to push value-derived features into the policy's representation, the sharing survives — just relocated to its own phase where it can't disturb the policy update.

Let me design the policy phase first, because it's almost just PPO. Use *separate* networks for policy and value during this phase — a policy net and a true value net — so that while I'm optimizing the policy there is zero value-gradient on the policy's parameters. Run N_π iterations of: collect rollouts under the current π, compute the value targets V̂^targ and advantages Â with GAE, then optimize the policy net for E_π epochs on the PPO clipped surrogate plus an entropy bonus, L^clip + β_S·S[π], and optimize the true value net for E_V epochs on L^value = Ê[½(V(s) − V̂^targ)²]. Because the two nets are disjoint here, E_π and E_V are finally independent — I can set E_π = 1 (the policy only wants one pass) and give value its own reuse. And along the way, stash every (s_t, V̂^targ_t) into a buffer B.

But wait — if the policy and value live in *separate* networks during the policy phase, where did the feature sharing go? I haven't shared anything yet; I've just built separate-net PPO with decoupled epochs, which has the non-interference and the independent reuse but *not* the sharing. The sharing is the job of the second phase, and I need a place for it to land. The policy network needs to *learn* value-predictive features in its own trunk. So give the policy network an extra head — an auxiliary value head V_θπ that shares all of the policy net's parameters except its own final linear layer. During the policy phase this head does nothing; it just exists, waiting. Its entire purpose is to be a hook through which value information can be poured into the policy's trunk later.

Now the auxiliary phase, run once every N_π policy updates over all the data accumulated in B. The goal: train the policy net's auxiliary value head to predict the value targets — which, because the head shares the policy trunk, forces the trunk to learn value-predictive features → the feature sharing I wanted. The loss for that is just the auxiliary head's value regression, L^aux = ½·Ê[(V_θπ(s) − V̂^targ)²]. But there's an immediate danger: if I optimize the policy *network* to predict value, I'm changing the shared trunk, and the policy head sitting on top of that trunk will drift — I'll wreck the policy I just spent the policy phase carefully improving. I need to change the trunk's *features* without changing the policy's *outputs*. That's exactly what a distillation / behavioral-cloning term does: snapshot the policy right before the auxiliary phase as π_old, and add a clone term that pins the current policy to it. So the auxiliary objective is

  L^joint = L^aux + β_clone · Ê[ KL( π_old(·|s), π_θ(·|s) ) ],

optimized with respect to the policy net's parameters θ_π. I want to be careful about what this term actually buys, because "holds the policy fixed" is the kind of claim I've talked myself into before and been wrong. Let me look at the gradient of the clone term at the moment the auxiliary phase starts, when π_θ is still exactly π_old. There KL(π_old, π_θ) = 0, and since KL is non-negative and minimized at π_θ = π_old, that point is a minimum, so its gradient w.r.t. the policy logits is zero. I'll check on a 4-action categorical: with π_θ snapshotted as π_old, KL evaluates to 0.0 and ∇_logits KL = [0, 0, 0, 0] — exactly inert. So on the very first aux step the *only* gradient on the trunk through the policy path is from L^aux; the clone term does nothing yet. Good — that means the trunk genuinely starts absorbing value features rather than being frozen.

But that also tells me the clone term cannot literally pin the policy: it has zero force precisely where I'd want it strongest. What it actually provides is a *restoring* force that grows as the policy drifts. I perturb the same logits by a step (standing in for what the L^aux gradient does to the shared trunk) and recompute: KL rises to ≈ 0.034 and ∇_logits KL becomes ≈ [+.097, −.019, +.002, −.080] against a drift of [+.4, −.2, +.1, −.3] — component by component the gradient opposes the drift, i.e. it pulls π_θ back toward π_old. So the honest statement is not "the policy doesn't move," it's "the trunk is free to change features, and any resulting drift in the action distribution is met by a spring of stiffness β_clone pulling it back." With β_clone large enough the policy is held *close* to π_old while the features move; it is a soft constraint, not an identity. That's the property I actually need, and it's the one I can defend.

A few details fall out. The value targets V̂^targ I regress against in the auxiliary phase are the *same* ones computed during the policy phase, and they stay fixed throughout the auxiliary phase — so the auxiliary phase is pure supervised regression onto stationary targets, which is stable and exactly the kind of thing that tolerates many epochs. And since I'm already replaying all of B in the auxiliary phase, I'll take the opportunity to also further train the *true* value net here, optimizing L^value on B as well — this is now the natural place to give the value function its high sample reuse, by cranking the number of auxiliary epochs E_aux rather than E_V. Note L^joint (acting on the policy net) and L^value (acting on the true value net) share no parameters, so I optimize them separately within the auxiliary phase.

So the sample-reuse story is fully decoupled now: E_π controls policy reuse in the policy phase, and E_aux controls value/feature reuse in the auxiliary phase. The conservative policy choice is E_π = 1: take one trust-region-like pass near the data-generating policy, then stop before the surrogate is overused. The value side is supervised regression onto fixed targets, so the extra passes belong there. E_aux is therefore the main value-reuse knob; I want enough passes to train features and the true value function, but not so many that I overfit a recent buffer, so I use a small finite number, about six.

One more knob: how often to run the auxiliary phase, controlled by N_π. The auxiliary phase, even with the KL clone, still perturbs the policy network's trunk a little, so running it too frequently compounds that perturbation and interferes with policy optimization. Running it rarely keeps the disturbance small. I therefore make the auxiliary phase infrequent, with N_π = 32 as the default scale.

Let me also sanity-check the memory cost and an alternative. The dual-network design uses roughly twice the parameters of a shared-net baseline (separate policy and value trunks). If that matters, there's a single-network variant that keeps one shared trunk but detaches the true-value path from the shared representation, so the true value loss cannot push the policy features during the policy update. In the implementation I will write below, that detach is a static guard on the true-value head's input; the shared features in the auxiliary phase are trained by the auxiliary value head, which is exactly the branch whose purpose is to make the policy trunk value-predictive.

And the policy objective itself is swappable: I've used the PPO clipped surrogate, but I could replace L^clip with a fixed-weight KL-penalty form, L^KL = Ê[ −Â_t·r_t(θ) + β_π·KL(π_old, π_θ) ] with β_π = 1. Clipping matters most when rewards are poorly scaled, and I normalize rewards so discounted returns have roughly unit variance, which makes a fixed policy-distance penalty a reasonable alternative. Stepping back, nothing in the auxiliary-phase mechanism actually referred to the value function: it distilled *some* target into the trunk while a clone term protected the policy. So I'd expect the same scaffold to carry any auxiliary objective alongside RL, with value-function error just being the one I happen to use here — though that's a conjecture I haven't tested with any other auxiliary task, so I'd want to try one before claiming it.

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
```

This backward recursion and the `V̂^targ = V + Â` line are exactly what the auxiliary phase will regress against, so I want to be sure they're right before building everything on top of them. Let me trace one rollout by hand. Take one env, three steps, rewards (1, 0, 2), and a flat value estimate V = .5 at every state including the bootstrap V₃, with no episode boundary so `notlast = 1`. Set γ = λ = 1 first, which collapses GAE to a plain return-minus-baseline and is easy to check. The TD residuals are δ_t = r_t + V_{t+1} − V_t = r_t (the .5's cancel), so δ = (1, 0, 2). The recursion accumulates from the back: Â₂ = 2, Â₁ = 0 + Â₂ = 2, Â₀ = 1 + Â₁ = 3, giving Â = (3, 2, 2); then V̂^targ = .5 + Â = (3.5, 2.5, 2.5). Running the code on this input returns `adv = [3.0, 2.0, 2.0]` and `vtarg = [3.5, 2.5, 2.5]` — matches. As a second check with the real discounting, γ = .99, λ = .95: now δ_t = r_t + .99·.5 − .5 = r_t − .005, and Â₀ should be δ₀ + (γλ)δ₁ + (γλ)²δ₂ = .995 + .9405·(−.005) + .8845·1.995 = 2.755; the code returns 2.755. So the geometric (γλ)^l weighting is being applied correctly and the bootstrap term enters where it should. The advantage/target machinery is trustworthy; the rest of the design can lean on it.

```python
def normalize_adv(adv, eps=1e-8):
    return (adv - adv.mean()) / (adv.var(unbiased=False).sqrt() + eps)


def ppo_losses(model, mb, clip_param, ent_coef, vf_coef=0.5, kl_penalty=0.0):
    pd, vpred_true, _aux = model(mb["ob"])
    newlogp = pd.log_prob(mb["ac"])
    logratio = newlogp - mb["logp"]
    ratio = th.exp(logratio)                                             # π_θ / π_old
    pg = th.max(-mb["adv"] * ratio,
                -mb["adv"] * th.clamp(ratio, 1 - clip_param, 1 + clip_param)).mean()   # clipped surrogate
    entropy = pd.entropy().mean()
    approx_kl_penalty = kl_penalty * 0.5 * (logratio ** 2).mean()
    pi_loss = pg - ent_coef * entropy + approx_kl_penalty               # minimize negative policy objective
    vf_loss = vf_coef * ((vpred_true - mb["vtarg"]) ** 2).mean()        # vf_coef=.5 gives L^value
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
    ppo_opt = th.optim.Adam(model.parameters(), lr=hp["lr"])
    vf_opt = ppo_opt if hp["E_pi"] == hp["E_v"] else th.optim.Adam(model.parameters(), lr=hp["lr"])
    aux_opt = th.optim.Adam(model.parameters(), lr=hp["aux_lr"])
    while True:
        buffer = []
        for _ in range(hp["n_pi"]):                                     # POLICY PHASE (N_π PPO iterations)
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
                for _ in range(hp["E_v"]):                              # value epochs first, as in the implementation
                    for mb in minibatches(rollouts, hp["nminibatch"]):
                        _, vf_loss = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"],
                                                hp["vf_coef"], hp.get("kl_penalty", 0.0))
                        vf_opt.zero_grad(); vf_loss.backward(); vf_opt.step()
                for _ in range(hp["E_pi"]):
                    for mb in minibatches(rollouts, hp["nminibatch"]):
                        pi_loss, _ = ppo_losses(model, mb, hp["clip_param"], hp["ent_coef"],
                                                hp["vf_coef"], hp.get("kl_penalty", 0.0))
                        ppo_opt.zero_grad(); pi_loss.backward(); ppo_opt.step()
            buffer.append({k: rollouts[k] for k in ("ob", "vtarg")})    # stash (s, V̂^targ)

        for seg in buffer:                                              # snapshot π_old before the aux phase
            with th.no_grad():
                seg["oldpd"], _, _ = model(seg["ob"])
        for _ in range(hp["n_aux_epochs"]):                             # AUXILIARY PHASE (E_aux epochs over B)
            aux_train(model, BufferView(buffer), aux_opt,
                      hp["beta_clone"], hp["vf_true_weight"])

# hp: n_pi=32, E_pi=1, E_v=1, n_aux_epochs=6, beta_clone=1, vf_true_weight=1,
#     gamma=.999, lam=.95, nstep=256, nminibatch=8, ent_coef=.01, clip_param=.2,
#     vf_coef=.5, kl_penalty=0.0, lr=aux_lr=5e-4, reward normalization on.
```

The chain, end to end: a shared actor-critic net gets feature sharing but suffers policy↔value interference and forces a single sample-reuse on both objectives; a separate-net version kills the interference and decouples reuse but loses the sharing — and you can't even measure whether policy and value want different reuse because the shared loss confounds them. Separating the two trainings *in time* breaks the deadlock: a policy phase runs PPO on disjoint policy and value nets (so no value gradient touches the policy, and E_π and E_V are independent, with E_π set to one conservative policy pass), accumulating (s, V̂^targ) into a buffer; an auxiliary phase then distills value-predictive features into the policy net's trunk through an auxiliary value head (L^aux = ½(V_θπ − V̂^targ)²) while a behavioral-cloning term β_clone·KL(π_old, π_θ) acts as a restoring spring on the policy's outputs (inert at the snapshot, growing as the policy drifts) — so the representation is free to change while the action distribution is held close to π_old — and re-fits the true value net on the same buffer, giving value its high sample reuse via E_aux ≈ 6 rather than E_V. The auxiliary phase is run only every N_π = 32 policy updates because more frequent phases re-perturb the policy. Targets stay fixed across the auxiliary phase, and the policy objective is swappable between PPO clipping and a fixed KL penalty.
