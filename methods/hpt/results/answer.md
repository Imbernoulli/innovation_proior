# Hybrid Post-Training (HPT), distilled

HPT is a single-stage post-training algorithm for LLM reasoning that routes *each question*
between on-policy RL (GRPO) and off-policy SFT based on the model's real-time rollout accuracy
on that question. It rests on a unifying observation — the **Unified Policy Gradient Estimator
(UPGE)** — that SFT and RL are not competing objectives but two halves of the gradient of one
common objective, so combining them is a choice of estimator, not a hack. HPT exploits the
demonstration (SFT) exactly on the questions the model cannot yet solve, and explores with RL
on the rest, with the mixture self-adjusting as the model improves.

## Problem it solves

Post-train `π_θ` on verifiable reasoning tasks where each prompt has both a verifier (binary
correctness) and a teacher demonstration `τ★`. On-policy RL cannot bootstrap questions the
model fails completely; SFT alone memorizes and kills exploration; hand-built SFT+RL blends and
SFT→RL pipelines commit to a fixed combination that ignores how the model's competence varies
across questions and over training. HPT makes the SFT/RL balance adaptive and per-question, in
one pass, with one interpretable knob.

## The unification (UPGE)

Start from one common objective — maximize expected verifier reward while staying close to the
demonstration policy `π_β`:

```
J_μ(θ) = E_{τ~π_θ}[ r(τ|q) ] − μ · KL( π_β(·|q) ‖ π_θ(·|q) ),   μ ≥ 0.
```

Its gradient (score-function identity for the reward term; `π_β` fixed in θ for the KL term)
splits cleanly into an on-policy RL term and an SFT term:

```
∇J_μ = E_{τ~π_θ}[ r ∇log π_θ ]  +  μ E_{τ~π_β}[ ∇log π_θ ].
       \________ RL ________/      \_______ SFT _______/
```

A change of measure to a common reference `π_ref` (using `∇log π_θ = (1/π_θ)∇π_θ`) collapses
both into one estimator:

```
grad_uni = 1_stable · (1/π_ref) · Â · ∇π_θ,     Â_uni = r  +  μ · π_β/π_θ.
```

Four interchangeable components: **stabilization mask** `1_stable` (PPO clipping = a
stop-gradient on unsafe samples; DAPO/CISPO/GSPO variants), **reference-policy denominator**
`1/π_ref` (SFT/REINFORCE: `π_θ`; PPO/GRPO: `π_{θ_old}`; offline: `π_ref ≡ 1`), **advantage** `Â`
(SFT: `≡1`; REINFORCE: `±1`; GRPO: group-normalized), and the shared **likelihood gradient**
`∇π_θ`. SFT, REINFORCE, PPO, GRPO, SRFT, LUFFY are all points in this space. A trust-region
penalty `λ KL(π_θ‖π_ref)` simply shifts the advantage: `Â^(λ) = r − λ log(π_θ/π_ref) + μ π_β/π_θ`.

**Consequence:** the major post-training losses can be read as estimators, sometimes biased
approximations, of the same gradient under different data assumptions. The right weighting is not
fixed — it depends on the model's current competence on each question — which motivates an
adaptive, per-question router rather than a global blend.

## Key idea (HPT)

Use a mixed loss `L = α L_RL + β L_SFT` with coefficients driven by per-question rollout
accuracy. For prompt `q`, draw `n` on-policy rollouts, verify them (`v(τ_i) = R(τ_i) ∈ {0,1}`),
and form the pass-rate `P = (1/n) Σ_i v(τ_i)` — free, since it is the same verifier scores GRPO
already consumes. Then `α = f(P)` (RL weight, increasing in `P`), `β = g(P)` (SFT weight,
decreasing in `P`). The simple, effective instance is a hard switch at gate `γ`:

```
(α, β) = (1, 0)  if P > γ   →  pure on-policy RL (explore; the model is competent here)
(α, β) = (0, 1)  if P ≤ γ   →  pure SFT on τ★ (exploit the demo; the model is stuck here)
```

- **L_RL** = GRPO clipped surrogate with group-normalized advantage
  `Â_i = (R(τ_i) − mean) / std` computed over the **on-policy group only**.
- **L_SFT** = token NLL on the demonstration: `−(1/|τ★|) Σ_t log π_θ(τ★_t | q, τ★_{<t})`.

## Why these choices

- **Why route on P:** it is a direct per-question competence measure and is already computed by
  the RL loop. The bias-variance of the RL vs SFT estimator depends on this competence, so the
  weighting must track it per question and over time (a fixed coefficient — LUFFY, SRFT — cannot).
- **Why `γ = 0` (Qwen):** switch to SFT only when **all** rollouts fail (`P = 0`). GRPO's
  group advantage degenerates whenever all rewards are equal (all-wrong or all-correct), but the
  all-wrong case is the one where the model still needs a bootstrap signal and on-policy RL
  contributes **no gradient**. SFT supplies the useful signal there; RL is kept when there is
  reward contrast, and an all-correct group is treated as saturated rather than copied again. It is
  the *minimal* SFT intervention, which preserves as much exploration as possible. Weaker models
  (Llama) use a higher bar, `γ = 2/8`.
- **Why SFT, not off-policy RL, for stuck prompts:** off-policy RL needs a reference policy for
  the teacher trace, forcing `π_ref ≡ 1` (rejection-sampling bias under an unmet uniform-coverage
  assumption). Plain SFT (`Â ≡ 1`, `π_ref = π_θ`) has no ill-posed ratio and is the clean way to
  consume an offline trajectory.
- **Why the advantage is normalized over on-policy samples only:** keep the RL measurement clean
  rather than contaminating the group statistics with injected demonstration samples.
- **SFT coefficient:** the actor multiplies the supervised term by `sft_loss_coef`; in the hard
  switch setting, that coefficient only scales the SFT rows that were inserted into the mixed
  actor batch.
- **Self-adjusting mixture:** early on a weak model fails many prompts ⇒ more demonstration
  updates; as competence rises, prompts cross the gate ⇒ more of the batch stays on on-policy RL.

## Final algorithm

```
Input: policy π_θ; demos {(q, τ★)}; verifier v; rollouts/question n; gate γ; steps T; lr η
for t = 1..T:
    for each prompt q in batch:
        sample {τ_i}_{i=1..n} ~ π_θ(·|q)
        R(τ_i) = v(τ_i) ∈ {0,1};   P = (1/n) Σ_i R(τ_i)
        (α, β) = (1, 0) if P > γ else (0, 1)
        L_RL  = GRPO_clip_surrogate({τ_i}, group_norm_adv({R(τ_i)}))   # used iff α
        L_SFT = −(1/|τ★|) Σ_t log π_θ(τ★_t | q, τ★_{<t})               # used iff β
        L_q   = α L_RL + β L_SFT
    θ ← θ − η ∇_θ Σ_q L_q     # AdamW
return π_θ
```

## Working code

The verl-style implementation expresses the switch as a batch edit. The trainer-side controller
returns `(on_remove_num, on_add_num, off_add_num)` per prompt-id; the SFT branch drops the
eight-response on-policy rollout group and adds a prefix-masked demonstration sample; `grpo_split`
normalizes advantages over on-policy samples only; the actor computes GRPO on non-prefix samples,
SFT on prefix samples, and uses `pg_loss = sft_loss * sft_loss_coef + pg_loss`.

```python
from collections import defaultdict
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos


def select_on_off_ada_balance(config, on_solve_num):
    """Return (on_remove_num, on_add_num, off_add_num), as in the mix_src trainer."""
    if config.trainer.unify_strategy == "switch":
        on_add_num = 0
        if on_solve_num <= config.trainer.switch_gate:
            return 8, on_add_num, 1          # remove on-policy group, add SFT target sample
        if on_solve_num <= config.trainer.switch_gate_off:
            return 8, on_add_num, -1         # optional off-policy-RL arm in the shared path
        return 0, on_add_num, 0              # keep on-policy GRPO samples

    if config.trainer.unify_strategy == "soft":
        return 0, 0, 1

    raise NotImplementedError


def compute_grpo_outcome_advantage_split(token_level_rewards, eos_mask, index,
                                         on_policy_mask, epsilon=1e-6, use_std=True):
    """Compute group-normalized advantages using only non-prefix (on-policy) samples."""
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    id2score, id2mean, id2std = defaultdict(list), {}, {}

    with torch.no_grad():
        for i in range(scores.shape[0]):
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for uid, values in id2score.items():
            if len(values) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = torch.mean(torch.tensor(values))
                id2std[uid] = torch.std(torch.tensor([values]))
                if id2std[uid].item() == 0:
                    id2std[uid] = torch.tensor(1.0)
        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            scores[i] = centered / (id2std[index[i]] + epsilon) if use_std else centered

    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return advantages, advantages


def compute_sft_pure_loss(log_prob, eos_mask):
    return verl_F.masked_mean(-log_prob, eos_mask)


def actor_mixed_loss(log_prob, old_log_prob, advantages, response_mask, prefix_mask, config):
    off_policy_mask = prefix_mask.any(-1)
    sft_loss = compute_sft_pure_loss(
        log_prob=log_prob[off_policy_mask],
        eos_mask=response_mask[off_policy_mask],
    )

    on_policy_mask = ~off_policy_mask
    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob[on_policy_mask],
        log_prob=log_prob[on_policy_mask],
        advantages=advantages[on_policy_mask],
        eos_mask=response_mask[on_policy_mask],
        cliprange=config.clip_ratio,
        loss_remove_token_mean=config.loss_remove_token_mean,
        loss_remove_clip=config.loss_remove_clip,
    )

    if not torch.isnan(sft_loss):
        pg_loss = sft_loss * config.sft_loss_coef + pg_loss

    return pg_loss, pg_clipfrac, ppo_kl
```
