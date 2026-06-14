# On-Policy Distillation (OPD), distilled

On-policy distillation transfers a large teacher's behavior into a small student by training
the student on **its own** generated rollouts, with a **dense per-token** signal: the
**per-token reverse KL** between the student and teacher next-token distributions, summed over
the completion tokens. It combines the on-policy state distribution of RL (the student trains
on the states it actually visits at inference) with the dense, every-token supervision of
distillation — and, because the teacher's log-probabilities give a closed-form per-token
target, it is optimized as a stable supervised-style loss rather than a high-variance policy
gradient.

## Problem it solves

Compressing a frontier reasoning model into a small student. Two standard recipes each fail in
a specific way:

- **Off-policy / SFT distillation (forward KL on a fixed dataset).** Off-policy: the student
  trains on the teacher's state distribution, so an early mistake drops it into states it never
  trained on and the auto-regressive dependence makes the error compound (exposure bias /
  imitation-learning state mismatch). Forward KL `D_KL(p_T || p_S)` is **mass-covering**: under
  capacity mismatch the small student is forced to spread mass over the teacher's full support,
  including its low-probability tail, producing incoherent free-run generation.
- **RL (outcome reward).** On-policy (correct states) but **sparse**: ~`O(1)` bits/episode, one
  scalar reward, expensive credit assignment, and it needs a verifier or a learned (hackable)
  reward model.

The fix must be on-policy AND dense AND mode-seeking under capacity mismatch.

## Key idea

Minimize the **sequence-level reverse KL under student rollouts**,
`L(θ) = E_x D_KL(p_S(·|x) || p_T(·|x)) = E_x E_{y ~ p_S(·|x)} [log p_S(y|x) - log p_T(y|x)]`. Two design choices fall out:

- **On-policy** (`y ~ p_S`): trains exactly on the states the student visits at inference —
  removes the compounding-error cascade by construction.
- **Reverse KL** (weighted by `p_S`, not `p_T`): **mode-seeking / zero-forcing** — the student
  withdraws mass from tokens the teacher finds unlikely and concentrates on one coherent
  teacher mode, instead of smearing to cover everything. The right instinct under capacity
  mismatch. It is also "unhackable": the reward is the teacher's own log-prob of the student's
  token, so a low KL always corresponds to high teacher-probability behavior — there is no
  separate reward model to game. And it is dense — `O(N)` bits/episode for `N` tokens.

The sequence reverse KL is an RL objective; its policy gradient is
`∇L = -E_{y~p_S} Σ_t (R_t - 1) ∇ log p_S(y_t|y_<t,x)`, with per-step reward
`r_t = log(p_T(y_t|·)/p_S(y_t|·))` and reward-to-go `R_t = Σ_{t'≥t} r_{t'}`. The long-horizon
`R_t` is what causes high variance, reward hacking, and a short-response length bias, and it
needs a battery of stabilizers (baselines, teacher-mixed sampling, importance weights, length
normalization, PPO clipping).

**Discount factor 0.** Decompose the gradient into the immediate-step term and the
long-horizon term:

```
∇L = -E_{y~p_S} Σ_t ∇ E_{y_t~p_S(·|y_<t,x)}[ r_t ]   −   E_{y~p_S} Σ_t R_{t+1} ∇ log p_S(y_t|·)
   = (∇L)_Single                                      +   (∇L)_Long
```

The single-step expectation is a full vocabulary sum:
`E_{y_t~p_S}[r_t] = Σ_v p_S(v|y_<t,x) log(p_T(v|·)/p_S(v|·)) = -D_KL(p_S(·|y_<t,x) || p_T(·|y_<t,x))`.
So `(∇L)_Single` is exactly the gradient of the **per-token reverse KL computed analytically
over the vocabulary** — differentiable, no Monte-Carlo, zero sampling variance. All pathologies live in
`(∇L)_Long`. Because the teacher gives a **dense** per-token signal, long-horizon credit
assignment is unnecessary, so set the discount to 0 and drop `(∇L)_Long`. By the
auto-regressive factorization
`D_KL(p_S(·|x)||p_T(·|x)) = E_{y~p_S} Σ_t [log p_S(y_t|·) - log p_T(y_t|·)]`, this surrogate
keeps the direct gradient of the same nonnegative conditional-KL factors and drops only the
future-prefix score-function term. Dropping `(∇L)_Long` eliminates the need for every stabilizer,
leaving a supervised-style loss.

## Final objective

Per training step, over the student's on-policy completion tokens (prompt/padding masked):

```
L = (1 / N_tokens) Σ_{t : completion} D_KL( p_S(·|y_<t,x) || p_T(·|y_<t,x) )
  = (1 / N_tokens) Σ_{t : completion} Σ_v p_S(v|y_<t,x) [ log p_S(v|·) - log p_T(v|·) ]
```

with both logit tensors divided by a shared distillation temperature before the softmax.

## Where it sits (design space)

It is the corner `λ = 1` (fully on-policy), divergence = reverse KL (`β = 1` endpoint of the
generalized JSD family `D_JSD(β) = β·D_KL(P||M) + (1-β)·D_KL(Q||M)`, `M = βP + (1-β)Q`, where
`β→0` behaves like forward KL and the `β=1` implementation endpoint gives reverse KL) of the general distillation objective
`(1-λ) E_{data}[D(p_T||p_S)] + λ E_{x, y~p_S}[D(p_T||p_S)]` (no backprop through sampling).
On-policy because the state cascade demanded it; reverse KL because capacity mismatch demanded
mode-seeking; analytic + discount-0 because the teacher's dense log-probs made the closed form
available.

## Working code

Fills the one empty slot — the per-token transfer loss — in the on-policy distillation trainer.
The trainer samples the student's rollouts, runs both models over the same tokens, and supplies
`student_logits`, `teacher_logits` `[B, T, V]` and `labels` `[B, T]` (`-100` on prompt/padding).

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,    # [B, T, V] student logits over the rollout tokens
    teacher_logits: torch.Tensor,    # [B, T, V] frozen-teacher logits over the same tokens
    labels: torch.Tensor = None,     # [B, T]; -100 on prompt/padding, token id on completion
    beta: float = 1.0,               # reverse-KL endpoint; kept for trainer API
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,
    total_steps: int = 0,
    lmbda: float = 1.0,              # student-rollout corner; applied upstream by the trainer
) -> torch.Tensor:
    # On-Policy Distillation: per-token reverse KL on the student's own tokens, KL(p_S || p_T).
    # Mode-seeking under capacity mismatch; dense at every token; discount factor 0
    # (no long-horizon term) makes it a supervised-style differentiable loss.

    # Shared distillation temperature: soften BOTH distributions equally so the divergence
    # measures behavior, not a sharpness mismatch.
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_S || p_T) = Sum_v p_S(v) [log p_S(v) - log p_T(v)], summed over vocab -> [B, T].
    # F.kl_div(input=log_q, target=log_p, log_target=True) = Sum_v p_target (log p_target - input).
    # input = teacher_log_probs, target = student_log_probs  ==>  KL(p_S || p_T) (reverse).
    # Swapping the two args yields forward KL(p_T || p_S) -- the wrong direction.
    per_token = F.kl_div(
        teacher_log_probs, student_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)  # [B, T]

    # Only completion tokens count; mask out prompt/padding (labels == -100).
    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(
            per_token.numel(), device=per_token.device, dtype=torch.long
        ).clamp_min(1)

    if reduction == "batchmean":          # per-token mean -> length-scale-free
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```

Score-function form for sampled tokens (discount 0): keep the immediate token only, and include
the `-1` normalization term that comes from differentiating the student's own `log p_S` inside
the reverse-KL expectation.

```python
reverse_kl = sampled_logprobs - teacher_logprobs   # log p_S(y_t) - log p_T(y_t)
r_t = -reverse_kl                                  # log p_T(y_t) - log p_S(y_t)
pg_weight = r_t - 1.0                              # exact score-function weight, gamma = 0
# loss_pg = -(pg_weight.detach() * sampled_logprobs), masked to completion tokens.
# Prefer the analytic vocabulary-sum loss above when teacher logits are available.
```
