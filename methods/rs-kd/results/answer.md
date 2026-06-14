# Random Sampling KD (RS-KD), distilled

Random Sampling Knowledge Distillation is a sparse, cache-able way to do distribution-matching
distillation for LLM pre-training. Instead of caching the teacher's full per-token distribution
(infeasible — petabytes) or its deterministic top-`K` entries (biased, miscalibrated), RS-KD
caches an **importance-sampled** summary: draw `N` token ids from the teacher distribution and
store the empirical counts. The summary is an *unbiased* estimate of the full teacher
distribution, so the student's logit gradient equals the full-distillation gradient in
expectation — at roughly a dozen stored entries per token.

## Problem it solves

Offline cached distillation: run the teacher once over the corpus, cache its output, reuse to
train students. The blocker is storage — the full `|V|`-dim teacher distribution per token is
unstorable at LLM scale (~`10` PB for `100`B tokens). RS-KD produces a per-token target that is
sparse (≈`12` unique tokens), cheap to store (≈`3` bytes per stored entry, or ≈`36` bytes per
training token at that sparsity), and unbiased in the sense needed to preserve the full-KD
gradient in expectation.

## Why top-K fails (the thing RS-KD fixes)

Keeping the top-`K` teacher probabilities is `L1`-optimal per token (error `2(1 - a)`,
`a = sum_{i in K} t_i`, minimized by the largest `a`), but it is a **biased** summary
(`E[t^s_i] = 0 != t_i` on the tail) and it deletes the tail. With the softmax-KL logit gradient
`dL/dx_j = (sum_i t_i) p_j - t_j`, a truncated (sub-stochastic) target gives
`a * p_j - t_j`: the optimum has the kept probabilities **scaled up** to `p_i = t_i / a`
(over-confidence, calibration error growing as `K` shrinks) and the tail driven to `0` (no
rare-token supervision). Reframed: top-`K` is an importance-sampling proposal that is zero on the
tail, violating the support condition for an unbiased estimate.

## Key idea

For a forward-KL distillation loss the logit gradient is `g_j = p_j - t_j` (full distribution,
`sum t = 1`). Replace the teacher target `t` with an **unbiased sparse estimate** `t^s` and the
gradient becomes `g^s_j = p_j - t^s_j` with `E[g^s_j] = p_j - t_j = g_j`. Importance sampling
provides such an estimate: draw from a proposal `q > 0` (everywhere `t > 0`) and reweight by
`t/q`; the estimator is unbiased for any supported `q`. Choosing the proposal as a tempered
teacher `q ∝ t^tau` gives a bias-free variance knob; at `tau = 1` the proposal is the teacher
itself, the importance ratio `t_i/q_i ≡ 1`, and the procedure reduces to:

1. Sample `N` token ids with replacement from the teacher distribution `t`.
2. Let `c_i` = count of token `i`; set the sparse target `t^s_i = c_i / N` (sums to `1`).
3. Train the student with forward KL `sum_i t^s_i log(t^s_i / p_i)` — equivalently, the average
   cross-entropy of the student against the `N` sampled ids.

**Unbiasedness:** `c_i` is multinomial(`N`, `t_i`), so `E[c_i/N] = t_i` for every token (head and
tail). **Gradient preservation:** `E[p_j - t^s_j] = p_j - t_j`, the full-distillation gradient.
**No up-scaling:** `t^s` is automatically normalized (`sum_i c_i/N = 1`), so the `sum t` factor is
exactly `1` and the top-`K` `1/a` inflation cannot occur. **Tail handled:** every token has
non-zero sampling probability, so nothing is structurally zeroed.

## Design choices and why

- **Sample, don't truncate.** Top-`K` zeros the tail → biased; sampling from `t` keeps support
  everywhere → unbiased. This is the core move.
- **Proposal `q ∝ t^tau`, default `tau = 1`.** Optimal IS proposal `q* ∝ t|f|` is intractable
  (needs the unknown normalizer), so use the tractable tempered family. `tau` trades coverage vs.
  per-token accuracy: `tau < 1` flattens toward uniform and increases tail coverage at higher
  variance; `tau > 1` sharpens toward the head and risks under-sampling the tail. Variance is
  lowest near `tau ∈ [0.8, 1.2]`. `tau = 1` makes importance weights `≡ 1` (just multinomial +
  count) and keeps the cache format simplest.
- **Forward KL** (not reverse KL / MSE): its gradient `(sum t) p_j - t_j` is what the
  unbiasedness-in-expectation argument is built on; it is mean-seeking, so it covers the whole
  teacher support (the tail), unlike mode-seeking reverse KL.
- **`N ≈ 50` rounds → ≈ `12` unique tokens.** Unique-token count grows sublinearly in `N`
  (coupon-collector / power-law), since head tokens repeat. This gives many averaging rounds while
  keeping the stored target near the desired dozen-entry scale.
- **Exact `7`-bit storage.** With `N = 50`, every `t^s_i = x/50` with integer `x ≤ 50 < 2^7`, so
  the probability stores exactly in `7` bits (store the numerator); `17` bits for the vocab id
  (`V ≈ 100k`) → `24` bits = `3` bytes per unique token (no quantization error below `N = 128`).
- **Sequence alignment (offline correctness).** Teacher logits depend on prefix context; the
  teacher (caching) and student (training) must see identical packing/shuffling, else the cached
  targets correspond to the wrong context.

## Final algorithm

```
Caching (teacher, once):
  for each training position with teacher distribution t (over vocab V):
      draw N token ids ~ multinomial(t), with replacement
      c_i  <- count of token i among the N draws        # <= N nonzeros, << N unique in practice
      store the nonzero (id, c_i) pairs                  # t^s_i = c_i / N

Training (student):
  t^s_i <- c_i / N                                       # reconstruct sparse target, sums to 1
  p     <- softmax(student_logits)
  L     <- sum_i t^s_i * log(t^s_i / p_i)                # forward KL == avg CE over sampled ids
  (optionally mix:  L = alpha * L + (1 - alpha) * L_CE)
```

## Working code

The downsampler (builds the cache-able sparse target) and the loss, faithful to the canonical
pseudocode (`torch.multinomial` with replacement, scatter counts/`N`, KL of student log-probs
against the sparse target):

```python
import torch
import torch.nn.functional as F


def downsample_random_sampling(teacher_probs, N=50):
    """RS-KD downsampler (tau = 1): sample N token ids ~ teacher, accumulate counts/N.
    Unbiased estimate of the teacher: E[c_i / N] = t_i for every token i."""
    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)      # [B, N]
    values = torch.full((teacher_probs.size(0), N), 1.0 / N,                 # each draw -> 1/N
                        device=teacher_probs.device, dtype=teacher_probs.dtype)
    sparse_target = torch.zeros_like(teacher_probs)                          # [B, V]
    sparse_target.scatter_add_(1, sampled_idx, values)                       # accumulate counts/N
    return sparse_target                                                     # sums to 1, <= N nonzeros


def distillation_loss(student_logits, sparse_target):
    """Forward KL( sparse_target || student )."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, sparse_target, reduction="batchmean")
```

As a single distillation-loss body that takes student/teacher logits over the same tokens, with
`labels` marking completion positions (`-100` on padding/prompt):

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(student_logits, teacher_logits, labels=None,
                         N=50, reduction="batchmean"):
    # RS-KD: sample N ids ~ teacher per position, train on the empirical counts/N target.
    # tau = 1 -> proposal is the teacher, importance ratio t_i/q_i == 1, weight 1/N per draw.
    B, T, V = student_logits.shape
    teacher_probs = F.softmax(teacher_logits, dim=-1).reshape(B * T, V)

    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)      # [B*T, N]
    sparse_target = torch.zeros_like(teacher_probs)                          # [B*T, V]
    sparse_target.scatter_add_(
        1, sampled_idx,
        torch.full_like(sampled_idx, 1.0 / N, dtype=teacher_probs.dtype),    # each draw -> 1/N
    )                                                                        # t^s, sums to 1

    student_log_probs = F.log_softmax(student_logits, dim=-1).reshape(B * T, V)
    per_token = F.kl_div(student_log_probs, sparse_target,
                         reduction="none").sum(dim=-1)                       # [B*T]

    if labels is not None:                                                   # mask pad/prompt
        mask = labels.reshape(-1) != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(per_token.numel(), 1),
                             device=per_token.device, dtype=per_token.dtype)

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```

For `tau != 1`, sample from `q ∝ t^tau` and add `(t_i / q_i) / N` to the sampled token instead of
the constant `1/N`; the unnormalized importance-weighted estimator is the unbiased object. The
shipped `tau = 1` case is the multinomial counts/`N` form shown above.
