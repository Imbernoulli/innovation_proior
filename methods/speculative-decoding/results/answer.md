# Speculative Decoding

## Problem

Generating `K` tokens from a large autoregressive Transformer `M_p` requires `K` *serial* forward passes — each pass emits one token, and the next pass cannot start until the current token is known. At batch size 1 each pass is bound by **memory bandwidth** (streaming the model weights and the KV cache out of HBM), not by arithmetic, so the accelerator's compute sits mostly idle. Two facts create slack: (1) scoring a *block* of `K` tokens in a single teacher-forced pass costs about the same wall-time as scoring one token (the weights are streamed once either way); (2) some next-token decisions are "easy" and a far cheaper model predicts them correctly. The goal is to cash that slack into fewer serial `M_p` passes **without changing the output distribution** and without retraining or touching the architecture.

## Key idea

Use a small, fast **approximation model** `M_q` to *guess* the next `γ` tokens autoregressively, then run the large **target model** `M_p` **once** over the whole guessed block in parallel to score every position. A novel accept/reject rule — **speculative sampling** — then decides how many guesses to keep, and provably yields tokens distributed *exactly* as if they had been sampled from `M_p` alone. Each `M_p` pass emits between 1 and `γ+1` tokens; wall-clock speedup depends on whether the acceptance rate is high enough to pay for the `γ` cheap draft passes.

All sampling variants (argmax, top-k, nucleus, temperature) are first folded into "sample from an adjusted categorical distribution," so `p` and `q` below denote those standardized distributions.

## Speculative sampling (single position)

To draw `x ~ p(x)` using a proposal `q(x)`:

1. Sample `x ~ q(x)`.
2. Accept `x` with probability `min(1, p(x)/q(x))` (i.e. always accept if `q(x) ≤ p(x)`; if `q(x) > p(x)` reject with probability `1 − p(x)/q(x)`).
3. If rejected, resample `x` from the **residual distribution** `p'(x) = norm(max(0, p(x) − q(x)))`.

**Exactness.** With acceptance probability `β = Σ_x min(p(x), q(x))`, the normalizer of `p'` is `Σ_x max(0, p−q) = 1 − Σ_x min(p,q) = 1 − β`. Then for any token `x'`:

```
P(accept, x=x') = q(x')·min(1, p(x')/q(x')) = min(q(x'), p(x'))
P(reject, x=x') = (1 − β)·p'(x')           = p(x') − min(q(x'), p(x'))
P(x=x')         = min(p(x'),q(x')) + p(x') − min(p(x'),q(x')) = p(x').   ∎
```

This holds for **any** `q`, with no coverage/support condition.

## Algorithm (one step, draft of γ)

```
Inputs: M_p, M_q, prefix
# 1. Draft γ guesses with M_q, autoregressively.
for i = 1..γ:
    q_i(x) = M_q(prefix + [x_1, ..., x_{i-1}])
    x_i    ~ q_i(x)
# 2. Score all γ+1 positions with ONE parallel M_p pass.
p_1(x), ..., p_{γ+1}(x) = M_p(prefix), ..., M_p(prefix + [x_1, ..., x_γ])
# 3. Accept the longest matching prefix.
r_1, ..., r_γ ~ U(0,1)
n = min({ i-1 | 1 ≤ i ≤ γ, r_i > p_i(x_i)/q_i(x_i) } ∪ { γ })
# 4. Fix-up token: residual on first reject, else free bonus from M_p.
if n < γ:  p'(x) = norm(max(0, p_{n+1}(x) − q_{n+1}(x)))
else:      p'(x) = p_{γ+1}(x)
t ~ p'(x)
return prefix + [x_1, ..., x_n, t]
```

Repeat until the sequence is complete. At least one token (`t`) is emitted per `M_p` pass; up to `γ+1` when all guesses are accepted.

## Analysis

**Acceptance rate.** `β = E_{x~q} min(1, p(x)/q(x)) = Σ_x min(p(x), q(x))`. Defining the total-variation distance `D(p,q) = Σ_x |p−q|/2 = 1 − Σ_x min(p,q)`, we get `β = 1 − D(p,q)`, and `α := E(β) = E[Σ_x min(p(x),q(x))]` measures how well `M_q` matches `M_p`.

**Expected tokens per step.** Under the i.i.d. assumption (acceptances independent, each with probability `α`), the number of accepted guesses is geometric capped at `γ`, and the emitted count is that plus the fix-up token:

```
E[# tokens] = Σ_{i=0}^{γ} α^i = (1 − α^{γ+1}) / (1 − α).
```

**Walltime.** With `c =` (cost of one `M_q` run)/(cost of one `M_p` run), a step costs `T(cγ + 1)` and yields `(1−α^{γ+1})/(1−α)` tokens, so the expected speedup over standard decoding is

```
(1 − α^{γ+1}) / ((1 − α)(cγ + 1)).
```

For `γ=1` this is `(1+α)/(1+c)`, so any `α > c` already yields a speedup; the optimal integer `γ` maximizes the expression and is found numerically. With a negligible-cost `M_q` (`c ≈ 0`) the speedup is `(1−α^{γ+1})/(1−α)`, bounded above by `1/(1−α)`.

**Arithmetic operations.** Total ops rise by `(1−α)(γĉ + γ + 1)/(1−α^{γ+1})` (`ĉ` = ops-per-token ratio of `M_q` to `M_p`): low `α` wastes more compute on rejected guesses. The total *memory accesses* for the target weights/KV go *down* by `(1−α^{γ+1})/(1−α)`, which is exactly the regime that matters when decoding is memory-bound.

**vs. rejection sampling.** Classic (von Neumann) rejection sampling accepts with probability `p(x)/(M q(x))`, `M = max_x p/q`; for a fixed prefix its expected accept rate is `Σ_x p·min_{x'} q/p ≤ Σ_x min(p,q) = β`, so its task-average accept rate is at most `α`. Speculative sampling accepts at least as often, needs no global constant `M`, and reuses the parallel target computation instead of discarding it on rejection.

## Extensions

**Lenience.** If exact equality is relaxed, compare `l q(x)` to `p(x)` with `l in [0,1]`. The case split is `1` when `lq(x) ≤ p(x)` and `p(x)/(lq(x))` otherwise, giving

```
alpha_l = E_{x~q}[p(x)/max(p(x),lq(x))]
        = (1/l) Σ_x min(p(x),lq(x))
        = Σ_x min(p(x)/l,q(x)).
```

This guarantees no token is sampled with probability more than `p(x)/l`, but it no longer preserves the exact target distribution. For argmax sampling, lenience has to be applied before converting logits into the one-hot adjusted distribution.

**Beam search.** For target beam width `w`, let the approximation model use beam width `u ≥ w` for `γ` steps, score the `w + uγ` target candidates in parallel, and accept only while `top_w(M_p) ⊆ top_u(M_q)`. That condition preserves the beam set of target-only search; after it fails, downstream helper beams are conditioned on histories that may no longer be kept.

## Code

This is the no-KV-cache variant; a cached version keeps probability histories and rolls them back to position `n+1` on rejection.

```python
import torch
from torch.nn import functional as F


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < kth[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
    return logits


def norm_logits(logits, temperature, top_k, top_p):
    # Standardize any sampling scheme into a single categorical distribution.
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs.scatter_(1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs, num_samples=1):
    return torch.multinomial(probs, num_samples=num_samples)


@torch.no_grad()
def autoregressive_decoding(prefix, model, max_len, temperature=1.0, top_k=0, top_p=0.0):
    T = prefix.shape[1] + max_len
    while prefix.shape[1] < T:
        logits = model(prefix).logits
        probs = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
        prefix = torch.cat((prefix, sample(probs)), dim=1)
    return prefix


@torch.no_grad()
def fast_decoding(prefix, helper_model, target_model, max_len, lookahead=4,
                  temperature=1.0, top_k=0, top_p=0.0):
    """Sample from target_model with the output distribution unchanged, using
    helper_model to propose lookahead tokens per target pass. Batch size 1."""
    assert prefix.shape[0] == 1
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]

        # 1. Draft: the helper proposes lookahead tokens autoregressively.
        x = prefix
        for _ in range(lookahead):
            q_logits = helper_model(x).logits
            q_next = norm_logits(q_logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat((x, sample(q_next)), dim=1)

        # Recompute q_i for the drafted positions.
        q = helper_model(x).logits
        for i in range(q.shape[1]):
            q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)

        # 2. Verify: one parallel target pass scores all lookahead+1 positions.
        p = target_model(x).logits
        for i in range(p.shape[1]):
            p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

        # 3. Accept the longest matching prefix; n = end index of valid prefix.
        is_all_accept = True
        n = prefix_len - 1
        for i in range(lookahead):
            drafted = x[:, prefix_len + i].unsqueeze(-1)
            p_i = p[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            q_i = q[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            accept_prob = torch.minimum(torch.ones_like(p_i), p_i / q_i.clamp_min(1e-12))
            if (torch.rand_like(accept_prob) <= accept_prob).item():
                n += 1
            else:
                # 4a. Rejection: resample from the residual at this position.
                residual = torch.clamp(p[:, n, :] - q[:, n, :], min=0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                t = sample(residual)
                is_all_accept = False
                break

        prefix = x[:, :n + 1]
        if is_all_accept:
            # 4b. All accepted: free bonus token from the extra target distribution.
            t = sample(p[:, -1, :])
        prefix = torch.cat((prefix, t), dim=1)

    return prefix[:, :T]
```
