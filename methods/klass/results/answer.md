# KLASS, distilled

KLASS (KL-Adaptive Stability Sampling) is a training-free demasking strategy for masked diffusion
language models. At each denoising step it commits a masked position only when the model's prediction
there is **both** high-confidence (top probability above a threshold) **and** dynamically stable (its
step-to-step KL divergence has stayed below a threshold for the last few steps). Committing every such
position in parallel lets it unmask many tokens per step at "settled" steps and fall back to the
uniform schedule when none are ready, cutting the number of model forward passes while avoiding the
premature, confident-but-wrong commits that fixed-schedule confidence-only samplers make. It adds no
extra model, no training, and only a flat per-step memory buffer.

## Problem it solves

A masked diffusion LM generates by iteratively unmasking from an all-mask region; each commit is
permanent (carry-over parameterization). Committing many tokens per step on a fixed schedule is fast
but locks in errors the still-masked context would have fixed; committing one per step is safe but as
slow as autoregressive decoding. The goal is to commit *many* tokens per step exactly when it is safe,
fall back to the schedule-sized confidence choice when nothing is ready, and stay training-free, with
no auxiliary model or growing memory, working in both semi-autoregressive block decoding and
fully-parallel decoding.

## Key idea

Confidence (max softmax probability) is a single-step snapshot and is *necessary but not sufficient* —
a token can be confidently wrong while the context that would overturn it is still masked. The missing
axis is **temporal consistency**: how much the model's distribution at a position moved between steps,
measured by KL divergence. KLASS gates unmasking on **both** signals.

- **Confidence score:** \(\mathrm{conf}_t^i = \max_v p_t^i(v)\).
- **KL score:** \(d_t^i = D_{\mathrm{KL}}(p_t^i \,\|\, p_{t+1}^i)\) — current step's distribution vs.
  the previous step's, at the same position.
- **Stable set** (history length \(n\), KL threshold \(\epsilon_{\mathrm{KL}}\), confidence threshold
  \(\tau\)):
  \[
  S_t = \Big\{ i \;\Big|\; \big(\forall k\in\{1,\dots,n\}:\, D_{\mathrm{KL}}(p_{t+k-1}^i \| p_{t+k}^i) < \epsilon_{\mathrm{KL}}\big) \;\wedge\; \mathrm{conf}_t^i > \tau \Big\}.
  \]
  Requiring *all* of the last \(n\) consecutive KLs to be low (not just the latest) demands sustained
  stability, robust to a single fluke step.
- **Unmasking rule:**
  \[
  x_t^i = \begin{cases} \text{commit }i, & i \in S_t,\\ \text{commit the Top-}u\text{ positions by } \mathrm{conf}_t^i, & S_t = \emptyset, \end{cases}
  \]
  where \(u\) is the fixed (uniform-schedule) fallback count. The fallback guarantees progress and
  degrades gracefully to a plain confidence sampler when nothing is stable; at settled steps \(S_t\) is
  large and many tokens are committed in one forward pass.

## Why the design choices

- **Both signals, not one.** Confidence-only is the prior art (MaskGIT, LLaDA, Dream margin, Fast-dLLM
  threshold) and admits confident-but-wrong commits; KL-only could commit a frozen-but-mushy low-
  probability position. Together: "the model has settled on this *and* is sure of it."
- **Consecutive-step KL, not entropy/margin.** Those are single-step certainty (covered by confidence);
  KL measures whether the belief has *stopped changing* — the temporal axis the theory is about.
- **History length \(n=2\) (default).** \(n=1\) commits on a single coincidental low KL (premature);
  large \(n\) is stricter but unmasks fewer tokens per step, losing the speedup. \(n=2\) kills the
  one-step fluke cheaply. The implementation forces stability off for the first \(n-1\) steps, and
  the first zero-origin KL remains too large to pass the all-low-KL test until it rolls out of the
  buffer.
- **Float64 softmax/KL.** KL of near-equal distributions is a small difference of large logs;
  low precision turns the \(\sim 10^{-2}\) threshold into noise (and low-precision softmax hurts MDM
  quality generally).
- **Training-free / no planner.** Every signal is the base model's own successive predictions — no
  second model to train or align, only the previous-distribution and KL-history buffers.
- **Canonical thresholds:** \(\tau = 0.9\), \(\epsilon_{\mathrm{KL}} = 0.01\), \(n = 2\). The
  implementation exposes \(\tau\) and \(\epsilon_{\mathrm{KL}}\) as tunable parameters.

## Theoretical rationale

For a model that is a conditional \(\delta\)-approximation of the task-correct conditionals, fix a
position \(i\) with a correct token \(x^\star\) that beats a suboptimal \(x^\dagger\) by margin \(\gamma\)
at the resolved context \(c^\star\), while the model currently prefers \(x^\dagger\) by margin \(\beta\)
at the under-resolved context \(c_M\). Along any denoising path \(c_M\to\cdots\to c_0=c^\star\) (changing
only variables outside \(i\)), with \(P_t = p_\theta(\cdot\mid c_t)\) and \(\Delta = \tfrac12(\beta+\gamma-2\delta)_+\):
\[
\mathrm{TV}(P_M, P_0) \ge \Delta, \qquad \frac1M \sum_{t=0}^{M-1} D_{\mathrm{KL}}(P_t \,\|\, P_{t+1}) \ge \frac{2\Delta^2}{M^2}.
\]
*Proof sketch.* With \(f = \mathbf 1\{x_i=x^\dagger\} - \mathbf 1\{x_i=x^\star\}\) (\(\|f\|_\infty\le1\)),
write \(A_M=p^\dagger(c_M)-p^\star(c_M)\) and \(A_0=p^\dagger(c^\star)-p^\star(c^\star)\). Total variation gives
\(2\,\mathrm{TV}(P_M,P_0) \ge |\mathbb E_{P_M}[f]-\mathbb E_{P_0}[f]| = |A_M-A_0|\). The model margin gives
\(A_M\ge\beta\). The \(\delta\)-approximation at \(c^\star\) gives
\(p^\star(c^\star)-p^\dagger(c^\star)\ge \gamma-2\delta\), hence \(A_0\le-(\gamma-2\delta)\). Therefore
\(A_M-A_0\ge\beta+\gamma-2\delta\), and the useful lower bound is the positive part:
\(\mathrm{TV}(P_M,P_0)\ge\tfrac12(\beta+\gamma-2\delta)_+=\Delta\).
The triangle inequality gives \(\sum_t \mathrm{TV}(P_{t+1},P_t) \ge \Delta\); Pinsker
(\(D_{\mathrm{KL}}(P_t\|P_{t+1}) \ge 2\,\mathrm{TV}(P_{t+1},P_t)^2\)) plus Cauchy–Schwarz
(\((\sum_t T_t)^2 \le M\sum_t T_t^2\)) yields the average-KL bound. A wrong-but-correctable token is
therefore *forced* to be dynamically unstable (average per-step KL bounded away from 0), so persistent
low KL is evidence against the token being a will-flip token — which is exactly what KLASS gates on.

## Final algorithm

```
initialize x = [prompt, MASK, ..., MASK]; p_prev = 0; kl_buffer = 0 (length n)
for each block left-to-right:
    u_t = uniform schedule (# masks // steps_per_block) for fallback
    for step in block:
        if block fully unmasked: break            # skip the forward pass
        logits = model(x).logits
        if temperature > 0: logits = add_gumbel_noise(logits, temperature)
        p_curr = softmax(logits.to(float64))
        conf   = max_v p_curr ; x0 = argmax p_curr
        kl     = D_KL(p_curr || p_prev)            # per position
        roll kl_buffer left; kl_buffer[-1] = kl ; p_prev = p_curr
        stable = (step >= n-1) and all(kl_buffer < eps_KL)
        ready  = stable and (conf > tau) and is_masked
        if any ready: commit x0 at all ready positions
        else:         commit x0 at Top-u positions by conf
return x, num_forward_passes
```

## Working code

Faithful to the canonical implementation (`alg="klass"`, `unmask_strategy="all"`), filling the
`DemaskDecoder` slot of the masked-diffusion decoding harness.

```python
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DemaskDecoder:
    """KLASS: KL-adaptive stability sampling. Commit a masked position only when
    its prediction is both high-confidence and dynamically stable (the last
    `history_length` consecutive-step KL divergences are all below kl_threshold).
    Works in semi-autoregressive (block_length < gen_length) and fully-parallel
    (block_length == gen_length) regimes; returns (sequence, forward passes)."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold        # tau
        self.kl_threshold = kl_threshold            # eps_KL
        self.history_length = history_length        # n

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        V = model.lm_head.out_features if hasattr(model, "lm_head") \
                                       else model.config.vocab_size
        # rolling window of last `history_length` per-position KLs; previous-step
        # categorical at every position (float64: KL of near-equal distributions
        # is a small difference of large logs).
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
        used = 0
        for b in range(num_blocks):
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(             # fixed fallback count u per step
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m               # masks in the current block only
                if not mask_idx.any():
                    break                                   # block filled; skip the pass

                logits = model(x).logits
                if self.temperature > 0:
                    logits = add_gumbel_noise(logits, self.temperature)
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)

                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)  # D_KL(p_curr || p_prev)
                kl_hist = torch.roll(kl_hist, -1, dims=-1)
                kl_hist[..., -1] = kl
                p_prev = p_curr.clone()

                if step >= self.history_length - 1:
                    stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
                else:
                    stable = torch.zeros_like(conf, dtype=torch.bool)
                ready = stable & (conf > self.conf_threshold) & mask_idx

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(ready.shape[0]):
                    rdy = torch.where(ready[j])[0]
                    if len(rdy) > 0:
                        xfer[j, rdy] = True                 # commit the whole stable+confident set
                    else:
                        c = conf[j].clone()
                        c[~mask_idx[j]] = -float("inf")     # fallback: Top-u by confidence
                        _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                        xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used                                      # used = model forward passes
```
