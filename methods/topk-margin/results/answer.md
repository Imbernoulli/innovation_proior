# topk_margin, distilled

`topk_margin` is a position-selection (demasking) strategy for masked / absorbing-state
diffusion language models. At each denoising step the model predicts, in one forward pass, a
vocabulary distribution at every masked position. `topk_margin` scores each masked position by
its **margin** — the gap between the top-1 and top-2 predicted probabilities — and **unmasks the
positions with the largest margin**, writing the argmax token. It is the active-learning *margin*
criterion (Scheffer et al. 2001), used in the certainty direction (keep the largest gaps), in
place of the max-probability confidence used by MaskGIT and LLaDA.

## Problem it solves

A masked-diffusion LM decodes by predicting all masked positions in parallel and committing only
a budgeted subset per step (the schedule fixes *how many*; the training objective fixes nothing
about *which*). Committed tokens become irreversible bidirectional context, so the
position-selection rule decides decoding quality. The rule must (a) prefer positions the model is
genuinely sure of, (b) be a cheap vector op on the one-pass logits (no extra forward passes), and
(c) work both fully-parallel (`block_length == gen_length`) and block-by-block.

## Key idea

For each masked position with predicted distribution `P_i`, define the margin

```
margin_i = P_i(top1) - P_i(top2)
```

the difference between the largest and second-largest probabilities. Unmask the `k` eligible
positions (masked, in the current block) with the **largest** `margin_i`; assign each its argmax
token. In the harness below, `k` comes from `get_num_transfer_tokens`; in the diffusion-step
generation loop, it is the current `floor(num_mask * (1 - s/t))` transfer count, with all
remaining masks transferred on the last step.

Why the margin, and not the alternatives:

- **vs random** (the faithful marginal sampler): random commits coin-flip tokens early; those
  poison the context every later step conditions on. Margin commits decisive positions first.
- **vs max-probability** `max_v P_i(v)` (MaskGIT confidence / LLaDA `low_confidence`): a function
  of one number, the winning mass. It cannot distinguish a decisive top token (`0.45` vs `0.02`)
  from a near-tie (`0.45` vs `0.44`) — both score `0.45` — yet the near-tie is the dangerous
  irreversible commit. Margin scores them `0.43` vs `0.01`, the ranking you want. Margin adds back
  exactly the second-best probability that max-prob discards — the standard repair for the
  least-confident criterion's "throws away the rest of the distribution" shortcoming.
- **vs entropy** `-sum_v P_i(v) log P_i(v)`: over a tens-of-thousands-token vocabulary, the
  entropy sum is dominated by tail tokens, so it dilutes the signal that decides correctness (top
  token clearly ahead of its rival?) with tail noise. The two-best gap is the robust certainty
  signal when there are many classes (the best-versus-second-best argument, Joshi et al. 2009).

The same scalar (margin) is *minimized* in active-learning uncertainty sampling (query the
ambiguous) and *maximized* here (commit the certain) — a certainty reading of best-versus-second-
best.

## Final algorithm

Per block (left to right), per denoising step over that block:

```
eligible = (x == MASK) within the current block
logits   = model(x)                                  # one forward pass scores all positions
P        = softmax(logits.to(float64), dim=-1)        # preserve close top-two gaps
x0       = argmax(P, dim=-1)                          # token to write
sorted_P = sort(P, descending=True)
margin   = sorted_P[...,0] - sorted_P[...,1]          # top1_prob - top2_prob
xfer     = false_like(x0)
for each row:
    margin_row[not eligible] = -inf                   # disqualify non-masked/out-of-block
    xfer[row, topk(margin_row, k)] = True              # largest-margin positions
x        = where(xfer, x0, x)                         # commit selected argmax tokens
```

`used_steps` = number of `model(x)` forward passes (the efficiency metric). Stop a block early
when it has no masked positions left.

## Working code

Filling the `DemaskDecoder.decode` slot of the masked-diffusion decoding harness:

```python
import torch
import torch.nn.functional as F


class DemaskDecoder:
    """topk_margin: unmask the eligible masked positions with the largest
    (top1_prob - top2_prob) margin, writing the argmax token. Handles both
    semi-autoregressive (block_length < gen_length) and fully-parallel
    (block_length == gen_length) decoding."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature

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
        used = 0
        for b in range(num_blocks):
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)        # per-step budget
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m                 # eligible = masked & in-block
                if not mask_idx.any():
                    break
                logits = model(x).logits                      # one forward pass
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)             # argmax token
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                margin = sorted_probs[..., 0] - sorted_probs[..., 1]   # P(top1) - P(top2)
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(margin.shape[0]):
                    m = margin[j].clone()
                    m[~mask_idx[j]] = -float("inf")           # disqualify ineligible
                    _, topk = torch.topk(m, int(num_xfer[j, step].item()))
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)                  # commit chosen positions
                used += 1
        return x, used
```

The generation-utility form uses the same confidence scalar: `sorted_probs[:, 0] -
sorted_probs[:, 1]` for the masked-position logits, then `torch.topk` over a full confidence row
whose non-masked entries are `-inf`. The sibling max-probability and entropy strategies change
only that scalar; the transfer loop is the same.
