**Problem.** Confidence-greedy ranks masked positions by `max_v p(v)` — a single number, the winning
mass. It cannot distinguish a decisive top token (0.45 vs a 0.02 runner-up) from a near-tie (0.45 vs
0.44); both score 0.45. Under argmax commitment with no revisiting, freezing the near-tie permanently
kills an almost-equally-good alternative and poisons the bidirectional context — the dominant failure in
the fully-parallel text regime, where the block is the whole region and early near-tie commits do the
most damage.

**Key idea (topk_margin).** Add back exactly the piece `max_v p(v)` discards — the second-place
probability — and rank by the *gap* `margin = P(top1) - P(top2)`. A large gap means the top token clearly
dominates its nearest rival (safe to freeze); a small gap means a near-tie (leave masked until context
resolves it). Commit the top-`k` by largest margin, write the argmax token. This is the active-learning
margin criterion read in the *certainty* direction (keep the largest margins, the reverse of querying the
smallest) and the `topk_margin` selection option in Dream's diffusion-LM decoding.

**Why margin, not entropy.** Over a tens-of-thousands-token vocabulary, entropy's sum is swamped by tail
noise — two positions with identical top-two structure can rank differently purely from how negligible
mass is smeared across the long tail. The two-best gap is the robust certainty signal in the many-class
regime; full entropy is the noisy one. Margin adds back the one missing piece (runner-up) and ignores the
tail. Token = argmax (greedy mode, raises accuracy on single-answer tasks). Float64 softmax is now
load-bearing: margin is a *difference* of two masses, exactly where low precision bites.

**Step-2 edit.** Pure selection-signal swap on the floor scaffold: keep the uniform budget, block walk,
carry-over, eligibility masking, and argmax token; change only the ranked score from `conf` (one sorted
prob) to `margin` (two sorted probs). One forward pass per step, one sort, one top-`k`, one masked write.

**What to watch.** Clearest win expected on open-ended text (MAUVE up from 0.0321 — not poisoning context
with frozen coin flips), with gen_ppl possibly still high (greedy one-block decoding, no diversity
injection; lower perplexity ≠ higher distributional match). Small real lift on accuracy (HumanEval the
likelier mover — code has many genuine near-ties). `avg_steps` stays pinned at full budget (256/256/224):
margin changes *which* `k` to commit, not *how many*. That fixed-count cost is what forces the adaptive,
stability-gated count at step 3.

```python
class DemaskDecoder:
    """topk_margin: unmask top-k positions by (top1_prob - top2_prob)."""

    def __init__(self, mask_id: int, temperature: float = 0.0,
                 conf_threshold: float = 0.9, kl_threshold: float = 0.01,
                 history_length: int = 2):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length: int, steps: int,
               block_length: int):
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
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m
                if not mask_idx.any():
                    break
                logits = model(x).logits
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                margin = sorted_probs[..., 0] - sorted_probs[..., 1]
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(margin.shape[0]):
                    m = margin[j].clone()
                    m[~mask_idx[j]] = -float("inf")
                    _, topk = torch.topk(m, int(num_xfer[j, step].item()))
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used
```
