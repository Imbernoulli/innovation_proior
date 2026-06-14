**Problem.** A masked diffusion LM exposes one primitive — predict every masked position's distribution
in parallel — and the decode loop freezes each committed token forever (carry-over). Each step must pick
which `k` of the masked positions in the current block to commit (with `k` fixed by the uniform
schedule) and what token to write. Random selection is the schedule-faithful sampler but uses none of
the per-position certainty the predictor produced, so it freezes coin-flip tokens early and poisons the
bidirectional context for everything decoded after.

**Key idea (low-confidence remasking / confidence-greedy).** Use the confidence the predictor already
gives. Two arguments converge on *commit the most confident, defer the rest*: (1) **irreversibility** —
a commit is permanent, the error cost is equal everywhere but the model-assigned error proxy
`1 - max_v p(v)` is smallest at peaked positions, so freeze those and let deferred positions sharpen as
context accumulates (easy-first); (2) **independence error** — a parallel step treats committed
proposals as conditionally independent, and that approximation error is smallest exactly at the
high-confidence positions. Score each masked position by `conf = p(x0)` with `x0 = argmax_v p(v)`
(at temperature zero, `conf = max_v p(v)`), commit the top-`k` by confidence, write the argmax token.

**Why these choices.** Greedy token = the annealed extreme of sampling: suppresses diversity, which
raises accuracy when there is one correct answer. Uniform per-step budget because the forward masking
schedule is linear (equal expected transitions per step) — not MaskGIT's cosine, which was tuned for
image diversity. Float64 softmax: low-precision softmax degrades MDM quality and keeps the top-`k`
ranking clean. The same rule serves both regimes (`block_length == gen_length` ⇒ fully parallel;
`block_length < gen_length` ⇒ semi-autoregressive) by only changing the in-block eligibility mask;
non-eligible positions get confidence `-inf` so the top-`k` never selects a frozen or future-block slot.

**Step-1 edit.** This is the floor: the slot is just the argmax over the float64 softmax and the
confidence read off the same softmax — no Gumbel path, no end-of-sequence suppression, no adaptive count.
It commits exactly `k` positions per step and spends the entire step budget (`used == steps`), so its
`avg_steps` is pinned at the worst value.

**What to watch.** A real improvement over random (it finally uses confidence), but the weakest
confidence-aware rule: `max_v p(v)` is one number — it cannot tell a decisive top token from one in a
dead heat with its runner-up, the most dangerous commit. Expect respectable-but-unspectacular accuracy,
full-budget `avg_steps`, and poor open-ended-text quality (greedy one-block decoding is degenerate).
That forces the margin signal at step 2.

```python
class DemaskDecoder:
    """low_confidence remasking: unmask top-k positions by confidence."""

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
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    c = conf[j].clone()
                    c[~mask_idx[j]] = -float("inf")
                    _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used
```
