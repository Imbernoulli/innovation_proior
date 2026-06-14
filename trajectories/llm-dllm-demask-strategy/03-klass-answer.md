**Problem.** Both prior rungs rank a single-step score (max prob, then margin) and commit a fixed,
schedule-sized `k` per step — so `avg_steps` is pinned at the full budget (256/256/224) regardless of how
sure the model is, and a confident-but-wrong commit (high score now, overturned by still-masked context)
is frozen permanently. The goal: commit *many* tokens per step exactly when it is safe, fall back to the
schedule-sized confidence choice when it is not, training-free, in both regimes.

**Key idea (KLASS — KL-adaptive stability sampling).** Single-step certainty is necessary but not
sufficient; the missing axis is **temporal consistency** — whether the model's belief at a position has
*settled*, measured by step-to-step KL. Gate commits on **both** stability and confidence:
- confidence `conf_t^i = max_v p_t^i(v)`; KL `d_t^i = D_KL(p_t^i || p_{t+1}^i)` (current vs previous step);
- stable set `S_t = { i : all of last n consecutive KLs < kl_threshold, and conf > conf_threshold }`;
- commit *every* position in `S_t` in one forward pass; if `S_t` is empty, commit top-`u` by confidence
  (`u` = uniform schedule count). The count adapts to the model's state, so settled blocks unmask many
  tokens per pass (cutting `avg_steps`) while the fallback guarantees progress and degrades to the
  confidence baseline.

**Why these choices.** Both signals, not one: confidence-only admits confident-but-wrong commits;
KL-only admits frozen-but-mushy ones. Consecutive-step KL (not entropy/margin, which are single-step):
KL measures whether the belief stopped *changing*. History `n = 2`: `n=1` commits on a one-step fluke,
large `n` is stingy (fewer per step, lost speedup); `n=2` kills the fluke cheaply. Float64 softmax/KL:
KL of near-equal distributions is a small difference of large logs — the ~1e-2 threshold becomes noise in
float32. Training-free: every signal is the base model's own successive predictions. Defaults
`conf_threshold=0.9`, `kl_threshold=0.01`, `history_length=2` (the harness runs `conf_threshold` lower in
some scripts).

**Theoretical rationale.** For a δ-approximate model, a token `x*` correct at the resolved context `c*`
(margin `γ` over a wrong `x†`) but currently preferred against by `β` at the under-resolved `c_M` is forced
unstable: with `Δ = ½(β+γ-2δ)_+`, `TV(P_M,P_0) ≥ Δ` (test function `f = 1{x†}-1{x*}`, the δ-bound at `c*`),
then triangle inequality + Pinsker + Cauchy–Schwarz give `(1/M) Σ_t D_KL(P_t||P_{t+1}) ≥ 2Δ²/M²`. Persistent
low KL is evidence the token is *not* a will-flip token — exactly what KLASS gates on.

**What it must clear (the bar, vs the strongest prior rung — margin).** Beat margin's 0.322 math / 0.3902
HumanEval and 237 gen_ppl / 0.1124 MAUVE *while* spending strictly fewer forward passes than the pinned
full budget. If it cannot pull `avg_steps` down without giving back quality, the stability axis was wrong;
the average-KL bound predicts it can.

```python
class DemaskDecoder:
    """KLASS: stability + confidence, KL-adaptive (Kim et al., NeurIPS 2025)."""

    def __init__(self, mask_id: int, temperature: float = 0.0,
                 conf_threshold: float = 0.9, kl_threshold: float = 0.01,
                 history_length: int = 2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold
        self.kl_threshold = kl_threshold
        self.history_length = history_length

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
        V = model.lm_head.out_features if hasattr(model, "lm_head") \
                                       else model.config.vocab_size
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
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
                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)
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
                        xfer[j, rdy] = True
                    else:
                        c = conf[j].clone()
                        c[~mask_idx[j]] = -float("inf")
                        _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                        xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used
```
