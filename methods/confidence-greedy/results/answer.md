# Low-confidence remasking (confidence-greedy decoding), distilled

Low-confidence remasking is the decoding strategy for a masked diffusion language model: turn a
one-shot bidirectional mask predictor into an iterative decoder by committing, at each reverse step,
the masked positions the model is *most confident* about, and leaving the rest masked to be
re-predicted next step. At the default temperature zero, the token written is the model's argmax
(greedy / annealed), and the confidence is the predicted probability of that selected token. With
nonzero temperature, the same code uses a Gumbel-perturbed argmax and still scores confidence using
the original model probability of the selected token. It is the temperature-zero, likelihood-aligned
specialization of confidence-keeping iterative decoding, and it runs both
fully-parallel and semi-autoregressively (block by block).

## Problem it solves

A masked diffusion LM only exposes one primitive: given a partially masked sequence, predict every
masked token's distribution in parallel. Generation is the reverse process from a fully masked
response down to a clean one; each discretized step may commit only a schedule-fixed fraction of the
masked positions (committing all at once breaks the tokenwise-independence approximation and
degrades the joint), and committed tokens are frozen (carry-over). So each step must choose which
masked positions to fill and what to write. The exact reverse kernel would randomize both token
sampling and remasking; the released decoder instead proposes tokens greedily by default and uses
random remasking only as the position-selection baseline. That baseline preserves the mask-rate
marginal, but it wastes the predictor and can freeze low-confidence mistakes.

## Key idea

Use the per-position confidence the predictor already produced. Two arguments converge on
"commit the most confident, defer the rest":

- **Irreversibility.** A commit is frozen forever; the *cost* of error is equal across positions but
  the model-assigned error proxy is `1 - c^i`, smallest at peaked (high-confidence) positions.
  Commit those; let deferred positions sharpen as context accumulates (easy-first curriculum).
- **Independence error.** A fast step treats the committed-position proposals as conditionally
  independent given context, although the clean-data conditional across masked slots is not generally
  a product. The perturbation from committing position `i` should be smaller when its selected token
  already carries most of the mass, so the tokenwise-independence error is smallest at the
  high-confidence positions.

Greedy token assignment (argmax) is the annealed extreme of sampling: it suppresses diversity, which
is useful when there is a single correct answer. A temperature knob can re-inject diversity for
open-ended generation through the code's Gumbel proposal path. The per-step count is *uniform*
(equal positions per step) because the forward masking schedule is linear, so the expected number of
transitions per step is constant.

## The masked-diffusion reverse process it discretizes

Forward (linear schedule): `q_{t|0}(x_t^i = M) = t`, `q_{t|0}(x_t^i = x_0^i) = 1 - t`. Reverse
kernel for a step `t → s`, `s < t`, per position:

```
q_{s|t}(x_s^i | x_t) = 1                              if x_t^i ≠ M, x_s^i = x_t^i   (carry-over: frozen)
                     = s/t                            if x_t^i = M, x_s^i = M        (stay masked)
                     = ((t-s)/t) · q_{0|t}(x_s^i|x_t) if x_t^i = M, x_s^i ≠ M        (fill from data-pred)
                     = 0                              otherwise.
```

The only learned object is the data-prediction `q_{0|t}(x_0^i | x_t) = p_data(x_0^i | x_t^{UM})`,
which is time-free (no `t` input). With `N` equal steps (`s = t - 1/N`) and a linear schedule, the
per-step fill budget is constant: about `m/N` of the `m` masked positions per step.

## Final algorithm

```
x <- [prompt | gen_length × MASK]
split the generation region into blocks (gen_length % block_length == 0); steps_per_block = steps / num_blocks
for each block b, left to right:
    num_xfer <- uniform schedule over this block's masks (m_b // steps_per_block per step, remainder front-loaded)
    for step = 0 .. steps_per_block - 1:
        mask_idx <- masked positions restricted to block b           # carry-over: skip committed
        if mask_idx empty: break
        logits <- model(x)                                            # ONE forward pass, all positions
        logits_choice <- add_gumbel_noise(logits, temperature)
        x0     <- argmax_v logits_choice                              # greedy at temperature zero
        p      <- softmax(logits)                                     # confidence from original logits
        conf   <- p[x0]                                               # confidence = prob of selected token
        conf[¬mask_idx] <- -inf                                       # never (re)commit frozen / future-block slots
        k      <- num_xfer[step]
        commit the top-k positions by conf: write x0 there into x
return x, (number of forward passes used)
```

`block_length == gen_length` ⇒ one block ⇒ fully-parallel decoding; `block_length < gen_length` ⇒
semi-autoregressive. Lower forward-pass count = more efficient. For instruction-tuned models with
heavy end-of-sequence padding, optionally set the end-of-sequence token's confidence (or logit) to
`-inf` to avoid premature termination.

## Relation to prior decoding rules

- **Random remasking** = same proposal loop but choose surviving positions uniformly at random.
  Faithful to the mask-rate marginal, but not an exact full-kernel sampler when the proposal is
  greedy/Gumbel argmax; it uses none of the predictor's confidence and can freeze uncertain tokens.
- **Confidence-keeping iterative decoding for masked image transformers** (Chang et al. 2022) is the
  same Predict → score-by-confidence → keep-most-confident → remask skeleton, but with a *cosine*
  schedule and *stochastic temperature-annealed* token sampling (tuned for image diversity).
  Confidence-greedy retargets it with a likelihood-consistent *uniform/linear* schedule and a
  *greedy* token, for deterministic-answer accuracy.
- **Margin selection** scores by `top1_prob - top2_prob` instead of `top1_prob`; a sensible
  alternative confidence signal (a large gap is a cleaner "unambiguous decision" cue), but a
  second-difference statistic that is less directly tied to "probability this commitment is wrong."

## Working code

The canonical decode loop; the strategy is exactly the commit rule. `get_num_transfer_tokens` is the
given uniform-schedule helper (linear schedule ⇒ equal expected transitions per step).

```python
import torch
import torch.nn.functional as F


def get_num_transfer_tokens(mask_index, steps):
    """Linear noise schedule -> equal number of tokens unmasked per step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num


def add_gumbel_noise(logits, temperature):
    """Gumbel-max categorical sampling; temperature == 0 reduces to argmax (greedy).
    float64 because low-precision Gumbel degrades generation quality."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DemaskDecoder:
    """Confidence-greedy / low-confidence remasking decoder for a masked diffusion LM."""

    def __init__(self, mask_id, temperature=0.0, eos_id=None, suppress_eos=False):
        self.mask_id = mask_id
        self.temperature = temperature
        self.eos_id = eos_id
        self.suppress_eos = suppress_eos

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length          # == 1 -> fully parallel
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        used = 0
        for b in range(num_blocks):                      # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens((x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m            # masks in the current block only
                if not mask_idx.any():
                    break

                logits = model(x).logits                 # one forward pass: all positions
                logits_noised = add_gumbel_noise(logits, self.temperature)
                x0 = torch.argmax(logits_noised, dim=-1) # greedy (or temperature-sampled) token

                p = F.softmax(logits, dim=-1)                               # canonical confidence path
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)     # prob of selected token
                if self.suppress_eos and self.eos_id is not None:
                    conf = torch.where(x0 == self.eos_id,
                                       torch.full_like(conf, -float("inf")), conf)

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    c = conf[j].clone()
                    c[~mask_idx[j]] = -float("inf")      # frozen / out-of-block: never selected
                    _, topk = torch.topk(c, int(num_xfer[j, step].item()))   # most-confident k
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)             # commit; carry-over keeps the rest masked
                used += 1
        return x, used
```
