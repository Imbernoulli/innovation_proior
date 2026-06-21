# Context: decoding a masked diffusion language model (circa 2024-2025)

## Research question

A masked diffusion language model is trained to take a partially masked sequence and predict,
*in parallel*, the clean token at every masked position at once. That is its only primitive: one
forward pass returns a categorical distribution over the vocabulary for each masked slot,
conditioned bidirectionally on all currently-visible tokens. The trained model is supposed to
define a generative distribution `p_θ(r_0 | p_0)` over a response `r_0` given a prompt `p_0`, by
*simulating a reverse process* that starts from a fully masked response and gradually fills it in.

The question is on the inference side: how do you turn that one-shot parallel predictor into a
sequence of denoising steps that produces text? The reverse process is discretized into a fixed
number of steps `N`, and at each step you may commit a fraction of the masked positions (the
fraction is pinned by the requirement that the chain match the forward masking marginals). So each
step has three sub-decisions: (1) *schedule* — how many masked positions to fill this step;
(2) *position selection* — which masked positions to fill; (3) *token assignment* — what token to
write at each. A decoding strategy answers all three, runs whether you decode the whole response in
one parallel block or carve it into left-to-right sub-blocks (semi-autoregressive), counts model
forward passes (each step is one full forward pass — the dominant cost), and is used both on tasks
with a single correct answer (math, code) and on open-ended generation.

Filling positions *uniformly at random* each step is the position-selection rule that preserves the
reverse-process masking rate; with the usual greedy or Gumbel token proposal, it keeps the remasking
marginal aligned. Filling *all* positions in a single step commits many tokens at once. The question
is what selection-and-assignment rule to put in the one empty slot of the decode loop.

## Background

**Masked / absorbing-state discrete diffusion.** Discrete diffusion defines a forward process that
gradually corrupts a clean sequence `x_0` and a learned reverse process that undoes it. Austin et
al. (2021, D3PM) built discrete diffusion as a Markov chain `q(z_t | z_{t-1}) = Cat(z_t; Q_t
z_{t-1})` over a transition matrix `Q_t`, and found empirically that the **absorbing-state** choice
— a special `[MASK]` symbol that, once a token transitions into it, the token stays there forever —
consistently performs best among discrete forward processes. In language modeling that absorbing
state is the same `[MASK]` token familiar from masked language modeling.

Sahoo et al. (2024, MDLM) and the parallel works specialized this to masking and derived a tight,
low-variance objective. With a strictly decreasing schedule `α_t` (from `α_0 ≈ 1` to `α_1 ≈ 0`),
the forward marginal is `q(z_t | x) = Cat(z_t; α_t x + (1-α_t) m)`, i.e. each token is intact with
probability `α_t` and `[MASK]` with probability `1-α_t`. The reverse posterior collapses to a clean
form (their Eq. 6):

```
q(z_s | z_t, x) = Cat(z_s; z_t)                                  if z_t ≠ m,
                = Cat(z_s; [(1-α_s) m + (α_s - α_t) x] / (1-α_t)) if z_t = m,    (s < t)
```

Two structural facts fall out and are reused everywhere. **Carry-over unmasking:** if a position is
already unmasked (`z_t ≠ m`), the reverse step leaves it unchanged — you never overwrite a token you
have already committed. **Data-prediction parameterization:** the only thing the network must
estimate is the clean-data distribution at the masked positions; one replaces the unknown `x` in the
posterior with a neural prediction `x_θ(z_t, t)`, the "SUBS" parameterization
`p_θ(z_s | z_t) = q(z_s | z_t, x = x_θ(z_t, t))`. Ou et al. (2024) sharpened this further: for an
absorbing process the data-prediction function is *time-independent* —
`q_{0|t}(x_0^i | x_t) = p_data(x_0^i | x_t^{UM})`, a function only of the unmasked context `x_t^{UM}`
— so the time `t` need not even be fed to the network, and the predictor is exactly a
bidirectional masked-token classifier.

Concretely, with a *linear* schedule (mask probability proportional to `t`, motivated by treating
information as roughly proportional to token count) the per-token reverse kernel for `0 ≤ s < t ≤ 1`
is

```
q_{s|t}(x_s^i | x_t) = 1                              if x_t^i ≠ M and x_s^i = x_t^i,
                     = s/t                            if x_t^i = M and x_s^i = M,
                     = ((t-s)/t) · q_{0|t}(x_s^i|x_t) if x_t^i = M and x_s^i ≠ M,
                     = 0                              otherwise.
```

So a masked position stays masked with probability `s/t` and gets filled (from the data-prediction
head) with probability `(t-s)/t`. Iterating this kernel from `t = 1` (all masked) down to `t = 0`
is the reverse process; the network is trained by a masked cross-entropy (a likelihood bound), and
any practical decoder has to choose a tractable discretization of this kernel.

**The parallel-decoding approximation.** The reverse kernel above factorizes across positions, so a
single step samples every masked position *independently* given the current context. This is the
tokenwise-independence approximation (Tweedie tau-leaping): it makes parallel decoding fast, and it
is approximate, because once you commit position `i` the correct conditional for position `j`
changes, and the independent sample did not account for that. The deviation grows with the number of
tokens committed per step, which is what the per-step budget controls.

**Greedy / annealed sampling.** For autoregressive LMs it is well established (Holtzman et al. 2019;
Brown et al. 2020) that *how* you read a token off the predicted distribution trades diversity
against fidelity. Greedy decoding (`argmax`, the mode) and low-temperature ("annealed") sampling
suppress diversity; nucleus / high-temperature sampling preserves it. On open-ended generation,
less diversity reads as more repetitive text; on tasks with a *single* correct answer, less
diversity tends to *raise* accuracy. Any decoding rule for a diffusion LM inherits this trade-off,
position by position.

## Baselines

**Random remasking (the mask-marginal-faithful position rule).** Run the predictor, form a token
proposal at every masked position (the released code uses the argmax of the logits, optionally after
Gumbel perturbation), and for each currently-masked position keep it masked with probability `s/t`;
equivalently, choose which proposals survive uniformly at random subject to the per-step transfer
count. This preserves the intended remasking rate, and is a faithful *position* rule once the token
proposal is greedy. It treats every masked position as equally eligible for commitment, regardless
of which proposals the predictor produced.

**Confidence-based parallel decoding for masked image transformers (Chang et al. 2022, MaskGIT).**
In the vision setting (masked visual-token modeling) the same one-shot parallel predictor is
decoded over `T` iterations with a four-stage step: *Predict* all masked locations' probabilities in
parallel; *Sample* a token at each masked location, and record its predicted probability as a
"confidence" score (unmasked positions get confidence `1.0`); *Mask-schedule* — compute how many
tokens to keep masked via a schedule function `γ(t/T)` that is continuous, decreasing, with
`γ(0) → 1`, `γ(1) → 0` (cosine works best, a concave "few confident predictions early, many late"
curve); *Mask* — keep the most confident tokens and remask the rest,
`m_i^{(t+1)} = 1 [c_i < sorted_j(c_j)[n]]`. The contribution is that "the model predicts all tokens
simultaneously but only keeps the most confident ones," re-predicting the rest next iteration. To
encourage diversity in image synthesis it samples the token stochastically with temperature
annealing. This result is for images, with a *cosine* schedule and *stochastic* temperature-annealed
token sampling.

**Margin-based selection (an alternative score).** Instead of the predicted probability of the top
token, score each masked position by the *margin* between its top-1 and top-2 predicted
probabilities, and commit the largest-margin positions. The idea is that a large gap is a signal of
an unambiguous decision; the margin is a second-difference statistic, sensitive to the tail shape of
the distribution.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **MATH-500** — competition-math problems; metric is exact-match accuracy of the extracted final
  answer, plus the average number of model forward passes (lower = more efficient). Generation length
  and step count are fixed; decoding is semi-autoregressive (block length dividing the generation
  length).
- **HumanEval (164 problems)** — Python program synthesis; metric is functional correctness (pass@1
  by executing unit tests), plus average forward passes. Same block-decoding regime.
- **Open-ended continuation** — given a short prefix, continue the text; measured by conditional
  perplexity under a separate reference LM, distributional similarity to reference text (MAUVE),
  lexical-diversity (bigram entropy), repeated-bigram ratio, and average forward passes. Here
  decoding is fully parallel (one block spanning the whole generation region).
- Protocol shared by all: a fixed pretrained-then-instruction-tuned mask predictor; fixed prompts
  and data; the generation length must be divisible by the block length; blocks are processed
  strictly left-to-right (no decoding into a later block before the current one is done); the *same*
  decoder must run in both the block and the fully-parallel regimes. For instruction-tuned models,
  heavy end-of-sequence padding in the fine-tuning data can make the decoder terminate too early, so
  the end-of-sequence token's confidence may need to be suppressed during sampling.

## Code framework

The decoder plugs into a fixed harness: a pretrained mask predictor `model` whose forward pass
returns per-position vocabulary `logits`; a tensor of input ids for the prompt; and a helper that is
*given* (not part of the strategy) for turning a per-step budget into an actual count. Because the
forward masking schedule is linear, the expected number of positions that transition per step is
constant, so the helper `get_num_transfer_tokens(mask, steps)` returns a *uniform* schedule
(`mask.sum() // steps` per step, remainder spread over the first steps). The harness lays out the
generation region as `[prompt | gen_length × MASK]`, splits it into blocks (`gen_length %
block_length == 0`; one block spanning everything is the fully-parallel case), and at each step runs
one forward pass. What it does *not* yet contain is the rule that decides, given this step's
predictions and budget, which masked positions to commit and what token to write — that is the one
empty slot.

```python
import torch
import torch.nn.functional as F


# GIVEN, outside the strategy: linear schedule -> uniform per-step unmask budget.
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num


class DemaskDecoder:
    """Turn a one-shot parallel mask predictor into an iterative decoder."""

    def __init__(self, mask_id, temperature=0.0):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        # [prompt | gen_length masks]
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length          # == 1  -> fully parallel
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):                      # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)   # per-step budget
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m            # masks in the current block
                if not mask_idx.any():
                    break
                logits = model(x).logits                 # one forward pass: all positions at once
                # TODO: given `logits`, `mask_idx`, and this step's budget
                #       num_xfer[:, step], decide which masked positions to
                #       commit and what token to write into them; update x.
                used += 1
        return x, used
```

The single empty slot is the commit rule. Everything around it — the mask layout, the block loop,
the uniform budget, the one-forward-pass-per-step cost model — already exists.
