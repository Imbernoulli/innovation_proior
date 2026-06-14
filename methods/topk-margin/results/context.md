# Context: deciding which masked positions to commit in masked-diffusion language model decoding

## Research question

A masked (absorbing-state) discrete-diffusion language model generates text by the reverse of
a corruption process: start from a generation region that is entirely `[MASK]`, and over a
budget of `steps` denoising iterations turn mask tokens into real tokens until none remain. At
every iteration the model takes the current partially-filled sequence and, in a *single*
forward pass, predicts a full categorical distribution over the vocabulary at **every** masked
position at once. That is the source of the speed advantage over left-to-right generation — one
pass scores all the blanks — but it forces a decision the autoregressive setting never has to
make: if the model has just produced predictions for, say, 200 masked positions and the
schedule says only 16 may be filled on this step, **which** 16 do you commit, and what do you
freeze them to?

The decision matters because commitment is consequential. Once a position is unmasked it
becomes part of the conditioning context for every remaining masked position on every later
step (the attention is bidirectional, so a frozen token reshapes the prediction for its
neighbours forward *and* backward). Commit the wrong tokens early and the error propagates into
everything decoded afterwards; in a regime where committed tokens are not later revisited, an
early mistake is permanent. So the goal is a *position-selection rule*: a per-position scalar,
computable from the one-pass logits, whose largest values mark the positions safest to commit
this step — and a corresponding token assignment. The rule has to work both when generation is
fully parallel (one block, all positions decoded together) and when it is semi-autoregressive
(the sequence is cut into blocks decoded left-to-right), and it must be cheap: a vector
operation over the logits, no extra forward passes, since `used_steps` (the count of forward
passes) is exactly the efficiency metric.

## Background

**Masked / absorbing-state discrete diffusion.** Discrete diffusion models (Austin et al.
2021, "Structured Denoising Diffusion Models in Discrete State-Spaces", D3PM; Hoogeboom et al.
2021; Campbell et al. 2022) define a forward Markov chain on token sequences via a per-step
transition matrix `Q_t`. One choice of `Q_t` has an **absorbing state**: each token either
stays itself or jumps to a special `[MASK]` symbol and, once masked, stays masked. The
stationary distribution puts all mass on `[MASK]`, so running the forward process to completion
yields an all-mask sequence — exactly the start state for generation. The reverse process is a
learned denoiser `p_theta(x_{t-1} | x_t)` that predicts the clean tokens at the masked
positions. The continuous-time / simplified formulations of this masked-diffusion family
(Lou et al. 2023; Ou et al. 2024; Shi et al. 2025, "Simplified and Generalized Masked
Diffusion"; Sahoo et al. 2024) reduce the training objective to a weighted cross-entropy on the
masked positions,

```
L(theta) = - E_{x0, t, x_t} w(t) * sum_n 1[x_t^n = MASK] * log p_theta(x0^n | x_t),
```

i.e. the network is just a mask predictor: given a partially masked sequence it outputs, for
each masked slot, a distribution over the vocabulary. The forward noise schedule `alpha_t`
(probability a token survives unmasked to time `t`) induces, at sampling time, a per-step
*budget*: moving the time variable from `t` down to `s` should unmask in expectation a fraction
`1 - s/t` of the currently-masked tokens. So the schedule answers "how many to unmask this
step"; it says nothing about *which* ones.

**The one-pass-many-positions decode and its degree of freedom.** Because the denoiser scores
all masked positions simultaneously and independently, decoding is a loop: predict all blanks,
choose a subset to fill (per the budget), fill them, repeat. The choice of subset and of the
filled token is unconstrained by the training objective — any rule that ends with all positions
unmasked is a valid sampler. This is the open slot. Two sub-decisions live inside it: a
**position-selection** signal (rank the masked positions, fill the best few) and a **token
assignment** (what id to write — typically the per-position `argmax`).

**Decoding can be parallel or blockwise.** If the whole generation region is treated as one
block, all positions compete on every step (fully parallel). If the region is split into blocks
of `block_length` and decoded block-by-block left-to-right (`gen_length % block_length == 0`),
each step only fills within the current block — a semi-autoregressive interpolation between
full parallelism and AR-style left-to-right generation. The same position-selection rule has to
serve both.

**Certainty signals from a predicted distribution.** A long line of work outside diffusion has
asked the dual question — *which predictions can the model be trusted on?* — by reading scalars
off a predicted categorical distribution. In active learning, **uncertainty sampling** (Lewis &
Gale 1994) ranks items by how *unsure* the model is. The classic survey of these measures
(Settles, "Active Learning Literature Survey", 2009) lays out three, for a model posterior
`P(y | x)`:

- **Least confident:** rank by `1 - P(y_hat | x)`, where `y_hat` is the top class. Uses only the
  single largest probability.
- **Margin** (Scheffer et al. 2001): rank by the gap `P(y_1 | x) - P(y_2 | x)` between the two
  most probable classes `y_1, y_2`. A small gap means the top two are nearly tied (ambiguous);
  a large gap means the top class clearly dominates.
- **Entropy** (Shannon 1948): rank by `-sum_y P(y | x) log P(y | x)`, using the whole
  distribution.

The survey's own critique is the load-bearing observation: least-confident "throws away
information about the remaining label distribution," looking only at the top mass; entropy uses
the entire distribution; margin is the middle ground that adds back exactly the second-best
mass. For problems with very large label sets, Joshi, Porikli & Papanikolopoulos (2009,
"Multi-class active learning for image classification") argue the **Best-versus-Second-Best**
(BvSB) gap is the robust certainty signal precisely because, when there are many classes, the
mass spread across the long tail is noise relative to the contest between the two front-runners.
A language-model softmax is a several-tens-of-thousands-class distribution per position, so this
many-class regime is the operating point.

**Confidence-first parallel decoding already exists for images.** MaskGIT (Chang et al. 2022,
"Masked Generative Image Transformer") decodes a masked image-token grid in a handful of
parallel steps with exactly the predict-all / keep-the-confident / re-mask-the-rest loop: at
each iteration the model predicts every masked location, a **confidence** score is attached to
each (the probability of the sampled token), the most confident positions are kept, the rest are
re-masked and re-predicted next iteration, and the number kept grows along a (cosine) schedule.
MaskGIT's stated intuition is "less-to-more": early on, commit only the few positions the model
is sure of; later, with more context fixed, commit the harder ones. Its confidence score is the
single top probability — the least-confident measure above, used in the certainty direction.

## Baselines

**Random-order unmasking.** The schedule-faithful default: each step, pick the budgeted number
of masked positions *uniformly at random* and fill them (with the model's prediction). This is
the unbiased reverse-process sampler. Limitation: it ignores the model's own certainty
entirely, so it will commit a coin-flip token at a position the model barely has an opinion on
just as readily as one it is sure of — and because committed tokens condition all later steps,
the early random commitments degrade the context the rest of the decode depends on.

**Max-probability (least-confident) confidence ordering** (the MaskGIT / LLaDA rule). Attach to
each masked position the probability of its most likely token, `conf_i = max_v P_i(v)`, and
unmask the positions with the highest `conf_i`. LLaDA (Nie et al. 2025, "Large Language
Diffusion Models", an 8B masked-diffusion LM) adopts this as its `low_confidence` remasking
strategy and states it is the same approach as MaskGIT: keep the highest-probability
predictions, re-mask the lowest. Core idea: commit where the top token's probability is largest.
Limitation: `max_v P_i(v)` is a function of one number, the winning mass, and discards the shape
of the rest of the distribution. Two positions can tie on top probability — one with a single
clear runner-up far below, one with a near-equal second place — and the rule cannot tell them
apart, even though the model is decisively committed in the first case and genuinely torn in the
second. Under greedy `argmax` commitment with no later revisiting, freezing the torn position is
a frozen error.

**Entropy ordering.** Attach to each masked position the negative entropy of its full predicted
distribution, `-H_i = sum_v P_i(v) log P_i(v)`, and unmask the positions whose distributions are
peaked (low entropy). Core idea: a low-entropy distribution is a confident one; use the whole
distribution rather than just the top mass. Limitation: over a vocabulary of tens of thousands
of tokens, the entropy sum is dominated by the enormous count of near-zero-probability tail
tokens; small, roughly-equal contributions across a huge tail can move the score as much as the
contest at the top, so the signal that actually decides correctness — whether the top token
clearly beats its nearest rival — is diluted by tail noise.

## Evaluation settings

The natural yardsticks, all using fixed pretrained masked-diffusion LMs (an 8B instruction-tuned
model and a 7B instruction-tuned model) with prompts, data, and runners held fixed across
strategies; only the position-selection rule varies. Generation budget is `gen_length` tokens
over `steps` denoising iterations, with a block structure `gen_length % block_length == 0`
(equal ⇒ fully parallel).

- **Downstream-task accuracy, semi-autoregressive (block) decoding.** Grade-school-to-competition
  math word problems (MATH-500), graded by exact match of the extracted final answer; and code
  synthesis (HumanEval, 164 problems), graded by pass@1 unit tests. Block decoding
  (`block_length < gen_length`), following the blockwise protocol used for these models.
- **Open-ended generation, fully-parallel decoding.** Prefix-conditioned continuation of web
  text (a C4 sample): given a short prefix, continue for a few hundred tokens, all positions
  decoded together (`block_length == gen_length`). Quality read out as conditional perplexity
  under an external small LM, distributional similarity to reference text (MAUVE), bigram
  entropy / repeated-bigram rate for lexical diversity.
- **Efficiency.** `avg_steps` = the number of model forward passes actually used (fewer is
  better), reported alongside every quality metric, since the position-selection rule and the
  schedule together determine how fast the masks clear.

## Code framework

The decoder plugs into a fixed harness. The model, mask token id, prompt, generation length,
step count and block length are all given; a helper that turns a per-step budget into a count of
positions to fill is provided (it returns `mask.sum() // steps` per step, a uniform schedule).
What is *not* fixed — the single empty slot — is the rule that, given the one-pass logits at the
currently-masked positions, decides which of them to unmask this step and what token to place.
Everything around that slot is generic masked-diffusion decoding machinery that already exists:
build the working sequence by padding the prompt with `[MASK]` over the generation region, walk
the blocks left-to-right, and within each block run the denoising steps, each one a single
`model(x)` forward pass whose logits feed the slot.

```python
import torch
import torch.nn.functional as F


def get_num_transfer_tokens(mask, steps):
    """Provided helper: uniform schedule — how many masked positions to unmask
    per step. Returns an integer count for each of the `steps` steps such that
    they sum to mask.sum() (i.e. mask.sum() // steps per step)."""
    ...


class DemaskDecoder:
    """Decodes a masked-diffusion LM. Per step the model scores all masked
    positions in one forward pass; the selection rule below decides which to
    commit and to what. Works fully-parallel (block_length == gen_length) or
    block-by-block (block_length < gen_length)."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        # working sequence = prompt followed by a fully-masked generation region
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):                       # blocks, left to right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)    # per-step budget
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m             # eligible = masked & in-block
                if not mask_idx.any():
                    break
                logits = model(x).logits                  # one pass scores all blanks
                # TODO: from `logits` at the masked positions, decide which of the
                #       budgeted positions to unmask this step and what token to
                #       place, then write those tokens into x.
                #       (compute a per-position selection signal + a token id,
                #        unmask the top `num_xfer[:, step]` eligible positions.)
                used += 1
        return x, used                                    # used = forward passes
```

The single empty slot is the position-selection-and-assignment rule; the loop, the block walk,
the budget helper, and the forward pass already exist.
