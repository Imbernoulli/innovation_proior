## Research question

A masked diffusion language model generates by starting from a generation region that is entirely
`[MASK]` and iteratively unmasking over a fixed budget of `steps` denoising iterations. Its only
primitive is a one-shot bidirectional mask predictor: feed it the partially-filled sequence and a
single forward pass returns, for every still-masked position at once, a full categorical distribution
over the vocabulary. The single thing being designed is the **demasking (decoding) strategy** — the
rule that, at each step, decides (1) the *schedule* (how many masked positions to unmask), (2)
*position selection* (which masked positions), and (3) *token assignment* (what token id to place).
Everything else — the pretrained models, prompts, data, runners — is fixed.

The rule has to generalize across **two decoding regimes** from the *same* `DemaskDecoder`:
**block-based semi-autoregressive** decoding for downstream-task accuracy (`block_length < gen_length`,
one block at a time, left to right) and **fully-parallel** decoding for open-ended text generation
(`block_length == gen_length`, all positions decoded together). The number of model forward passes is
the efficiency metric, so the strategy must keep `used_steps` low.

## Prior art before the first rung (masked-diffusion decoding lineage)

The first rung reacts to this line of decoding rules for the one-shot parallel predictor. Each commits
tokens from the per-step logits without any extra forward pass; they differ only in the per-position
*signal* used to choose which masked positions to freeze.

- **Masked / absorbing-state discrete diffusion (Austin et al. 2021, D3PM; Ou et al. 2024; Sahoo et
  al. 2024).** The forward process corrupts each token to an absorbing `[MASK]` symbol; the learned
  reverse process predicts the clean token at the masked slots. Two facts are reused everywhere:
  **carry-over** (once a position is unmasked the reverse step leaves it unchanged — committed tokens
  are frozen) and the **data-prediction parameterization** (the only learned object is the clean-token
  distribution at the masked positions, time-free). Gap: the training objective constrains *what*
  token goes in a slot, never *which* slots to fill in what order — that degree of freedom is the
  whole decoding problem.
- **Random-order unmasking.** The schedule-faithful default: each step pick the budgeted number of
  masked positions uniformly at random and fill them with the model's prediction. It is the unbiased
  reverse-process sampler and gets the masking rate right, but it uses none of the per-position
  certainty the predictor already produced, so it freezes coin-flip tokens as readily as certain ones —
  and a frozen mistake corrupts the bidirectional context for everything decoded after it. Gap:
  discards the model's own confidence.
- **Confidence-keeping iterative decoding for masked image transformers (Chang et al. 2022,
  MaskGIT).** The same predict-all / score-by-confidence / keep-most-confident / re-mask-the-rest loop,
  with confidence = the probability of the sampled token, grown along a *cosine* schedule, with
  *stochastic temperature-annealed* token sampling for image diversity. Gap: tuned for images — a
  cosine schedule and stochastic sampling — not obviously the right choices for a likelihood-trained
  language model on deterministic-answer tasks.
- **Certainty signals from a predicted distribution (active-learning lineage; Settles 2009).** Three
  scalars read off a categorical: *least-confident* `1 - max_v P(v)` (top mass only), *margin*
  `P(top1) - P(top2)` (adds back the runner-up), *entropy* `-Σ P log P` (whole distribution). The
  survey's own critique — least-confident "throws away information about the remaining label
  distribution" — and the many-class argument (Joshi et al. 2009: over a huge label set the tail is
  noise relative to the two front-runners) are the cues the position-selection signal reacts to. Gap:
  these were designed to find the *most uncertain* items to query; here the same scalars must be read
  in the *certainty* direction (keep the most certain), and which scalar is right for an irreversible
  parallel commit is unsettled.

## The fixed substrate

A single masked-diffusion decode harness is frozen and must not be touched. It builds the working
sequence as `[prompt | gen_length × MASK]`, splits the generation region into blocks
(`gen_length % block_length == 0`; equal ⇒ a single block ⇒ fully parallel), divides the step budget
evenly across blocks (`steps % num_blocks == 0`), walks the blocks strictly left to right, and at each
step runs exactly one `model(x)` forward pass whose per-position `logits` feed the decision. The
constraint that committed tokens are frozen is enforced by the loop itself: each step only ever *writes*
into still-masked positions and re-checks `x == mid`, so a position, once unmasked, is never revisited.
The harness also provides, outside the editable region, `get_num_transfer_tokens(mask, steps)` — the
uniform schedule (`mask.sum() // steps` per step, remainder front-loaded), which is the principled
per-step budget because the forward masking schedule is linear (equal expected transitions per step).

## The editable interface

Exactly one region is editable — the `DemaskDecoder` class in `LLaDA/custom_demask_eval.py` (the
`__init__` plus the `@torch.no_grad() decode(...)` method). The contract:

- `__init__(self, mask_id, temperature=0.0, conf_threshold=0.9, kl_threshold=0.01, history_length=2)` —
  the constructor signature is fixed (the harness always passes these five keywords); a strategy uses
  only the knobs it needs.
- `decode(self, model, input_ids, gen_length, steps, block_length)` — returns
  `(x_output [1, prompt_len + gen_length], used_steps)`, where `used_steps` is the number of model
  forward passes (lower = more efficient). The output shape must always be `[1, prompt_len + gen_length]`.
- `get_num_transfer_tokens(mask, steps)` is available (defined outside the editable region).

Every method on the ladder is a fill of this same contract. The starting point is the scaffold default
— the KLASS reference decoder shipped in the template — but the trajectory derives the ladder from its
weakest rung, so the rungs replace this `decode` body in turn.

```python
# EDITABLE region of LLaDA/custom_demask_eval.py — DemaskDecoder (default scaffold fill)
class DemaskDecoder:
    """Masked-diffusion decoding strategy. Semi-autoregressive in blocks of
    block_length; fully parallel when block_length == gen_length."""

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
        x[:, :input_ids.shape[1]] = input_ids.clone()           # [prompt | gen_length masks]
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length                  # == 1 -> fully parallel
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):                              # blocks, left to right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)           # uniform per-step budget
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m                    # masks in current block only
                if not mask_idx.any():
                    break                                        # block filled; skip the pass
                logits = model(x).logits                         # ONE forward pass, all positions
                # TODO: from `logits`, `mask_idx`, and this step's budget num_xfer[:, step],
                #       decide which masked positions to commit and what token to write, then
                #       write those tokens into x (carry-over keeps the rest masked).
                used += 1
        return x, used                                           # used = forward passes
```

The single empty slot is the position-selection-and-assignment rule (and, for adaptive strategies, how
many to commit). The mask layout, the block walk, the uniform-budget helper, and the one-forward-pass
cost model already exist; each rung fills the slot and nothing else.

## Evaluation settings

Three pre-existing settings, all using fixed pretrained masked-diffusion LMs with prompts, data, and
runners held fixed; only the `DemaskDecoder` varies. Single seed {42}.

| Label | Task | Model | gen_len | steps | block_len | Regime | Metrics |
|-------|------|-------|---------|-------|-----------|--------|---------|
| `llada-math` | MATH-500 | LLaDA-8B-Instruct | 256 | 256 | 64 | semi-AR (4 blocks) | accuracy ↑, avg_steps ↓ |
| `llada-humaneval` | HumanEval (164) | LLaDA-8B-Instruct | 256 | 256 | 64 | semi-AR (4 blocks) | accuracy ↑, avg_steps ↓ |
| `dream-text` | C4 prefix-continuation (256 samples, 32-tok prefix → 224 continuation) | Dream-v0-Instruct-7B | 224 | 256 | 224 | fully parallel (1 block) | gen_ppl ↓, mauve ↑, entropy ↑, rep2 ↓, avg_steps ↓ |

Metric directions: `accuracy` (exact-match MATH / pass@1 HumanEval) ↑; `gen_ppl` (conditional
perplexity via GPT-2-Large) ↓; `mauve` (distributional similarity to C4 reference) ↑; `entropy`
(bigram entropy / lexical diversity) ↑; `rep2` (repeated-bigram ratio) ↓; `avg_steps` (actual model
forward passes) ↓. MATH/HumanEval use the KLASS protocol's answer extraction; the text setting follows
MDLM/ReMDM-style prefix-conditioned C4 continuation.
