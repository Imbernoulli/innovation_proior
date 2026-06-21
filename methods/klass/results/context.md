## Research question

A masked diffusion language model (MDM) generates a sequence not left-to-right but by *unmasking*.
It begins with the generation region set entirely to a special mask token and runs a fixed number of
denoising steps; at each step the network looks at the whole partially-filled sequence and produces,
for every still-masked position, a categorical distribution over the vocabulary. A *demasking
strategy* then decides three things at that step: a **schedule** (how many positions to commit this
step), a **position selection** (which masked positions), and a **token assignment** (which token id
to write). The number of steps is a budget; the fewer forward passes used, the cheaper the
generation. The question is how to design a demasking strategy that controls how many and which
positions to commit at each step, within the existing decoding harness, working both in
**semi-autoregressive block decoding** (the generation region split into blocks decoded left-to-right,
used for reasoning-task accuracy) and in **fully-parallel decoding** (the whole region treated as one
block, used for open-ended text).

## Background

**The generative process of a masked diffusion model.** Absorbing-state discrete diffusion (Austin
et al. 2021, D3PM) defines a forward process that, with a monotonically decreasing schedule
\(\alpha_t\), replaces clean tokens by an absorbing mask: \(q(\mathbf z_t\mid\mathbf x) =
\mathrm{Cat}(\mathbf z_t;\,\alpha_t\mathbf x + (1-\alpha_t)\mathbf m)\). Its posterior has a clean
two-case form — an already-revealed token is carried over unchanged, and a masked token is drawn from
\(\big((1-\alpha_s)\mathbf m + (\alpha_s-\alpha_t)\mathbf x\big)/(1-\alpha_t)\). The reverse process
is learned by a network \(\mu_\theta\) parameterizing this posterior; the simplified masked-diffusion
line (Ou et al. 2024; Sahoo et al. 2024, MDLM; Shi et al. 2024) trains it to a negative ELBO that
reduces to a weighted masked-language-modeling loss over the masked positions, and Sahoo et al. show
the continuous-time objective is invariant to the noise schedule. Two structural facts fall out and
matter for decoding. First, the *carry-over* (SUBS) parameterization fixes already-revealed tokens:
the model only ever resamples masked positions, so the generation is monotone — a position, once
written, stays. Second, when the denoiser is time-agnostic, a step at which **no** token is unmasked
leaves the sequence unchanged, so its logits can be reused and the network evaluation skipped (Ou et
al. 2024; Sahoo et al. 2024; Zheng et al. 2024). At inference the interval \([0,1]\) is discretized
into \(t_T>\dots>t_1\), the region initialized to \([\mathrm{mask}]^L\), and the chain run backward,
each masked coordinate drawn from the model's prediction.

**LLaDA and Dream — large MDMs that put this into practice.** LLaDA (Nie et al. 2025) scales a masked
diffusion LM to 8B parameters and shows it reasons; it decodes either fully-parallel or in
*block (semi-autoregressive) diffusion* — split the generation region into blocks and fill them
left-to-right — directly, with no retraining. Dream 7B (Ye et al. 2025) is a comparable
diffusion LM supporting arbitrary-order generation and a tunable quality/speed trade-off. Both expose
the same three demasking levers above and ship default position-selection heuristics.

## Baselines

These are the prior demasking strategies a new one would be measured against and would react to.

**Confidence-based parallel iterative decoding (MaskGIT, Chang et al. 2022).** The original recipe for
generating a masked sequence in a few parallel steps: predict every masked position at once, keep only
the most confident predictions (confidence = the maximum softmax probability \(\max_v p^i(v)\)),
remask the rest, and repeat under a decreasing mask schedule \(\gamma\) (cosine/concave) so fewer
positions stay masked each step.

**Low-confidence remasking (LLaDA, Nie et al. 2025).** LLaDA's default sampler is exactly this idea
carried to a large LM: predict all masked tokens, then remask the ones with the *lowest* confidence,
keeping the high-confidence ones — equivalently, unmask the top-\(k\) by max probability, where \(k\)
follows the uniform schedule \(\lfloor(\#\text{masks})/\text{steps}\rfloor\) per step. In block
diffusion this runs block-by-block.

**Top-\(k\) margin (Dream, Ye et al. 2025).** Dream offers a sharper single-step certainty signal:
rank masked positions by the *margin* between the top and second probabilities,
\(p^i_{(1)} - p^i_{(2)}\), and unmask the top-\(k\) by margin. The margin penalizes positions where
two tokens are nearly tied, which max probability alone misses.

**Confidence-threshold parallel decoding (Fast-dLLM, Wu et al. 2025).** Instead of a fixed per-step
count, unmask *every* masked position whose confidence clears an absolute threshold \(\tau\) — so the
number committed per step adapts to the model's certainty — and, if no position clears \(\tau\),
always unmask the single most-confident one to guarantee progress. It is justified by a clean result:
if for some target sequence the model's per-position marginals are all
high-confidence, \(p_\theta(x^*_{i_j}\mid E) > 1-\epsilon\), and \((n+1)\epsilon \le 1\), then greedy
parallel decoding from the product of marginals \(q\) selects the same sequence as greedy sequential
decoding from the true joint \(p\) (the bound is tight at \(\epsilon = 1/(n+1)\)), with accompanying
distance bounds between \(q\) and \(p\).

**Single-token-per-step and planner-based samplers.** The First-Hitting Sampler (Zheng et al. 2024)
draws each unmasking event's time without discretization error and unmasks exactly one token per
event. A separate line adds a "planner" or auxiliary distribution to choose what to unmask (Liu et
al. 2024; Peng et al. 2025; Kim et al. 2025) or remasks committed tokens (ReMDM, Wang et al. 2025).
Concurrent training-free certainty heuristics include SlowFast (Wei et al. 2025), entropy-bounded
EB-Sampler (Ben-Hamu et al. 2025), top-2-gap Prophet (Li et al. 2025), and Dimple (Yu et al. 2025),
all keying off some flavor of single-step certainty.

## Evaluation settings

The natural yardsticks already in use, fixed before any new strategy:

- **MATH-500** (Hendrycks et al. 2021) with **LLaDA-8B-Instruct**, generation length 256, block length
  64 (semi-autoregressive), temperature 0; metric is exact-match accuracy plus the average number of
  model forward passes used. Answer extraction uses a task-specific normalizer/comparator on
  `data/math_test.json`.
- **HumanEval** (Chen et al. 2021, 164 problems) with LLaDA-8B-Instruct, generation length 256, block
  length 64, temperature 0; metric is pass@1 plus average forward passes.
- **Open-ended text** — prefix-conditioned C4 continuation (256 samples, a 32-token prefix continued
  for 224 tokens) with **Dream-v0-Instruct-7B**, generation length 224, block length 224 (fully
  parallel), following MDLM/ReMDM-style continuation; metrics are conditional generative perplexity
  (under GPT-2-Large), MAUVE versus held-out reference text, bigram (Shannon) entropy as a lexical-
  diversity measure, repeated-bigram ratio, and average forward passes.
- Across all settings the schedule constraint is `gen_length % block_length == 0` (equal ⇒ fully
  parallel), blocks are processed strictly in order with no early-decoding into later blocks, and
  `avg_steps` (model forward passes) is reported alongside quality, lower being more efficient.

## Code framework

The strategy plugs into a fixed harness. The pretrained model, prompts, data, and task runners are all
given; the model exposes `model(x).logits` over the current sequence, and a helper
`get_num_transfer_tokens(mask, steps)` returns the *uniform* schedule — the number of tokens a plain
fixed-schedule sampler would unmask at each step, \(\lfloor(\#\text{masks})/\text{steps}\rfloor\) per
step. The one open slot is the decoder itself: a `DemaskDecoder` whose `decode` runs the denoising
loop and returns the completed sequence and the count of forward passes it actually used. The
surrounding loop structure (initialize an all-mask region with the prompt, walk blocks left-to-right,
step within each block until the block is filled, break early when a block is done) is shared
machinery; the per-step decision rule is the empty body.

```python
import torch


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DemaskDecoder:
    """Decides, at each denoising step, which masked positions to commit.
    Works in both semi-autoregressive (block_length < gen_length) and
    fully-parallel (block_length == gen_length) regimes. Returns the
    completed sequence and the number of model forward passes used."""

    def __init__(self, mask_id, temperature=0.0, **decision_kwargs):
        self.mask_id = mask_id
        self.temperature = temperature
        self.decision_kwargs = decision_kwargs
        # TODO: initialize any rule-specific fixed state

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()              # prompt + all-mask region
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):                                 # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(                     # uniform schedule
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m                       # masks in the current block
                if not mask_idx.any():
                    break                                          # block done; skip the pass
                logits = model(x).logits
                if self.temperature > 0:
                    logits = add_gumbel_noise(logits, self.temperature)
                # TODO: fill in the per-step decoding rule we will design.
                used += 1
        return x, used
```

The loop hands one full forward pass per step; the body is where the per-step commit rule will live.
