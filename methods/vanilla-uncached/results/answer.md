# LLaDA (Large Language Diffusion with mAsking), distilled

LLaDA is a masked-diffusion language model: a bidirectional Transformer trained to predict masked
tokens, defining a generative distribution `p_θ(x_0)` through a forward token-masking process and a
learned reverse unmasking process. It keeps the maximum-likelihood principle of language modeling
(`max_θ E_{p_data}[log p_θ(x)]`) while dropping the autoregressive left-to-right factorization. The
"vanilla, uncached" form is the plain reference: a full bidirectional forward over the entire
sequence at every denoising step, with low-confidence remasking and semi-autoregressive (block)
decoding, reusing no cached computation.

## Problem it solves

A principled, likelihood-based generative model of text that conditions bidirectionally instead of
on a fixed causal prefix — recovering scalability and in-context learning from the generative
principle rather than from next-token prediction, and removing the directional bias (the reversal
curse) that the left-to-right factorization bakes in.

## Forward and reverse process

Continuous time `t ∈ [0,1]`, factorized across positions. Forward (independent per-token masking):

```
q_{t|0}(x_t^i | x_0^i) = 1 - t   if x_t^i = x_0^i       (token survives)
                       = t       if x_t^i = M           (token masked)
```

At `t=0` nothing is masked; at `t=1` all is masked (the noise endpoint). Reverse posterior for
`0 ≤ s < t ≤ 1`:

```
q_{s|t}(x_s^i | x_t) = 1                            if x_t^i ≠ M, x_s^i = x_t^i   (revealed stays)
                     = s/t                          if x_t^i = M, x_s^i = M       (stays masked)
                     = (t-s)/t · q_{0|t}(x_s^i|x_t) if x_t^i = M, x_s^i ≠ M       (gets revealed)
                     = 0                            otherwise.
```

The `s/t` / `(t-s)/t` fractions are forced by matching the forward marginals. The only learnable
piece is the data-prediction `q_{0|t}`, and it is **time-independent**: a masked token's clean value
depends only on the unmasked context (which equals clean data), so
`q_{0|t}(x_0^i | x_t) = p_data(x_0^i | x_t^{UM})`. Hence the predictor `p_θ(· | x_t)` takes no
timestep input and is a plain **bidirectional** Transformer (no causal mask). This any-order
conditioning is what fixes the reversal curse.

## Key idea: the 1/t-weighted objective is an NLL upper bound

Train a mask predictor with cross-entropy on the masked positions only:

```
L(θ) = -E_{t, x_0, x_t} [ (1/t) · Σ_{i=1}^L 1[x_t^i = M] · log p_θ(x_0^i | x_t) ],   t ~ U(0,1].
```

This is an upper bound on the negative log-likelihood of the model distribution:

```
-E_{p_data(x_0)} [ log p_θ(x_0) ]  ≤  L(θ).
```

**Where the `1/t` comes from.** Specializing the discrete-diffusion variational bound to absorbing
(masking) corruption, with two structural properties of the denoiser — never predict `M`
(`⟨p_θ, m⟩ = 0`) and carry over already-revealed tokens — collapses the dense per-step KL to a
weighted sum of masked-token cross-entropies. For a reverse step from `t` to `s < t`, with
survival probability `α_t = 1-t`, the discrete cross-entropy weight is
`(α_s - α_t)/(1 - α_t) = (t-s)/t`. The continuous-time (T→∞) negative-ELBO is

```
L∞ = E_q ∫_0^1 [ -α'_t / (1 - α_t) ] · [ -log⟨p_θ(x_t, t), x_0⟩ ] dt,
```

a weighted average of masked-language-modeling losses. With LLaDA's **linear** schedule `α_t = 1-t`
(`α'_t = -1`, `1-α_t = t`) the cross-entropy weight is exactly `1/t`, giving `L(θ)`. The `1/t` rebalances every
corruption level so heavily-masked states do not dominate by term count — it is the negative-ELBO
weight, not a heuristic. Equivalently (Ou et al. 2024), the absorbing-diffusion objective equals the
any-order autoregressive objective `-E_{x_0, π∼U_π}[Σ_i log p_θ(x_0^{π(i)} | x_0^{π(<i)})]` in the
fully-noised limit, explaining the bidirectional/any-order conditioning.

Distinctions: **BERT** uses a fixed mask ratio (~15%) — one corruption level, no generative process,
no likelihood bound. **MaskGIT** uses `-E[Σ_{masked} log p]` with no `1/t` — a sound reconstruction
loss but not a derived likelihood bound. The `1/t` is the bridge from "fill in masks" to
"maximum-likelihood generative model."

## Training, SFT, likelihood

- **Pre-training:** sample `x_0`, `t ~ U(0,1]`, mask each token i.i.d. with prob `t`; one
  bidirectional forward; loss `-1/(t·L) Σ_i 1[x_t^i=M] log p_θ(x_0^i|x_t)`; backprop.
- **SFT:** identical, but leave the prompt `p_0` unmasked and mask only the response `r_0`:
  `-1/(t·L') Σ_i 1[r_t^i=M] log p_θ(r_0^i | p_0, r_t)`.
- **Likelihood / scoring (lower variance):** mask exactly `l ~ U{1,…,L}` tokens (deterministic mask
  count `l/L`); weight `L/l`:
  `-E_{l, r_0, r_l}[ (L/l) Σ_i 1[r_l^i=M] log p_θ(r_0^i | p_0, r_l) ]`. Same expectation as the `1/t`
  form; stabilizes with ~128 Monte-Carlo draws vs ~1000 for the independent-mask form.

## Sampling (the reverse process)

Discretize `[0,1]` into `N` steps; start from a fully-masked response. At each step from `t` to
`s = t - 1/N`: forward `(p_0, r_t)`, predict all masked positions, then re-mask a fraction `s/t` of
them to land on `r_s`. Two remasking strategies:

- **Random remasking:** faithful to the exact reverse process; re-mask a uniformly random `s/t`
  fraction. Reference, but weaker.
- **Low-confidence remasking** (default): keep the highest-confidence predictions, re-mask the
  lowest-confidence ones (`n_un = ⌊L(1-s)⌋` kept at time `s`), confidence
  `c^i = p_θ(r_0^i | p_0, r_t)_{r_0^i}`, committed tokens get `c=1`. An annealed coarse-to-fine
  schedule (easy tokens lock in early, hard tokens stay revisable), analogous to ARM annealed
  sampling — trades diversity for accuracy and substantially beats random remasking.

```
# Low-confidence remasking strategy
r_1 = fully masked sequence of length L
for t = 1 down to 1/N step 1/N:
    s = t - 1/N
    for i in 1..L:
        if r_t^i != M:  r_0^i = r_t^i;  c^i = 1
        else:           r_0^i = argmax_v p_θ(v | p_0, r_t);  c^i = p_θ(r_0^i | p_0, r_t)
    n_un = floor(L * (1 - s))                      # tokens that should be unmasked at time s
    re-mask the (L - n_un) positions with lowest c^i   # keep the most confident
    r_s = r_0
return r_0
```

**Per-step reveal count.** A linear schedule has a constant unmask rate, so reveal the same expected
number of tokens each step: `base = M // N` per step plus the remainder `M mod N` spread over the
early steps. Under the usual `N <= M` setting every step transfers at least one token; with
`N = gen_length`, one token per step.

**Block / semi-autoregressive sampling.** Partition the response into contiguous blocks; run the
diffusion reverse process *within* a block and move left-to-right *across* blocks. Block length =
gen_length is pure diffusion; block length 1 is autoregressive; in between trades global revisability
against locality.

**Uncached control (vanilla).** The construction makes a cache *possible* (the predictor is
time-independent and revealed tokens never change, so at most `L` distinct inputs occur over a
rollout), but the plain control declines it: a full bidirectional forward over the entire sequence
every step, no prefix KV cache (there is no causal prefix; committed tokens are re-attended as new
tokens reveal), no feature cache, recomputing everything. This is the exactly-correct reference.

## Working code (the vanilla, uncached LLaDA generation procedure)

```python
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Gumbel-max categorical sampling; float64 because low-precision Gumbel-max
    degrades generation quality for masked diffusion. temperature 0 -> argmax."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Linear schedule => equal expected reveals per step: floor(M/steps) each,
    remainder spread over the early steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
             temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
             logits_eos_inf=False, confidence_eos_eot_inf=False):
    """Vanilla no-cache reverse generation: a full bidirectional forward over the
    whole sequence at every denoising step. block_length < gen_length => semi-AR."""
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype,
                        device=model.device)], dim=-1)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks                          # steps within each block

    for num_block in range(num_blocks):                  # semi-autoregressive across blocks
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
                 prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):                           # diffusion reverse process within the block
            mask_index = (x == mask_id)
            if cfg_scale > 0.:                            # optional unsupervised classifier-free guidance
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = None
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits  # full uncached forward

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf        # suppress early EOS for some tasks

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf  # current block only

            x0 = torch.where(mask_index, x0, x)          # carry over revealed tokens
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):         # keep top-k confident; re-mask the rest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
```

## Relation to prior methods

- **D3PM (Austin et al. 2021):** discrete diffusion with general `Q_t` and a dense KL ELBO; LLaDA
  specializes to the absorbing/masking case and collapses the KL to masked-token cross-entropy.
- **MDLM / simplified masked diffusion (Sahoo et al. 2024; Shi et al. 2024):** the SUBS / carry-over
  reductions and the continuous-time negative-ELBO cross-entropy weight
  `∫ [-α'_t/(1-α_t)] · [-log⟨p_θ,x⟩] dt`; LLaDA = this at LLM scale with the
  linear-schedule `1/t` weight.
- **RADD (Ou et al. 2024):** time-independent clean-data parameterization (drop the timestep input)
  and the absorbing-diffusion ↔ AO-ARM equivalence.
- **MaskGIT (Chang et al. 2022):** confidence-based iterative parallel decoding (keep most confident,
  re-mask the rest) — the source of low-confidence remasking; LLaDA adds the `1/t` likelihood weight
  the MaskGIT objective lacked.
- **Autoregressive LLMs:** dropped the causal factorization and its prefix KV cache; gained
  bidirectional/any-order conditioning (reversal-curse fix) at the cost of cheap incremental decoding
  — which the uncached control simply forgoes by recomputing every step.
