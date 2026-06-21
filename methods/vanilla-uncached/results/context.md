## Research question

A language model is, at bottom, an attempt to capture an unknown distribution `p_data` over token
sequences by fitting a model `p_θ`, equivalently minimizing `KL(p_data || p_θ)`, equivalently
maximizing `E_{p_data}[log p_θ(x)]`. The overwhelmingly dominant way to do this is the
autoregressive (AR) factorization `p_θ(x) = p_θ(x^1) Π_{i=2}^L p_θ(x^i | x^1, …, x^{i-1})` —
next-token prediction, trained by teacher forcing, sampled left to right. It works spectacularly,
and it has carried scalability, in-context learning, and instruction following all the way up to
the largest models.

The question is whether that one factorization is *necessary* for those capabilities, or whether
they follow from the more basic principle — maximum-likelihood / KL minimization with an expressive
model — and could be obtained from a different factorization that does not commit to a fixed
left-to-right order. One observable consequence of the strict left-to-right order is the reversal
curse: a model trained on "A is B" systematically fails to answer "B is ?" because at training
time the gradient for predicting B from A never also teaches predicting A from B.

## Background

**Diffusion as a generative recipe.** Continuous diffusion models define a forward process that
gradually corrupts data `x` into noise `z_t`, `t ∈ [0,1]`, e.g. `z_t = √(α_t) x + √(1-α_t) ε`
with a monotone schedule `α_t`, and train a reverse model to undo it by maximizing a variational
lower bound on the log-likelihood (the negative-ELBO is a sum of per-step KL terms plus a
reconstruction and a prior term). The appeal for sequences: generation is no longer forced to be
sequential, so it could in principle revise and plan globally.

**Discrete diffusion.** Carrying this to tokens means a Markov forward process directly on the
categorical state, `q(x_t | x_{t-1}) = Cat(x_t; Q_t x_{t-1})`, with cumulative marginals
`q(x_t | x_0) = Cat(x_t; Q̄_t x_0)`, `Q̄_t = Q_t ⋯ Q_1` (Austin et al. 2021, D3PM). One trains by
optimizing the variational upper bound on `-log p_θ(x_0)`,
`L_vb = E_q[ D_KL(q(x_T|x_0) || p(x_T)) + Σ_{t} D_KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) - log p_θ(x_0|x_1) ]`.
The transition `Q_t` can encode different structure. Among the choices, the **absorbing-state**
("masking") process stands out empirically: each token either stays itself or jumps to a special
`[MASK]` token with some probability `β_t`, and once masked it stays masked, so the stationary
distribution puts all its mass on the fully-masked sequence. This is the discrete analogue of a
clean→noise schedule, and it draws an immediate parallel to BERT-style conditional masked language
models. Masking diffusion consistently outperforms uniform and Gaussian-similarity discrete
processes on text perplexity.

**Masked language modeling.** BERT (Devlin 2018) trains a bidirectional transformer to fill in
`[MASK]`ed positions, `L = -E[ Σ_{i ∈ masked} log p(x^i | x_masked) ]`, using a *fixed* mask ratio
(~15%). It is a superb representation learner, with full bidirectional context.

**Any-order autoregressive models (AO-ARM).** Uria et al. 2014, Hoogeboom et al. 2021: model the
joint over *all* orders `π` of the `L` positions with a shared network and `[MASK]` placeholders,
minimizing `-E_{x_0, π∼U_π}[ Σ_{i} log p_θ(x_0^{π(i)} | x_0^{π(<i)}; π) ]`. Filling a position
given a subset of revealed positions is literally predicting a masked token from an unmasked
context, so AO-ARM and masked corruption are describing the same object from two directions.

**Measured properties of masking-based systems.** Several phenomena about these systems are
measured and taken as given: masking/absorbing discrete diffusion beats uniform and
Gaussian-structured discrete diffusion on text; when the denoising network leaves already-revealed
tokens untouched and never emits `[MASK]`, the per-step objective simplifies and measured
likelihood improves; once a token is revealed in an absorbing reverse process it never changes
again, so over a full rollout there are at most `L` genuinely distinct denoiser inputs; and in
iterative parallel decoding of masked predictors, keeping the highest-confidence predictions and
re-masking the rest produces markedly better samples than keeping a random subset.

## Baselines

**D3PM, absorbing variant (Austin et al. 2021).** The first discrete-diffusion framework for text:
general `Q_t`, the variational bound above, with `[MASK]` as the absorbing state the best-performing
instance. Core math is the KL-per-step ELBO. Because it is built to support *arbitrary* `Q_t`, the
objective is evaluated by materializing the full transition matrices `Q̄_t` and comparing full true
and approximate posterior distributions at every step; the loss is a sum of dense KL terms rather
than something that reduces to a plain token-level cross-entropy.

**Continuous-time / Rao-Blackwellized masked diffusion (Sahoo et al. 2024; Shi et al. 2024).**
Specialize to masking and exploit two structural properties of the denoiser — never predict
`[MASK]`, and copy already-revealed tokens unchanged. Under these, the dense D3PM KL collapses; the
discrete-time diffusion loss becomes a masked-token cross-entropy with weight
`(α_s-α_t)/(1-α_t)` for a reverse step from `t` to `s < t`. Letting the number of
steps `T -> ∞` gives a continuous-time negative-ELBO with cross-entropy weight
`(-α'_t)/(1-α_t)`: `L∞ = E_q ∫_0^1 [(-α'_t)/(1-α_t)] · [-log⟨x_θ(z_t,t), x⟩] dt`, a
*weighted average of masked-token cross-entropies*. These works are demonstrated at the
perplexity-benchmark scale (LM1B / OpenWebText, hundreds of millions of parameters), approaching
but not matching AR perplexity.

**Reparameterized absorbing diffusion (Ou et al. 2024).** Shows the concrete score of absorbing
diffusion factors as a *time-independent* clean-data conditional times an analytic time-dependent
scalar, `[e^{-σ̄(t)}/(1-e^{-σ̄(t)})] · p_0(x̂^i | x_t^UM)`, so the network can drop its time input
and simply output a distribution over clean tokens (a softmax head, GPT-like). It further proves
the absorbing-diffusion objective equals the AO-ARM objective in the fully-noised limit.
Delivered as a likelihood/perplexity and sampling-efficiency result at small scale.

**MaskGIT (Chang et al. 2022).** A bidirectional transformer for *images* trained with masked
visual token modeling, `L = -E[ Σ_{i ∈ masked} log p(y_i | Y_M) ]`, and decoded by iterative
parallel sampling: at each of `T` steps predict all masked tokens, keep the most confident ones,
re-mask the rest, with the number kept set by a mask-schedule function `γ(r)` (e.g. cosine) and
temperature annealing on the confidences. This is the source of confidence-based parallel
decoding. The training objective is a flat sum of masked cross-entropies with no weighting across
corruption levels; the method is built and evaluated for image tokens.

**Standard autoregressive LLMs (LLaMA-family, etc.).** The yardstick: next-token transformers
with causal masking, optionally grouped-query attention and a prefix KV cache for fast decoding.
The causal factorization is strictly left-to-right, and the KV cache that makes them fast is a
direct consequence of that one-directional, write-once structure.

## Evaluation settings

The natural yardsticks already in use for an LLM-scale generative model, all pre-existing:

- **General language understanding:** MMLU (5-shot), BBH (3-shot), ARC-Challenge, HellaSwag,
  WinoGrande, PIQA, TruthfulQA — multiple-choice / short-answer, scored by accuracy or by
  conditional likelihood of each candidate.
- **Math:** GSM8K (few-shot), MATH — exact-match on the final answer of a generated solution.
- **Code:** HumanEval (pass@1, execution), MBPP, HumanEval-FIM — functional correctness of
  generated programs.
- **Chinese:** CMLU, C-Eval.
- **Likelihood / perplexity** on held-out text (LM1B, OpenWebText), the classic measure for a
  generative LM, requiring the model to *score* a sequence, not just sample it.
- **Reversal generalization:** paired forward/backward completion tasks (e.g. reversal poem
  completion) to probe the directional-bias failure directly.
- **Protocol:** for sampled tasks, fix a generation length and a number of decoding steps, decode
  deterministically, and score the decoded text with the benchmark-native metric; for
  likelihood-scored tasks, evaluate the conditional likelihood of each candidate answer under the
  model. Standard pre-training (Warmup-Stable-Decay LR schedule, AdamW) and SFT-on-pairs pipelines.

## Code framework

The substrate is the ordinary transformer-LM training-and-decoding setup that already exists for
sequence models, with the pieces that are specific to the generative factorization left as empty
slots. What is genuinely known beforehand: token embeddings, transformer blocks, a vocabulary
projection, a cross-entropy primitive, an optimizer and training loop, and — for decoding — a loop
that repeatedly calls the network and edits a working sequence. The attention pattern, the
corruption variables, the likelihood-tied loss, and the rule for turning network outputs into a
finished sample are exactly what has to be designed.

```python
import torch
import torch.nn.functional as F


class SequenceModel(torch.nn.Module):
    """A transformer over token ids whose attention policy is supplied by the
    generative process being designed."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        )
        self.head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_policy):     # x: (B, L) token ids
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, attention_policy)        # causal, bidirectional, or something else
        return self.head(h)                     # (B, L, vocab)  per-position logits


def corrupt(x0):
    """Map a clean sequence to latent/training inputs for the proposed factorization."""
    # TODO: define the latent variables or conditioning pattern, including any
    #       schedule over corruption or reveal levels.
    raise NotImplementedError


def loss(model, x0):
    """A training loss for the generative process — must be tied to the
    likelihood of the data under the process the corruption/decoding implement."""
    x_corrupt, info = corrupt(x0)
    logits = model(x_corrupt, attention_policy=info.attention_policy)
    # TODO: compute an objective whose expectation is a valid likelihood term
    #       or variational upper bound for the model distribution.
    raise NotImplementedError


@torch.no_grad()
def generate(model, prompt, gen_length, steps):
    """Turn the trained predictor into a sample: start from a blank/unspecified
    generation region appended to the prompt and iterate the network."""
    state = init_generation_state(prompt, gen_length)
    for _ in range(steps):
        logits = model(state.tokens, attention_policy=state.attention_policy)
        # TODO: read predictions from `logits` and update `state` toward a finished
        #       sequence according to the chosen generative process.
        state = update(state, logits)
    return state.tokens
```

The three `# TODO`s — the latent/corruption process, the likelihood-tied loss, and the decoding rule
— are the slots the method fills. The loop is intentionally a plain repeated call to the network;
any reuse of intermediate computation is outside the generic scaffold.
