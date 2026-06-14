# Context: a sequence-mixing layer that both gates and deltas (circa 2024)

## Research question

Softmax attention costs `O(L^2 d)` and grows an unbounded KV cache, so a family of subquadratic
alternatives compresses the past into a *fixed-size* matrix-valued state `S` and runs as a linear RNN.
Two largely separate threads have each fixed one failure of the simplest such layer (plain additive
linear attention, `S_t = S_{t-1} + k_t^T v_t`, which never forgets):

1. **Gating / data-dependent decay.** Multiply the state by an input-dependent factor before the
   additive write, `S_t = S_{t-1} ⊙ M_t + k_t^T v_t`. With a scalar or diagonal `M_t` this gives a
   content-chosen *forgetting rate* and recovers a parallel scan / chunkwise matmul form. It controls
   *how fast memory fades* but the decay is elementwise — it cannot remove the specific stored
   association that a new key collides with.

2. **The delta rule.** Replace the additive Hebbian write with one LMS/Widrow-Hoff gradient step on the
   squared retrieval error, `S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`. The write is
   error-correcting and *content-addressed* — at `beta_t = 1` (with L2-normalized keys) the transition
   is an orthogonal projection that erases exactly the overwritten direction. It controls *what is
   written/removed* precisely, but as usually stated it has *no decay term* at all: an association,
   once written and not directly overwritten, persists indefinitely.

The precise question: these two mechanisms appear complementary — gating gives rapid *global* erasure,
the delta rule gives targeted *local* updates — but no single layer had both. Can a recurrence combine
a data-dependent decay with the delta-rule write, (a) retaining a strict generalization of each (so it
is never worse than the better parent), (b) staying stable, and (c) keeping the hardware-efficient,
matmul-rich, sequence-parallel chunkwise training form that made each parent trainable at scale?

## Background

**Linear attention as a matrix-valued linear RNN (Katharopoulos et al. 2020).** Replacing
`exp(k_i^T q_t)` with `phi(k_i)^T phi(q_t)` lets the causal sum re-associate into a running state
`S_t = S_{t-1} + k_t^T v_t`, read by `o_t = q_t S_t` — constant-memory inference, but the additive
write has bounded capacity and cross-talk corrupts retrieval once `L` exceeds the key dimension.

**Two computational forms and the chunkwise interpolation.** The identical output is a parallel
`O = (Q K^T ⊙ M) V` (matmul-rich, `O(L^2 d)`) or a recurrent `S_t = S_{t-1} + k_t^T v_t` (`O(Ld^2)`,
strictly sequential). The chunkwise form (Hua 2022; Sun et al. 2023; Yang et al. 2023) splits the
sequence into `L/C` chunks, carries a chunk state, does intra-chunk work as matmuls and inter-chunk
propagation by recurrence — subquadratic *and* tensor-core-friendly. FlashLinearAttention-style I/O-aware
kernels make it fast in practice.

**Gated linear attention / Mamba2 (Yang et al. 2023; Dao & Gu 2024).** Data-dependent decay,
`S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t` (GLA, per-key-channel) or a scalar-per-head decay (Mamba2 /
scalar-gated linear attention). Mamba2 in particular parameterizes the scalar decay through a
discretization step size: a positive `Delta_t = softplus(W_dt x_t + dt_bias)` and a per-head rate
`A = exp(A_log)` give a log-decay `g_t = -A * Delta_t <= 0`, so `alpha_t = exp(g_t) ∈ (0,1]` sits near 1
at initialization (a long-memory prior). Competitive with transformers on plain LM but trailing on
recall-intensive tasks, because elementwise decay forgets globally/per-channel, not per-association.

**DeltaNet (Schlag et al. 2021; Yang et al. 2024).** The error-correcting delta write, made
hardware-efficient by Yang et al. (2024): the state stays additive in pseudo-values
`u_t = beta_t(v_t - sum_{i<t} u_i (k_i^T k_t))`, and the product of Householder transitions has a WY
representation, so a chunk's writes are obtained in closed form by a triangular (UT-transform) inverse —
turning the otherwise-sequential delta recurrence into dense matmuls with `O(L/C)` sequential steps.
Stability: L2-normalize keys so the only non-unit eigenvalue of `I - beta_t k_t k_t^T` is `1 - beta_t`.
Strong on associative recall, but with no global decay.

**Negative eigenvalues / state tracking (Grazzi et al. 2024).** Allowing `beta_t` up to 2 (so the
contractive factor can be negative) extends the expressivity of these linear RNNs to certain
state-tracking problems; an optional setting in the layers above.

## Baselines

A combined layer would be measured against and react to:

- **Scalar/diagonal gated linear attention (GLA, Mamba2).** Data-dependent decay, chunkwise/scan
  training, competitive LM. **Limitation:** elementwise decay cannot localize removal to the colliding
  association; trails on recall.
- **DeltaNet.** Error-correcting, content-addressed write with the UT-transform chunk algorithm; strong
  recall. **Limitation:** no decay term, so it cannot do rapid global forgetting; weaker where a
  binding must fade as content streams past.
- **Mamba2.** Scalar-gated SSM with the structured-state-space-duality chunk form. **Limitation:** the
  selective decay is the only memory-control knob; no content-addressed delta correction.

## Evaluation settings

The usual yardsticks for subquadratic LM layers: WikiText / FineWeb perplexity and zero-shot
common-sense accuracy (LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-easy/-challenge) via the LM-eval
harness; associative-recall synthetics (MQAR / Zoology, MAD) and recall-intensive real tasks (FDA,
SWDE, SQuAD) that probe the capacity/recall mode; in-context retrieval, length extrapolation, and
long-context understanding; GPT/LLaMA-style decoders at a few hundred M to a few B parameters with
AdamW, cosine schedule, head dimension 128, short-conv kernel 4; and training throughput / per-kernel
wall-clock for the recurrent vs. chunkwise forms.

## Code framework

The layer plugs into a standard decoder-only transformer: token embedding, a stack of pre-norm blocks
each with a token-mixing sublayer and a feed-forward sublayer, a final norm and the tied LM head.
Everything except the token-mixing sublayer exists. The open slot is a causal mixer with a fixed-size
state, a hardware-efficient chunkwise training path, and — the design target here — both a
data-dependent decay and a delta-rule write.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMixer(nn.Module):
    """The sequence-mixing layer to design: maps x in R^{B x L x d} to o in R^{B x L x d} causally,
    with a fixed-size per-step state (constant-memory inference, subquadratic training). The state
    update and a matmul-rich, sequence-parallel chunkwise training algorithm for it are what is to be
    designed."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # TODO: gate / decay parameters, writing-strength projection, short conv, output norm.

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # TODO: the causal recurrence (decay + delta-rule write) and its chunkwise training path.
        o = None
        return self.o_proj(o)


class Block(nn.Module):
    """Pre-norm transformer block: token mixing then channel mixing, each residual."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mixer = TokenMixer(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)  # feed-forward, already defined

    def forward(self, x):
        x = x + self.mixer(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

A completed mixer supplies the `forward` body, the decay and writing-strength parameters, any short
convolution / output normalization, and the corresponding state-update path for training and inference.
