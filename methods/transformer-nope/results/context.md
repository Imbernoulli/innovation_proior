## Research question

A causal (decoder-only) Transformer can be trained to solve algorithmic sequence tasks —
copy a string after a delimiter, repeat it twice, reverse it — and at the *training* lengths
it fits them to near-perfect accuracy. The hard part is **length generalization**: train only
on content of length up to `L` (here `L = 20`, sampled uniformly from `[1, L]`) and ask the
model, at inference, to handle unseen longer inputs up to `2L`. This matters far beyond toy
tasks: it is infeasible to train on every context length a model will eventually see, the
number of available examples drops sharply as length grows, and longer contexts buy more
in-context examples, more reasoning steps, and longer generations. So the model must
extrapolate from finite, short training examples.

The architectural knob most implicated in this behavior is how the model represents **token
order**. Self-attention is a set operation; absent some position signal it cannot tell `a b`
from `b a`. The precise question is: *what way of injecting (or not injecting) position
information lets a small causal Transformer extrapolate to lengths it never saw during
training?*

## Background

Transformers, unlike RNNs, process a sequence in parallel and so have no built-in notion of
order. The standard fix is a **positional encoding** (PE). Two broad families exist.
*Absolute* schemes assign a vector to each absolute index `1, 2, 3, …`; *relative* schemes
make the attention between two tokens depend on their distance `t − i`.

The load-bearing prior facts:

- **Sinusoidal absolute PE (Vaswani et al. 2017).** Each position `j` gets a fixed,
  non-parametric vector
  `p_j = [sin(ω_1 j), cos(ω_1 j), …, sin(ω_{d/2} j), cos(ω_{d/2} j)]` with
  `ω_i = 1/10000^{2i/d}`, added to the word embedding before layer 1. Because it is a closed
  form it is defined at any index.
- **Learned absolute PE (used in GPT-3, OPT).** A trainable embedding per position.
- **T5 relative bias (Raffel et al. 2020).** Add a learned scalar `b_{bucket(t−i)}` to the
  query–key score `q_t^T k_i`. Distances are mapped through a logarithmic bucketing function
  that lumps all large distances into one bucket (in T5, 32 buckets, max distance 128), so
  unseen distances reuse a learned parameter.
- **Rotary PE (Su et al. 2021), used in PaLM and LLaMA.** Rotate `q_t` and `k_i` by an angle
  proportional to their absolute positions; because `(R^{tθ})^T R^{iθ} = R^{(i−t)θ}`, the
  dot product depends only on the relative offset `i − t`. It is usually classed as relative.
- **ALiBi (Press et al. 2022), used in BLOOM.** Subtract a distance-proportional penalty from
  the score: `q_t^T k_i - (t-i) m_h`; for a power-of-two head count `n`,
  `m_h = start^(h+1)` with `start = 2^(-2^(-(log2(n)-3)))`, so for 8 heads the slopes are
  `1/2, 1/2^2, ..., 1/2^8`. The linear penalty creates a recency bias. It was introduced to
  make language-model perplexity extrapolate to longer inputs.

Two diagnostic facts about *existing* systems are knowable before any new method:

- **Self-attention's order-sensitivity depends on masking (Tsai et al. 2019).** Viewing
  attention through a kernel lens, an *encoder* (bidirectional) self-attention layer with no
  PE is order-invariant — it collapses to a bag-of-words. But a *decoder* with a causal mask
  is *not* order-agnostic: each query at position `t` attends only to the `t` positions
  `≤ t`, so masking itself carries order information. Shen et al. (2017) observed this early.
- **A causal LM with no PE is competitive in-distribution, and seems to build position
  implicitly (Haviv et al. 2022).** Training GPT-style LMs with *no* positional encoding at
  all yields perplexity competitive with sinusoidal, learned, and ALiBi models across
  datasets, sizes, and lengths. Probing those models recovers an implicit notion of *absolute*
  position spread through the network; Haviv et al. conjecture the causal mask lets each token
  infer *how many predecessors* it has, thereby approximating its absolute index.
- **In-distribution perplexity does not distinguish PEs and does not transfer.** At training
  length, the different PEs (and the no-PE model) reach very similar perplexity / near-perfect
  accuracy, so an IID metric cannot separate them (Haviv et al. 2022; Scao et al. 2022); and
  perplexity is known not to track downstream performance (Tay et al. 2022). The signal lives
  in the out-of-distribution, longer-length regime on structured tasks.

The construction tools available for proving what a masked attention model *can* represent come
from the program-synthesis-into-attention line: RASP (Weiss et al. 2021) and Tracr (Lindner et
al. 2023) show how to hand-build attention weights that compute specific quantities, which makes
it possible to exhibit an explicit weight setting rather than argue abstractly.

## Baselines

- **Recurrent attention seq2seq (Bahdanau, Cho & Bengio 2015).** An encoder–decoder LSTM
  with a learned attention over encoder states. Order is handled by recurrence, so there is
  no PE question.

- **Decoder-only Transformer with sinusoidal absolute PE (Vaswani et al. 2017).** Add the
  fixed `p_j` vectors to the embeddings; attention is plain `q_t^T k_i`. Closed-form so it is
  defined at any length.

- **Decoder-only Transformer with ALiBi (Press et al. 2022).** Score `q_t^T k_i − (t−i)·m_h`
  with fixed geometric per-head slopes. The linear recency penalty extends to any distance.

- **Decoder-only Transformer with T5 relative bias (Raffel et al. 2020).** A bucketed
  learned bias is added to every attention score, tying together large distances via the
  bucketing scheme.

## Evaluation settings

- **Tasks.** Algorithmic sequence-to-sequence problems with full control over the train/test
  length distribution. Here three variants of a copy-family task: `delim` (output equals the
  input after a delimiter), `repeat` (output is the input repeated `repeat_k = 2` times), and
  `reverse` (output is the reversed input). The broader literature this sits in also uses
  Addition, Polynomial Evaluation, Sorting, Summation, Parity, LEGO, SCAN, PCFG.
- **Sequence layout.** `[BOS] x_1 … x_T [SEP] y_1 … y_M [EOS]`, with the autoregressive
  language-modeling loss computed only on positions whose next token is part of the target.
- **Length splits.** Train content lengths sampled from `[1, L_train]` with `L_train = 20`;
  evaluate greedily on two splits per variant — `id` with lengths in `[1, L_train]` and `ood`
  with lengths in `[L_train + 1, 2 L_train]`.
- **Vocabulary / model.** 16 symbols plus PAD/BOS/EOS/SEP (vocab size 20); 4 layers,
  `d_model = 128`, 4 heads, GELU MLP with 4× expansion.
- **Optimization.** AdamW, lr `5e-4`, weight decay `1e-2`, batch 256, 6,000 steps, 200-step
  linear warm-up, gradient clip 1.0.
- **Metrics.** Per variant: `exact_match_id`, `token_acc_id`, `exact_match_ood`,
  `token_acc_ood`, and `score = 0.5·exact_match_id + 0.5·exact_match_ood`; the task summary is
  the geometric mean across the three variants, which punishes any variant that fails to
  extrapolate.

## Code framework

The harness already exists. A `PositionalScheme` is a container of three optional hooks the
model consults; the model is a standard causal Transformer (or an LSTM seq2seq via a flag).
The two slots to fill are how position information is supplied and how the model is assembled.

```python
import torch.nn as nn


class PositionalScheme:
    """Container of optional hooks the attention/embedding code consults.

    token_embedding_extra(positions) -> additive [.., d] token-level embedding, or None
    attn_bias(T, device, dtype)      -> additive [n_heads, T, T] (or [1, T, T]) score bias, or None
    rotary(q, k)                     -> (q, k) rotated before the dot product, or None
    extra_modules                    -> nn.ModuleList holding any learnable params so AdamW sees them
    """
    def __init__(self, name, token_embedding_extra=None, attn_bias=None,
                 rotary=None, extra_modules=None):
        self.name = name
        self.token_embedding_extra = token_embedding_extra
        self.attn_bias = attn_bias
        self.rotary = rotary
        self.extra_modules = extra_modules or nn.ModuleList()


class SeqModel(nn.Module):
    """Embedding -> causal backbone (Transformer, or LSTM if use_lstm) -> vocab projection.

    The Transformer backbone consults scheme.token_embedding_extra after embedding,
    scheme.rotary inside each attention head, and scheme.attn_bias on the scores.
    forward(tokens) -> [B, T, vocab_size].
    """
    def __init__(self, config, scheme, use_lstm=False):
        super().__init__()
        # standard token embedding, causal backbone, output projection over the vocab
        ...


def build_positional_scheme(config) -> PositionalScheme:
    # TODO: decide what positional information, if any, to supply, and how
    pass


def build_model(config) -> nn.Module:
    # TODO: assemble the model around the chosen scheme
    pass
```
