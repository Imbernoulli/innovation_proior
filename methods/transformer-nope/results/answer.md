# NoPE: a decoder-only Transformer with no positional encoding, distilled

NoPE is a causal (decoder-only) Transformer trained with **no explicit positional encoding
of any kind** — no absolute position embedding added to the input, no additive relative bias
on the attention scores, no rotation of queries/keys. The attention score is exactly
`q_t^T k_i` under the causal mask, and the order signal comes entirely from the mask itself.
For length generalization on algorithmic sequence tasks it matches or beats every explicit
positional-encoding scheme while adding zero attention compute.

## Problem it solves

Train a small causal Transformer on content lengths up to `L` and have it extrapolate to
unseen longer lengths (here up to `2L`) on copy-family tasks (copy after a delimiter, repeat
twice, reverse). At training length every positional encoding works; the differences appear
only out of distribution, where each explicit scheme's fixed inductive bias (periodic
absolute code, log-bucketed relative table, rotation schedule, linear recency penalty) caps
how it can use position at lengths it never saw.

## Key idea

A decoder's causal mask already breaks permutation invariance: the query at position `t`
attends to exactly `t` positions, so the visible-window size *is* the position. Removing the
positional encoding therefore does not blind the model to order — it removes a *prescribed*
distance bias and lets gradient descent learn whichever position representation the task
rewards. Two constructions show the architecture is expressive enough:

- **Absolute (layer 1).** Build the embedding so coordinate 1 is `1` for all tokens,
  coordinate 2 is `1` only at `<bos>`. A head whose keys all read coordinate 1 has identical
  logits, so softmax over the `t` masked positions is uniform `= 1/t`; values read coordinate
  2, so only `<bos>` contributes, giving attention output `(1/t) e_1`. Write it to coordinate
  3: now `h_t^(1)` carries `1/t`, and the ReLU/GELU MLP (a universal approximator) maps
  `1/t ↦ t`. The causal mask sets the denominator `t`; `<bos>` anchors the numerator.

- **Relative (layers `l ≥ 2`).** With absolute position in coordinate 3, choose `W_Q`, `W_K`
  so `q_t = [1, −t, …]` (row 1 reads coord 1, row 2 reads coord 3 with `−1`) and
  `k_i = [i, 1, …]` (row 1 reads coord 3, row 2 reads coord 1). Make the remaining
  content rows ignore the first three reserved coordinates. Then

  `⟨q_t, k_i⟩ = 1·i + (−t)·1 + Σ_{j≥3} q_{t,j} k_{i,j} = f_cnt(q_t, k_i) − (t − i)`,

  a score with a pure linear relative term. The sign is non-positive for causal positions
  `t ≥ i`, and farther past keys receive a larger negative offset. This construction proves the
  reachable `f_rel(d) = -d`; richer relative functions require additional position features and
  matching query-key weights, not just one scalar coordinate.

So NoPE contains an absolute code and a relative-score construction as explicit weight settings;
constraining neither with a hand-designed PE, it can deviate where a fixed prescription would
mismatch. Empirically (by Jensen–Shannon
distance between attention distributions) the learned mechanism resembles a relative
encoding with bimodal short- and long-range attention — exactly what copy/reverse need — and
it costs nothing extra in the attention because no bias or rotation is computed.

## Final form

The attention dot product in every layer is

`⟨q_t, k_i⟩ = q_t^T k_i`  (causal mask only),

and the input is `H^(0) = W_E X` with no positional term. Everything else is a conventional
decoder-only Transformer: multi-head causal attention, residual connections, layer norm,
GELU MLP with 4× expansion, autoregressive LM loss on the target positions.

In the task harness, the positional scheme supplies none of its hooks and the model is the
plain causal backbone:

```python
import torch.nn as nn


def build_positional_scheme(config) -> PositionalScheme:
    """No positional encoding: none of the three hooks is supplied.

    The decoder's causal mask carries order on its own. Layer 1 can recover
    absolute position (uniform attention over t identical keys -> 1/t,
    anchored at <bos>); later layers can express a relative score depending
    on (t - i). SGD learns which signal to use; no positional params are added.
    """
    return PositionalScheme(
        name="nope",
        token_embedding_extra=None,  # no absolute position added to embeddings
        attn_bias=None,              # no T5/ALiBi additive score bias
        rotary=None,                 # no rotation of q, k
        extra_modules=nn.ModuleList(),
    )


def build_model(config) -> nn.Module:
    """Decoder-only Transformer; the score is exactly q_t^T k_i under the mask."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```

Equivalently, in a standard HuggingFace-style decoder-only attention block, the positional
branch is empty — the score is computed and only the causal mask is added before softmax:

```python
# inside a causal self-attention head, NoPE branch
scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # q_t^T k_i
if attention_mask is not None:
    scores = scores + attention_mask  # causal mask only; no positional term
attn = torch.softmax(scores.float(), dim=-1).type_as(scores)
out = torch.matmul(attn, value_states)
```
