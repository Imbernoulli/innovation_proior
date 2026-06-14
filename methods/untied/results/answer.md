# Untied input/output embeddings, distilled

**Untied embeddings** keep a language model's input token-embedding matrix and its output
projection (softmax classifier) matrix as two *separate* learned matrices, rather than
sharing one matrix for both ends (weight tying). It is the unconstrained configuration of
the two `C × H` matrices a language model uses: `U` to look a token up on the way in, `V`
(here `lm_head`) to score every token on the way out.

## Problem it solves

A neural language model represents its vocabulary at two ends — an input embedding lookup
and an output softmax classifier — both naturally `C × H` (vocabulary × hidden). Tying
(`U = V`) saves the model's largest parameter block and regularizes, but forces one matrix
to satisfy two different jobs. Untying spends those parameters to give the output
classifier its own degrees of freedom, so its classifier rows are not forced to be the
same vectors used by the input lookup.

## Key idea

The two ends are driven by structurally different gradients and want different similarity
structures, so let them be different matrices.

- **Different roles.** The input matrix wants synonyms to perturb the body's state
  alike (react-alike); the output matrix is a softmax classifier weight, so it wants
  interchangeable continuations to score alike (predict-alike). Related, not identical.
- **Different gradient streams.** With separate matrices and per-step loss
  `L_t = −log p_t(o_t|·)`, where
  `p_t(k|·) = exp(V_kᵀh) / Σ_x exp(V_xᵀh)`:
  - Output: every row updated every step,
    `∂L_t/∂V_{o_t} = (p_t(o_t|·) − 1)h`, and `∂L_t/∂V_k = p_t(k|·)h` for `k ≠ o_t`
    (dense).
  - Input: only the current-input row updated,
    `∂L_t/∂U_{i_t} = (Σ_x p_t(x|·)V_xᵀ − V_{o_t}ᵀ)·∂h/∂U_{i_t}`, and `∂L_t/∂U_k = 0`
    for `k ≠ i_t` (sparse).
- **Why tying makes the shared matrix resemble the output role.** With `S = U = V`, every
  row `k ≠ i_t` is updated exactly like the untied output row. Row `i_t` gets both an
  input-role term and an output-role term; the output-role term on that row is usually the
  non-target case `p_t(i_t|·)h`, small because immediate repetition is rare, so the
  current-input row is locally dominated by the input term. Across the whole matrix,
  however, almost every row-time update is output-only, while input-role updates are sparse.
  That is why the tied matrix evolves more like the output embedding and leaves the input
  role underrepresented.
- **Why the word2vec objection is identity-body specific.** In word2vec skip-gram the body
  is the identity (`h ≈ U_{i_t}`). If the center and context matrices are forced to be the
  same, a word's self-score is `U_iᵀU_i = ||U_i||²`; because words are rarely their own
  contexts, making self-prediction unlikely tries to make that squared norm small
  (Goldberg & Levy 2014). A deep body (LSTM / transformer) makes `h` a nonlinear function
  of the whole history, *decoupling* the two ends, so that specific norm-collapse argument
  does not transmit — tying becomes a regularizing *choice*, not a law.
- **The trade.** Tying = fewer parameters + a regularizing constraint. Untying = extra
  output-head capacity: the output classifier can learn predictive rows that are not also
  the input lookup rows. The choice is whether that output-specific capacity is worth the
  extra `C·H` parameters and regularization burden.

## Initialization

- Output projection `_lm_head_weight`: **zero-init**. Initial logits are `0` → uniform
  softmax → initial loss `= ln(C)` (maximum entropy), no spurious logits to undo; rows
  grow from zero under the dense `(p − 1{k=o_t})·h` gradient.
- Input embedding `wte`: standard `normal(0, 0.02)`, so the slow-learning (sparse-gradient)
  input pathway carries signal from step 1.
- Position embedding `wpe`, body, and the bias-free output `Linear` are unchanged; untying
  touches only the input↔output matrix relationship. Position parameters are excluded from
  the reported parameter count.

## Working code

Fills `get_lm_head_weight()` (the one empty slot in the embedding module). The harness
wires `lm_head.weight = embedding.get_lm_head_weight()`; returning a separate parameter
instead of `wte.weight` is the entire change.

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding with a separate output projection.

    The matrix used to score tokens at the output (lm_head) is a separate learned
    parameter, decoupled from the input token-embedding lookup (wte). This lets the
    input (contextual / react-alike) and output (predictive / score-alike)
    representations specialize, at the cost of one extra (vocab_size x n_embd) block.
    """

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # input token lookup
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned positions
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        # Separate output-projection weight (NOT tied to wte).
        self._lm_head_weight = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))
        # Zero-init: logits start at 0 -> uniform softmax -> loss starts at ln(vocab_size).
        nn.init.zeros_(self._lm_head_weight)

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)            # (B, T, n_embd)

    def get_lm_head_weight(self):
        return self._lm_head_weight                    # the output matrix, free of wte

    def get_num_pos_params(self):
        return self.wpe.weight.numel()                 # positions excluded from param count

    def get_value_embed(self, layer_idx):
        return None                                    # no per-layer value residual
```

The surrounding model is unchanged: `self.embedding = TokenEmbedding(config)`;
`self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)`;
`self.lm_head.weight = self.embedding.get_lm_head_weight()`; forward embeds, runs the fixed
blocks (adding `get_value_embed(i)` when non-`None`), applies the final layer norm,
projects through `lm_head`, and takes cross-entropy against the targets.
