**Problem.** A GPT-style LM represents its vocabulary at two ends — an input embedding lookup `wte` and
an output softmax classifier `lm_head` — both `(vocab_size, n_embd)`. The scaffold default *ties* them
(`lm_head.weight = wte.weight`), forcing one matrix to satisfy two different jobs. This is the lightest
embedding decision to question first, and the floor of the ladder.

**Key idea.** Untie: keep the input table and the output projection as two *separate* learned matrices.
The two ends want different similarity structures (input: synonyms react-alike; output: interchangeable
continuations score-alike) and are driven by different gradient streams — the output matrix is dense
(every row updated every step, `∂L/∂V_k = (p_k − 1{k=o})h`), the input matrix is sparse (only the
current-input row, `∂L/∂U_{i_t} = (Σ_x p_x V_x − V_o)·∂h/∂U_{i_t}`). Tying makes the shared matrix track
the output role and underrepresent the input role.

**Why it works here.** Tying = fewer parameters + a regularizing constraint, a win in the small-corpus
regime that made it the default. With ~7.1B tokens against a ~355M model, overfitting is not binding;
spending the `vocab_size · n_embd` block (~51.5M params) to give the output classifier its own degrees of
freedom is the favorable trade. The word2vec "sharing is harmful" result is identity-body specific
(self-score `‖U_i‖²` collapse); a 24-layer body decouples the two ends, so tying is admissible, not
mandatory, and untying is the unconstrained, more expressive configuration.

**Hyperparameters / edit.** Add a separate `_lm_head_weight` of shape `(vocab_size, n_embd)`,
**zero-initialized** (logits start at 0 → uniform softmax → initial loss `ln(vocab_size)`, no spurious
logits to undo; rows grow under the dense gradient). Keep `wte` at `normal(0, 0.02)` so the sparse input
pathway carries signal from step 1. Positions, body, and the bias-free head unchanged; no per-layer
residual. The entire change is what `get_lm_head_weight()` returns.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 115-140) — step 1: untied
class TokenEmbedding(nn.Module):
    """Token + position embedding with UNTIED lm_head weight."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        # Separate output projection weight (not tied to wte)
        self._lm_head_weight = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))
        nn.init.zeros_(self._lm_head_weight)

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)

    def get_lm_head_weight(self):
        return self._lm_head_weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

The surrounding `GPT` is unchanged: it wires `self.lm_head.weight = self.embedding.get_lm_head_weight()`
(now the separate matrix), embeds, runs the 24 fixed blocks, applies the final layer norm, projects
through the untied `lm_head`, and takes cross-entropy against the targets.
