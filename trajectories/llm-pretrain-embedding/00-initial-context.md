## Research question

A GPT-style language model meets its vocabulary at exactly two places, and the single thing being
designed on this ladder is what happens at those two places. On the way in, a token id is looked up in
a learned table `wte` of shape `(vocab_size, n_embd)`; on the way out, after the fixed transformer body
has produced a per-position activation, a linear `lm_head` of the same shape `(vocab_size, n_embd)`
scores every token and a softmax gives the next-token distribution. The default also adds a learned
absolute position embedding `wpe` and *ties* the input table to the output projection
(`lm_head.weight = wte.weight`). Everything else — the corpus, the tokenizer, the 24-layer body, the
optimizer, the schedule — is frozen. The one editable object is the **embedding strategy**: the
`TokenEmbedding` module that owns the input lookup, decides what weight the output softmax borrows,
and may inject an optional per-layer residual. The goal is a *modular embedding-level* intervention
that lowers validation cross-entropy on FineWeb relative to the tied learned-token+position default,
without sneaking capacity in through positions (position parameters are excluded from the reported
count).

## Prior art before the first rung (the embedding lineage)

The default the first rung reacts to is itself the resolution of a line of word-representation choices.
These are the ancestors the ladder starts from; the fixed substrate below is what they converged to.

- **Per-word lookup table (Bengio et al. 2003; Mikolov et al. 2010).** Give every vocabulary item its
  own trainable `n_embd`-vector. Exact and differentiable, but the cost is `vocab_size · n_embd`, the
  single largest block in the model, and by Zipf most rows are rare and barely trained. Gap: the table
  scales with the vocabulary and wastes most of its capacity on the long tail.
- **Distributional / co-occurrence vectors (word2vec, Mikolov et al. 2013; GloVe, Pennington et al.
  2014).** Learn one vector per word from context statistics. They capture sub-token co-occurrence
  cheaply, but a plain token table gives the *same* vector for a token regardless of what preceded it —
  it carries no local order. Gap: single-token embeddings discard the (previous, current) context that
  cheap `n`-gram features would supply.
- **Tied input/output embedding (Press & Wolf 2016/2017, arXiv:1608.05859; Inan et al. 2017).** Reuse
  one matrix at both ends: the row that embeds a word on input is the row that scores it on output.
  This removes a `vocab_size · n_embd` block and regularizes — a real win when parameters and
  overfitting bind. Gap: it forces one matrix to satisfy two different jobs (react-alike on input,
  score-alike on output), which can cost output-classifier capacity when data is abundant.
- **Absolute learned positions (the GPT-2 `wpe`).** Add a learned per-position vector to the token
  embedding so the order-blind body can tell positions apart. It is fixed substrate here, and its
  parameters are excluded from the reported count, so capacity cannot be bought through it.

## The fixed substrate

The model is GPT-2 Medium: 24 layers, 16 heads, `n_embd = 1024`, `block_size = 1024`,
`vocab_size = 50304`, no bias, no dropout (~355M reported parameters). The corpus is FineWeb 10B
(`HuggingFaceFW/fineweb` `sample-10BT`) with the GPT-2 tokenizer, ~7.1B training tokens. Training is
12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP, fused AdamW
(`weight_decay = 0.1`, `betas = (0.9, 0.95)`), a cosine schedule with linear warmup and `grad_clip = 1`.
The transformer block (pre-norm attention + MLP, flash attention), the loss (cross-entropy with weight
tying wired from the embedding module), the data loader, and the parameter-budget check are all frozen
and must not be touched. The body exposes exactly one optional hook: before running block `i`, the model
adds whatever `self.embedding.get_value_embed(i)` returns to the residual stream `x` (`x = x + ve`), or
nothing if it returns `None`. So any per-layer embedding injection in this task lands on the **residual
stream before the block**, not inside attention — that is the only place the harness exposes.

## The editable interface

Exactly one region is editable: the `TokenEmbedding` class in `nanoGPT/custom_pretrain.py`
(scaffold lines 115–140). Every method on the ladder is a fill of this same contract:

- `forward(idx) -> x` — token indices `(B, T)` to embeddings `(B, T, n_embd)`.
- `get_lm_head_weight() -> Tensor` — the `(vocab_size, n_embd)` weight the output softmax borrows
  (tied by default: returns `wte.weight`).
- `get_num_pos_params() -> int` — the count of position parameters, *excluded* from the reported
  parameter count.
- `get_value_embed(layer_idx) -> Optional[Tensor]` — an optional per-layer residual `(B, T, n_embd)`
  added to the stream before layer `layer_idx`, or `None`.

The starting point is the scaffold default below: a learned token table plus learned absolute
positions, tied to the output head, with no per-layer injection. Each later method replaces exactly
this class and nothing else. The benchmark's parameter accounting subtracts `get_num_pos_params()`, so
a method cannot win by scaling positional capacity.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 115-140) — default fill
class TokenEmbedding(nn.Module):
    """Token + position embedding with weight tying to lm_head."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # input token lookup
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned absolute positions
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)                          # (B, T, n_embd)

    def get_lm_head_weight(self):
        """Return weight for the language model head (tied by default)."""
        return self.wte.weight

    def get_num_pos_params(self):
        """Return number of position embedding parameters (excluded from param count)."""
        return self.wpe.weight.numel()
```

The surrounding `GPT` wires the head from this module
(`self.lm_head.weight = self.embedding.get_lm_head_weight()`), and its forward, before each block,
does `ve = getattr(self.embedding, 'get_value_embed', lambda _: None)(i); if ve is not None: x = x + ve`.
The default class defines no `get_value_embed`, so the hook is inert and the model is the standard tied
GPT-2 Medium.

## Evaluation settings

One seed (42). Primary metric: **validation loss** — cross-entropy on FineWeb (lower is better).
Secondary perplexities (lower is better): **WikiText-2** and **LAMBADA**, computed on non-overlapping
`block_size` chunks. Downstream zero-shot accuracy (higher is better): **ARC-Easy** and **HellaSwag**
from the lm-evaluation-harness on the saved checkpoint. (PIQA and WinoGrande are also evaluated but
held out.) All comparisons hold the corpus, tokenizer, body, optimizer, schedule, and token budget
fixed, so the only difference between rungs is the `TokenEmbedding` fill.
