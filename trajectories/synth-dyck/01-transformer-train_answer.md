Dyck-(k,m) is, at heart, a language about *matching*: the token I must emit at position $t$ is fixed entirely by the single open bracket sitting on top of the stack, which may have been written many positions back. The core computational primitive is therefore the ability to reach back to an arbitrary earlier position and read it — and that is exactly what self-attention does in one hop, every position attending directly to every earlier position with no recurrence in between. Before reaching for anything with explicit memory, I want to measure how far a pure attention stack gets on a language whose entire content is long-range matching, and I want to place first the architecture with the strongest in-distribution prior, so that wherever it breaks names the problem cleanly rather than confounding fitting with generalization.

I propose a **4-layer causal Transformer with learned absolute positional embeddings** as the opening rung. The token embedding `nn.Embedding(vocab, hidden)` maps each of the $2k+2$ symbols to a $64$-dimensional vector; that part is uncontroversial. The decision the whole length-generalization question turns on is *positional information*. Self-attention is permutation-equivariant — $\mathrm{softmax}(QK^\top)V$ is a sum of dot products and weighted averages over the *set* of positions, with no term anywhere that knows which position is which, so if I shuffle the input rows the output rows shuffle identically. For a language where order is everything ( `( )` is valid, `) (` is not ), a model that cannot tell positions apart is a bag of brackets and fails outright. So I inject position with a second `nn.Embedding(max_len, hidden)` indexed by the integer position and added to the token embedding at the bottom of the stack: each position $0,1,2,\dots$ gets its own learned vector that the model may use as it likes.

The sequence is fed as $[\text{BOS}, x_1, \dots, x_n]$ and scored left-to-right, so the model must be **causal** — the prediction at position $t$ may depend only on tokens at positions $\le t$, because at every step it predicts the *next* token from the prefix it has seen. A bidirectional encoder would let position $t$ peek at the token it is supposed to predict, so I enforce causality with a mask. For a sequence of length $T$ the mask is the strictly-upper-triangular matrix of $-\infty$ (every entry where query $i$ looks at key $j>i$) and $0$ elsewhere; added to the pre-softmax scores, the $-\infty$ entries become $0$ after the softmax, so position $i$ attends only to positions $\le i$. I build it with `torch.triu(torch.full((T, T), -inf), diagonal=1)` and pass `is_causal=True` so the kernel takes its fast path.

The attention stack itself is four `nn.TransformerEncoderLayer` blocks — multi-head self-attention followed by a position-wise feed-forward, residual connections and layer norm around each. I set `norm_first=True` (pre-norm) because pre-norm Transformers are markedly more stable at small depth and few steps, and with only 100 gradient steps per environment I cannot afford a warmup schedule to tame post-norm. I use `dropout=0.0`: with 8000 training strings over a tiny vocabulary and a small model trained for 100 steps, I am nowhere near the overfitting regime, and dropout would only add noise to an already short optimization. `activation="gelu"` is the standard smooth nonlinearity and `dim_feedforward = 4 * hidden` the usual $4\times$ expansion. Four heads and four layers — comfortably above $\log m$ for $m\le 5$, which by the representational result for self-attention on bounded-depth Dyck is enough depth to track the nesting. The hidden width is $64$, giving head dimension $16$. I register `torch.arange(max_len)` as a non-persistent buffer and slice it to the actual $T$ at forward time, so the state dict stays lean and I only ever add as many positional codes as there are tokens. The forward pass embeds the tokens, adds the sliced positional embeddings, runs the causal encoder, and projects to logits $[B, T, \text{vocab}]$ with a linear head — exactly the `DyckModel.forward` contract.

I want to be honest about what learned absolute positions do and do not buy, because this is where I expect the architecture to struggle. During training the model only ever sees positions up to the training maximum — length $64$ in two of the three environments — so only those rows of the position table receive a gradient. The rows beyond the training maximum, the ones needed at OOD evaluation (lengths $65$–$96$, $97$–$128$, and the brutal $128$–$256$ of the length-OOD probe), sit at their random initialization. Whatever position-dependent computation the model learned in-distribution — "look back to the matching open bracket using the relationship between my position and its position" — is expressed in terms of position vectors that, in the OOD range, are noise. That is the structural reason the absolute-position recipe is expected to fail to length-generalize on Dyck: the mechanism that locates the matching bracket is anchored to absolute coordinates, and those coordinates run off the end of what was trained. I cap `max_len` at $1024$ so the model never indexes out of bounds, but a large table does not help — the untrained rows are still untrained — and it keeps the parameter count in the mid-200-thousands, well inside the $500{,}000$ budget, so the budget is not the binding constraint; generalization is.

The falsifiable signature is concrete: in-distribution `id_token_acc` should be high (the model can represent the matching when positions are familiar) while `id_string_acc` stays weak (Dyck's per-string accuracy is unforgiving — one wrong position fails the whole string), and `ood_token_acc` should sag furthest exactly where the train$\to$test length gap is largest, on `dyck-length-ood`. A fitting-fine, extrapolating-poorly pattern implicates the position table; if instead both in-distribution and OOD were poor, my read would be wrong and the next rung would be about capacity rather than position.

```python
def build_model(config: TaskConfig) -> DyckModel:
    """4-layer causal Transformer with learned absolute positional embeddings."""

    class CausalTransformer(DyckModel):
        def __init__(self, vocab: int, hidden: int, num_layers: int, num_heads: int, max_len: int):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(max_len, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=4 * hidden,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(hidden, vocab)
            self.register_buffer(
                "positions",
                torch.arange(max_len, dtype=torch.long).unsqueeze(0),
                persistent=False,
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            B, T = input_ids.shape
            pos = self.positions[:, :T].expand(B, T)
            h = self.embed(input_ids) + self.pos(pos)
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1,
            )
            h = self.encoder(h, mask=causal, is_causal=True)
            return self.head(h)

    # Cap max_len generously above any single-env requirement.
    return CausalTransformer(
        vocab=vocab_size(config.k),
        hidden=config.hidden_dim,
        num_layers=4,
        num_heads=4,
        max_len=1024,
    )
```
