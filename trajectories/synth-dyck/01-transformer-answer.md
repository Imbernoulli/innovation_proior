**Problem.** Learn Dyck-(k,m) as a left-to-right language model and length-generalize to strings longer
than training. The valid next token is fixed by the open bracket on top of the stack, which may sit far
back in the sequence — the model needs cheap long-range reach.

**Key idea.** A 4-layer **causal Transformer** with **learned absolute positional embeddings**. Self-
attention gives every position a one-hop reach to every earlier position, the natural primitive for
bracket matching; a causal mask enforces the autoregressive contract; learned absolute position embeddings
break attention's permutation-equivariance so the model can tell positions apart.

**Why this rung first.** It carries the strongest in-distribution prior on the board, so its length-OOD
behavior cleanly isolates *generalization across length* from *fitting*. Absolute positions are expected
to be the bottleneck: only position rows seen in training receive gradient, so at OOD lengths the model
attends over untrained, random position codes — the failure should be largest exactly where the train→test
length gap is largest (`dyck-length-ood`).

**Hyperparameters.** `num_layers=4`, `num_heads=4`, `hidden=config.hidden_dim` (64), `dim_feedforward=4×`,
`dropout=0.0`, `activation="gelu"`, `norm_first=True` (pre-norm, stable in 100 steps without warmup),
`max_len=1024` (generous cap, never indexes out of bounds, stays under the 500k-param budget).

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
