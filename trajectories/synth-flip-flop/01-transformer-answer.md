**Problem.** Track a single-bit memory across a stream of write/ignore/read tokens and predict each read
bit correctly. The architecture is the only free variable; data, optimizer, training loop, and read-only
grading are fixed. The headline challenge is the sparse OOD tail (`FFL(p_i=0.98)`), where the relevant
write is often >100 positions before the read.

**Key idea (the starting rung).** A vanilla causal Transformer — the model whose self-attention gives it
the *right-looking* inductive bias for this task: every earlier position is one hop away, so a read can,
in principle, attend back to the most recent write and copy its bit (an induction-head circuit). No
recurrence, so no vanishing-gradient pathology; the read-to-write gradient is direct.

**Why this and why it should crack.** Self-attention is a *soft, stateless* selection: the read bit is
re-derived from the raw token history at every read by a normalised attention distribution, not held in an
explicit register. On dense/short-range reads this is easy and accurate. On the sparse tail the read head
must out-vote a sea of distant distractor writes through soft attention, and the rare smear that flips the
predicted bit *will* occur over many reads — the persistent **attention glitch** that scale and data do
not remove. Learned absolute positions, trained at `T=512`, also extrapolate poorly to the `T=1024`
long-context split.

**Architecture / hyperparameters.** `d_model=256`, 4 layers, 8 heads, GELU MLP at 4× width, pre-norm
(`norm_first=True`), dropout 0 (data is online/infinite), learned token + absolute positional embeddings,
bias-free output head; ~3M params, under the 50M cap. Strict upper-triangular causal mask. Trained by the
fixed loop (AdamW lr=3e-4, wd=0.1, 3000 steps, batch 16).

**What to watch.** Strong on dense; a stubborn nonzero *sequence*-error rate on sparse (low read-error,
high seq-error — one slip flags a whole sequence); worst on long_ctx. If so, the fix is not "scale/train
more" but an explicit-state model that *writes and reads a register* instead of re-selecting it softly.

```python
# EDITABLE region of custom_strategy.py (lines 191-241) — step 1: vanilla causal Transformer
class FlipFlopModel(nn.Module):
    """Vanilla causal Transformer (the baseline / starting rung)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = 1024):
        super().__init__()
        d_model = 256
        n_layers = 4
        n_heads = 8
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_len {self.max_len}."
            )
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        h = self.tok(tokens) + self.pos(pos_ids)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal, is_causal=True)
        h = self.norm(h)
        return self.head(h)


def build_model(config: TaskConfig) -> nn.Module:
    """Construct the vanilla-Transformer FFLM."""
    return FlipFlopModel(
        vocab_size=VOCAB_SIZE,
        max_len=max(config.train_len, config.eval_long_len),
    )
```
