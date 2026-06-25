The task is to track a single-bit memory across a stream of write, ignore, and read tokens and to predict each read bit correctly, with the architecture as the only free variable — the data, the AdamW optimizer, the 3000-step loop, and the read-only grading are all frozen. Before I reach for anything clever I want to measure the model that *should* already be enough, because if it is enough I am done, and if it is not, the precise way it fails is what tells me what to build next. The flip-flop read is, on paper, the single thing attention is best at: a read at position $t$ needs the bit of the most recent write before $t$, and with self-attention every earlier position is one hop away. So this rung is not a strawman — it is the model with the *right-looking* inductive bias, and that is exactly why it deserves to be measured first.

I propose the vanilla causal Transformer as the starting rung. It maps a `LongTensor[batch, seq_len]` of token ids in $0..5$ to per-position next-token logits over the 6-symbol vocabulary, and the whole computation is a stack of pre-norm self-attention layers under a strict causal mask. Each token is embedded with a learned `nn.Embedding(6, d_model)` and a learned absolute positional embedding `nn.Embedding(max_len, d_model)` is added — learned rather than sinusoidal because the lengths here are fixed and modest and a learned table is the scaffold's convention. The body is a stack of `TransformerEncoderLayer` blocks with `norm_first=True`, each one multi-head self-attention followed by a GELU MLP at $4\times$ width, and every layer is fed a strict upper-triangular boolean mask so position $t$ can only attend to positions $\le t$. That mask is what makes the encoder *causal*, which the contract requires and the grading enforces — a read prediction may use only the past. A final LayerNorm and a bias-free linear head project to the 6 logits. I size it at $d\_model=256$, 4 layers, 8 heads, GELU MLP at $4\times$ width, dropout 0 — roughly 3M parameters, comfortably under the 50M cap. This is the standard architecture for this task, downsized from the 6-layer/$d\_model=512$ canonical baseline so the three evaluation splits finish inside the wall-clock budget; the qualitative behaviour, attention-glitch tail included, is the same.

The point worth being explicit about is *how* the read computation has to be expressed here, because that mechanism is the whole story of the failure to come. There is no place in a Transformer to keep a literal one-bit register that gets overwritten on each write and read out on each read. The only thing carried forward is the residual stream at each position, so the only way a read position learns the current memory bit is by *attending back to the write that set it*. The model must therefore learn end to end, and compose through depth, three sub-behaviours: a write-versus-ignore distinction (separable from the embeddings, since writes and ignores are different token ids); a query at the read position whose dot product with write keys is large and with ignore keys small, so attention concentrates on writes; and, among the writes, a recency preference so the *most recent* write wins rather than an older one. This is the induction-head circuit Transformers reliably discover — earlier layers build the write/ignore distinction and a coarse positional summary, later layers sharpen the recency-weighted selection. But every part of it is *soft*: it is a continuous attention distribution, and the recency preference in particular must fight off every older write whose bit happens to differ from the most recent one, which on a sparse sequence can be many distant competitors.

This is exactly why I expect it to crack, and the crack is structural rather than a matter of capacity or training time. On the training-like and dense distributions, where the last write is usually a few tokens back, the induction-head computation is short-range and a 4-layer Transformer has ample capacity for it. The trouble is the sparse tail, $\text{FFL}(p_i=0.98)$, where the relevant write can sit 100, 200, or more positions before the read with a long run of ignores in between. Softmax attention is a *soft* selection: the weights are a normalised distribution over all earlier positions, and even a well-trained head puts a little mass on the wrong positions. Usually that residual mass is harmless — the dominant write still wins the argmax — but over a 2000-sequence evaluation the rare configuration where the smear flips the predicted bit *will* occur. That is the **attention glitch**: a low-probability-per-read error that, accumulated over many reads per sequence and many sequences, shows up as a stubborn nonzero *sequence*-error rate. Because one slip anywhere flags the whole sequence, I expect a low *read*-error rate but a high *sequence*-error rate — and scale and data do not drive it to zero, because the defect is in the softness of the mechanism, not in undertraining. The long-context split sharpens the worry from a second angle: training is at $T=512$ and evaluation at $T=1024$, and learned *absolute* positions get far less gradient at the high indices, so the model's behaviour at far-out positions is poorly calibrated. Combined with more distractor positions to smear over, I expect long_ctx to be the worst of the three.

A few smaller choices follow from the substrate. Dropout is zero because the data is generated online and effectively infinite — there is no finite training set to overfit, so regularisation would only add noise. Pre-norm rather than post-norm trains more stably at this depth, which matters because the frozen warmup is only 50 steps. The head has no bias because that bias is redundant with the final LayerNorm's affine shift feeding into it. And I keep the full next-token cross-entropy the fixed loop imposes — the model is trained to predict every token even though it is graded only on reads — because I cannot change it; it is the same handicap every rung carries, so it is fair. The grading restricts the softmax at each read to just the two read-token logits $\{r0, r1\}$ and takes the argmax, so a glitch in the numbers is unambiguously a *memory* error, not a confusion about the instruction stream. That is the falsifiable expectation I set against the next rung: strong on dense, a long tail of errors on sparse, worst on long_ctx — and if so, the fix is not "train more" or "make it bigger" but a model that carries the bit in an explicit *state* rather than re-selecting it softly at every read.

```python
# EDITABLE region of custom_strategy.py (lines 191-241) — step 1: vanilla causal Transformer
class FlipFlopModel(nn.Module):
    """Vanilla causal Transformer (the 'baseline' architecture)."""

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
