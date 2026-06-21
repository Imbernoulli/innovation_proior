## Research question

The flip-flop language `FFL(T, p)` is a minimal state-tracking task: a stream of `(instruction, bit)` pairs over `Sigma = {w0, w1, i0, i1, r0, r1}`, where `w b` writes bit `b` into a single-bit memory, `i b` is an ignore token (random visible bit, memory unchanged), and `r b` is a read whose bit `b` must equal the current memory state. The model is graded only on predicting read bits. The only design choice is the causal sequence-model architecture; everything else — the data distribution, the AdamW optimizer, the 3000-step training loop, and the read-only evaluation — is fixed.

Trained on `FFL(p_i=0.8, T=512)`, the model is tested OOD on `FFL(p_i=0.98)`, where long ignore stretches push the last relevant write often more than 100 positions before the read. The goal is an architecture that handles these longer-range dependencies under the fixed substrate.

## Prior art / Background / Baselines

- **Recurrent nets with BPTT.** A hidden state carried step to step and trained by backpropagation through time. The gradient through the recurrence is a product of per-step factors.
- **Causal convolutional / windowed models.** Stacked causal convolutions widen the receptive field in parallel and stably, with the reach determined by depth and kernel size.
- **Attention over an encoder plus recurrent decoder.** A decoder reads a content-weighted combination of all encoder positions, so any position is one hop away.
- **Self-attention-only Transformer.** Causal scaled dot-product attention attends to all earlier positions, with positional encodings restoring order. Fully parallel and one-hop across positions, this is the scaffold's default fill and the starting baseline.

## Fixed substrate / Code framework

A single-file harness is frozen and must not be touched. It (i) samples `FFL(p_i, T)` batches online — the first token is always a write, instruction categories are drawn with `p_w = p_r = (1 - p_i)/2`, and read bits are forced to the current memory state; (ii) runs teacher-forced next-token training for 3000 steps with AdamW (`lr=3e-4`, `betas=(0.9, 0.999)`, `wd=0.1`, 50-step warmup, linear decay, batch 16, grad-clip 1.0) under full-vocabulary cross-entropy at every position; and (iii) grades only read positions, restricting the softmax to the two read-token logits `{r0, r1}` and comparing `argmax` to the target bit. The harness exposes the vocabulary constants (`VOCAB_SIZE=6`, `TOK_R0=4`, `TOK_R1=5`, …), the `TaskConfig` dataclass, and the data/eval helpers (`sample_ffl_batch`, `read_mask`). It calls one factory, `build_model(config)`, and feeds the returned module `LongTensor[batch, seq_len]` token ids.

## Editable interface

Exactly one region is editable — the `FlipFlopModel(nn.Module)` class plus the `build_model(config)` factory in `pytorch-examples/synth_flip_flop/custom_strategy.py` (lines 191–241). The contract is fixed: `forward(tokens)` takes a `LongTensor[batch, seq_len]` of ids in `0..5` and returns a `FloatTensor[batch, seq_len, 6]` of next-token logits; the module must be **causal** (position `t` depends only on positions `<= t`), must handle lengths up to 1024 (the long-context split), and must stay under 50M trainable parameters. Every rung on the ladder is a different fill of this same class.

The starting point is the scaffold default: a tiny vanilla causal Transformer.

```python
# EDITABLE region of custom_strategy.py (lines 191-241) — default fill: vanilla causal Transformer
class FlipFlopModel(nn.Module):
    """Default scaffold: a tiny causal Transformer (vanilla, no tricks)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = 1024):
        super().__init__()
        d_model = 128
        n_layers = 2
        n_heads = 4
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
    """Construct the autoregressive model graded by the FFLM evaluator."""
    return FlipFlopModel(vocab_size=VOCAB_SIZE, max_len=max(config.train_len, config.eval_long_len))
```

## Evaluation settings

One training distribution, three OOD evaluation splits, seed 42. Train on `FFL(p_i=0.8, T=512)`. Evaluate on **dense** (`FFL(p_i=0.5, T=512)`, 2000 sequences), **sparse** (`FFL(p_i=0.98, T=512)`, 2000 sequences — the headline tail), and **long_ctx** (`FFL(p_i=0.8, T=1024)`, 500 sequences). Per split the metrics are `read_error_rate` (errors / total reads, lower better), `seq_error_rate` (fraction of sequences with at least one read error, lower better), and `score = 1 - seq_error_rate` (higher better, in `[0,1]`). The overall task score is the geometric mean of the three per-split `score`s.
