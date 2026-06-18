## Research question

A 1-bit memory should be the easiest thing in the world for a sequence model to track, yet vanilla
Transformers do not quite manage it. The flip-flop language `FFL(T, p)` of Liu, Ash, Goel,
Krishnamurthy, Zhang (NeurIPS 2023) is the minimal state-tracking task: a stream of `(instruction, bit)`
pairs over `Sigma = {w0, w1, i0, i1, r0, r1}`, where `w b` *writes* bit `b` into a single-bit memory,
`i b` is an *ignore* (a random visible bit, memory unchanged), and `r b` is a *read* whose bit `b` is
constrained to equal the current memory state — and the model is graded only on whether it predicts that
read bit. The one thing being designed is the **causal sequence-model architecture** that maps token-id
sequences to per-position next-token logits. Everything else — the data distribution, the AdamW
optimizer, the 3000-step training loop, and the read-only evaluation — is fixed.

The headline failure mode is the *sparse tail*. Trained on `FFL(p_i=0.8, T=512)`, a vanilla Transformer
must be tested OOD on `FFL(p_i=0.98)`, where long stretches of ignore tokens push the last relevant
write often more than 100 positions before the read. Liu et al. document that vanilla Transformers
exhibit **attention glitches** there: rare but *persistent* read errors whose probability does not vanish
with scale or with more data. The goal is an architecture that eliminates those errors — especially on
the long sparse tail — under the fixed substrate.

## Prior art before the first rung (sequence-transduction lineage)

The first rung is the vanilla Transformer the scaffold ships as its default. It is itself the resolution
of a line of sequence models; these precede the ladder.

- **Recurrent nets with BPTT (Elman 1990; Werbos 1990).** A hidden state carried step to step, trained by
  backpropagation through time. General and naturally causal, but the gradient through the recurrence is a
  product of per-step factors, so it vanishes or explodes exponentially in the lag — long-range
  dependencies are nearly untrainable. Gap: cannot reliably learn dependencies across many steps.
- **Convolutional / windowed sequence models.** Stack causal convolutions to widen the receptive field.
  Parallel and stable, but the reach is fixed by depth × kernel; a dependency longer than the receptive
  field is simply invisible. Gap: bounded, architecture-fixed context.
- **Attention as an add-on to recurrence (Bahdanau et al. 2014).** Let a decoder read a content-weighted
  combination of *all* encoder positions, so any source position is one hop away regardless of distance.
  Removes the fixed-window limit, but it was bolted onto an RNN backbone, so the sequential bottleneck and
  its training pathology remained. Gap: long-range *access* solved, but still riding a recurrence.
- **Self-attention only (the Transformer line).** Drop the recurrence entirely: every position attends to
  every earlier position through scaled dot-product attention, with positional encodings restoring order.
  Fully parallel over the sequence and every pair of positions one hop apart — exactly the inductive bias
  one would want for "find the last write and copy its bit." This is the scaffold's default fill, and the
  rung the climb starts from.

## The fixed substrate

A single-file harness is frozen and must not be touched. It (i) samples `FFL(p_i, T)` batches online —
the first token is always a write, instruction categories drawn from `(p_w, p_i, p_r)` with
`p_w = p_r = (1 - p_i)/2`, and read bits forced to the current memory state; (ii) runs teacher-forced
next-token training for 3000 steps with AdamW (`lr=3e-4`, `betas=(0.9, 0.999)`, `wd=0.1`, 50-step warmup,
linear decay, batch 16, grad-clip 1.0) under full-vocabulary cross-entropy at *every* position; and
(iii) grades only read positions, restricting the softmax to the two read-token logits `{r0, r1}` and
comparing `argmax` to the target bit. The harness exposes the vocabulary constants (`VOCAB_SIZE=6`,
`TOK_R0=4`, `TOK_R1=5`, …), the `TaskConfig` dataclass, and the data/eval helpers (`sample_ffl_batch`,
`read_mask`). It calls one factory, `build_model(config)`, and feeds the returned module
`LongTensor[batch, seq_len]` token ids.

## The editable interface

Exactly one region is editable — the `FlipFlopModel(nn.Module)` class plus the `build_model(config)`
factory in `pytorch-examples/synth_flip_flop/custom_strategy.py` (lines 191–241). The contract is fixed:
`forward(tokens)` takes a `LongTensor[batch, seq_len]` of ids in `0..5` and returns a
`FloatTensor[batch, seq_len, 6]` of next-token logits; the module must be **causal** (position `t` depends
only on positions `<= t`), must handle lengths up to 1024 (the long-context split), and must stay under
50M trainable parameters. Every rung on the ladder is a different fill of this same class.

The starting point is the scaffold default: a tiny vanilla causal Transformer (the paper's "baseline").

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

One training distribution, three OOD evaluation splits, seed 42. Train on `FFL(p_i=0.8, T=512)`. Evaluate
on **dense** (`FFL(p_i=0.5, T=512)`, 2000 sequences), **sparse** (`FFL(p_i=0.98, T=512)`, 2000
sequences — the headline tail), and **long_ctx** (`FFL(p_i=0.8, T=1024)`, 500 sequences). Per split the
metrics are `read_error_rate` (errors / total reads, lower better), `seq_error_rate` (fraction of
sequences with at least one read error, lower better), and `score = 1 - seq_error_rate` (higher better,
in `[0,1]`). The overall task score is the geometric mean of the three per-split `score`s.
