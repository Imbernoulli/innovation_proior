**Problem.** The sinusoidal transformer improved retention (0.771 vs the LSTM's 0.530) but bought zero
genuine OOD generalization: `abc`/`length-ood` OOD accuracy stayed at chance (0.524) and `exact` OOD
stayed 0.0, because its positional code is a function of *absolute index* and the test lengths (up to
256) are out-of-distribution relative to the ~65 indices seen in training. The lever is the positional
representation.

**Key idea.** Index each symbol by its **offset from the start of its current run of identical symbols**
(`a a a b b b → 1 2 3 | 1 2 3`), pass that offset through a learned embedding, and add it to the token
embedding. Now the `k`-th `a`, `k`-th `b`, `k`-th `c` share a vector, so a head can directly check "do
the runs reach the same offset" — the `a^n b^n c^n` decision — and on `exact` the offset is the running
within-run count. This is the count-indexed embedding idea ported from per-*number* digit alignment to
per-*run* symbol counting.

**The coverage fix (load-bearing).** A per-run counter still counts up, so the table needs rows up to
~85 at OOD length while training only updates rows up to ~21 — the same untrained-row failure that sank
the sinusoidal code. Fix: during training add a single shift drawn from `{0, …, train_offset=100}`,
*shared across the batch*, to every positive offset. The `+1` within-run step is preserved (so
adjacency/alignment hold), but the large rows get trained on short sequences. One shift per batch (not
per run) keeps all runs in an example starting at the same place, so "same significance ⇒ same vector"
holds within every example. At eval the shift is 0 (most-trained start).

**Adaptation to this harness (vs the arithmetic single-round trace).** Counted tokens are "not PAD and
not CLS" (computed by exclusion), not a digit-token set; run boundary is symbol-change, not
contiguous-digit-run; CLS (position 0) and PAD are excluded so they cannot shift the alignment; no
operand reversal; encoder + CLS pooling through a fixed head, not a decoder LM. Offsets clamped at
`max_count=4096`.

**Expectations.** `abc`/`length-ood` OOD accuracy should rise off chance for the first time, retention
should beat 0.771; in-distribution `abc` may dip (randomized-shift training is slightly noisier to fit);
`exact` OOD count regression may remain the residual 0.0.

**Hyperparameters.** Positional embedding `nn.Embedding(max_count+1=4097, hidden_dim)`, init
`std=0.02`; `train_offset=100`; rest identical to step 2 (2 pre-norm layers, 4 heads, FFN 4×, GELU,
`dropout=0.0`, embedding scaled by `sqrt(hidden_dim)`, `src_key_padding_mask`, CLS pool, LayerNorm).

```python
# EDITABLE region of pytorch-examples/synth_counting/custom_strategy.py — step 3: Abacus (count-positional embeddings)
def build_model(config: TaskConfig) -> nn.Module:
    """Transformer with Abacus-style count-positional embeddings (McLeish 2024)."""

    class AbacusPE(nn.Module):
        def __init__(
            self,
            dim: int,
            max_count: int = 4096,
            pad_id: int = 0,
            cls_id: int = 4,
            train_offset: int = 100,
        ):
            super().__init__()
            self.pad_id = pad_id
            self.cls_id = cls_id
            self.max_count = max_count
            self.train_offset = train_offset
            self.embed = nn.Embedding(max_count + 1, dim)
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        def block_offsets(self, tokens: torch.Tensor) -> torch.Tensor:
            B, T = tokens.shape
            if T == 0:
                return torch.zeros_like(tokens)

            valid = tokens.ne(self.pad_id) & tokens.ne(self.cls_id)
            cur = torch.zeros(B, dtype=torch.long, device=tokens.device)
            prev_token = tokens.new_full((B,), -1)
            prev_valid = torch.zeros(B, dtype=torch.bool, device=tokens.device)
            cols = []
            for t in range(T):
                tok = tokens[:, t]
                is_valid = valid[:, t]
                same_run = is_valid & prev_valid & tok.eq(prev_token)
                cur = torch.where(
                    is_valid,
                    torch.where(same_run, cur + 1, torch.ones_like(cur)),
                    torch.zeros_like(cur),
                )
                cols.append(cur)
                prev_token = tok
                prev_valid = is_valid

            offsets = torch.stack(cols, dim=1).clamp(max=self.max_count)
            if self.training and self.train_offset > 0:
                shift = torch.randint(
                    0,
                    self.train_offset + 1,
                    (),
                    dtype=offsets.dtype,
                    device=offsets.device,
                )
                shifted = (offsets + shift).clamp(max=self.max_count)
                offsets = torch.where(offsets > 0, shifted, offsets)
            return offsets

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            offsets = self.block_offsets(tokens)
            return self.embed(offsets)

    class AbacusTransformer(nn.Module):
        def __init__(self, cfg: TaskConfig):
            super().__init__()
            self.cfg = cfg
            self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_id)
            self.abacus = AbacusPE(cfg.hidden_dim, max_count=4096, pad_id=cfg.pad_id, cls_id=cfg.cls_id)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=4,
                dim_feedforward=4 * cfg.hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(cfg.hidden_dim)

        def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            x = self.tok_embed(tokens) * math.sqrt(self.cfg.hidden_dim)
            x = x + self.abacus(tokens)
            key_padding = tokens.eq(self.cfg.pad_id)
            h = self.encoder(x, src_key_padding_mask=key_padding)
            cls = h[:, 0]
            return self.norm(cls)

    return AbacusTransformer(config)
```
