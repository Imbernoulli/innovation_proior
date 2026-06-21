The sinusoidal transformer moved the metric in the predicted direction: `length-ood` retention rose from the LSTM's 0.530 to 0.771 and the extrapolation gap shrank from 0.470 to 0.230, confirming that attention's block comparison degrades more gracefully than the LSTM's compounding accumulator. But every prediction about its *limit* also held. `exact` OOD is still exactly 0.0; `abc`/`length-ood` OOD is still 0.524, dead chance, so the retention gain is "degrades less," not "generalizes"; and in-distribution `abc` actually *fell* to 0.754, the most diagnostic number on the board — the transformer is worse than the recurrent counter at the in-range task and bought *zero* genuine OOD generalization. The cause is now clear, and it is deeper than "the encoder never learned to read absolute phases at OOD lengths": **absolute index is the wrong thing to encode at all.** Deciding $a^n b^n c^n$ is deciding that the run of `a`'s, the run of `b`'s, and the run of `c`'s all reach the same length — a statement about *position within the current block of identical symbols*, not about absolute position. Likewise, counting `a`'s on `exact` is counting within runs of `a`. The latent variable both tasks track is the *within-run count*, and absolute index scrambles it: the third `a` sits at index 4 if it is the first block but index 40 if 36 symbols preceded it. The transformer was reconstructing "same block length" from positions that move around.

I propose **Abacus, a count-positional embedding**: index each symbol by its *offset from the start of its current run of identical symbols*, pass that offset through a learned embedding, and add it to the token embedding. Walk along the sequence; every time the symbol changes (or a run begins), reset a counter to 1, give each symbol of the current run the counter's value, and increment — so `a a a b b b c c c` produces run-offsets `1 2 3 | 1 2 3 | 1 2 3`. Now the $k$-th `a`, the $k$-th `b`, and the $k$-th `c` all receive the *identical* positional vector — offset $k$ — so a head can directly check "do the runs reach the same maximum offset," which is precisely the $a^n b^n c^n$ decision, and on `exact` the offset *is* the running within-run count. The latent the tasks track is handed to the model through the position channel, with no extra tokens. This is the count-indexed embedding idea of McLeish et al. 2024, ported from per-*number* digit alignment to per-*run* symbol counting — the adaptation the research question demands, because here the "number" whose internal offset matters is the run of identical symbols, not a multi-digit numeral.

By itself this walks straight back into the wall that killed rung two. The counter resets per run, good, but it still *counts up* within a run, and a length-256 OOD string has runs of $n$ up to ~85, so the embedding table needs rows up to ~85. Training only on $3n \le 64$ (so $n \le 21$) leaves the rows for offsets 22, 23, …, 85 *never updated* — garbage noise at test, exactly the untrained-row failure that sank the sinusoidal code. The per-run reset fixed the *significance* problem and did nothing for *coverage*. The coverage fix is the one piece that makes this rung genuinely extrapolate rather than just re-index: during training, before laying down a run's offsets, draw a single shift uniformly from $\{0, \dots, \text{train\_offset}\}$ and add it to every positive offset, so a run `1 2 3` becomes `1+s, 2+s, 3+s`. The within-run step stays exactly $+1$ — so adjacency and "same significance" are preserved — but because the shift ranges up to $\text{train\_offset} = 100$, over many batches the embedding rows from 1 all the way to ~$100 + \text{max-run-length}$ get exercised, including all the rows the OOD lengths will need. At evaluation I add nothing, so the runs use rows $1, 2, 3, \dots$, the smallest and most-trained start.

Two details of the shift are load-bearing. First, it is *shared across the batch* — one draw per forward pass, the same shift for every run in it — not redrawn per run. If I drew an independent shift per run, within a single example the `a`-run might start at offset 7 and the `c`-run at offset 40, and "same significance $\Rightarrow$ same vector" would break in exactly the example where I need it; with one shift per batch, every run in an example starts at the same place, the $k$-th symbol of every run still lands on the same row, and column alignment holds intact while coverage is bought across batches. Second, the shift is applied *only to positive offsets* — the non-run positions (PAD and the CLS at position 0) keep offset 0 and use row 0, reserved for "not a counted symbol," so they are never perturbed.

The run-boundary definition must respect this task's vocabulary, and this is where the adaptation is concrete. The run is computed over content tokens, but two are *excluded* from the run logic entirely: `pad_id` (carries no symbol) and `cls_id` (the CLS, the subtle one — it sits at position 0 before any content, and if it participated it would start a spurious run or, worse, merge with a following `a` and shift every offset by one, breaking the alignment between the `a`-run and the `b`/`c`-runs). So I sweep the sequence, marking a position *valid* only if it is neither PAD nor CLS; a position *continues a run* only if it is valid, the previous position was valid, and the token equals the previous token; otherwise a valid position *starts* a new run at offset 1, and an invalid position gets offset 0. Offsets are clamped at a generous `max_count` (4096) so the lookup never indexes out of bounds. Unlike the arithmetic single-round trace, there is no digit-token set and no `isin` — counted tokens are "not PAD and not CLS," computed by exclusion; there is no operand reversal — the sequence is read left to right and offset 1 is the first symbol of a run; and the whole thing is an *encoder* feeding a CLS-pooled head, not a decoder LM.

The rest of the encoder I keep identical to rung two on purpose, so the comparison isolates the positional change: token embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)` scaled by $\sqrt{\text{hidden\_dim}}$, *add* the count-positional embedding (initialized small, `std=0.02`, so it starts as a gentle bias and the optimizer grows the rows it uses), 2 pre-norm `nn.TransformerEncoderLayer`s with 4 heads, FFN width $4\times\text{hidden\_dim}$, GELU, `dropout=0.0`, `src_key_padding_mask = tokens.eq(pad_id)`, CLS pooling `h[:, 0]`, final `LayerNorm`. Same depth, heads, pooling, and mask as rung two — the only thing that changed is that the sinusoidal absolute PE became a learned, run-indexed, training-shift-randomized embedding. The claim this whole construction rests on is that `abc`/`length-ood` OOD accuracy should rise off chance for the first time, because the run-offset is the actual latent the decision needs and the shift randomization keeps the needed rows in-distribution at test length; retention should then beat 0.771. In-distribution `abc` may dip toward the transformer's level, since the randomized shift makes the in-range fit slightly noisier — a trade the geometric-mean-of-scores rewards. And `exact` OOD count regression through a pooled CLS may remain the residual 0.0 even with the right positional signal — which would then name what a method past this ladder must attack.

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
