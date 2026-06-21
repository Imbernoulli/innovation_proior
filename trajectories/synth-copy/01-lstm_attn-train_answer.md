The one design lever in this task is how token order enters the model, and the scaffold's starting fill — a *learned* absolute positional table — already fails the only question that matters here by inspection. A learned table has one trainable vector per slot and literally no entry past the training range, so at content length 30 there is simply nothing to look up; it cannot even *represent* an out-of-distribution length, let alone generalize to one. Before I commit to any Transformer scheme, I want a rung-one reference whose handling of position is *structural* — a property of the architecture, not of a table that runs out — so that whatever it does at unseen lengths is something I can reason about rather than a missing index. The classical answer that encodes order with no explicit code at all, and the non-Transformer baseline this harness names, is the additive-attention sequence-to-sequence model.

I propose the **Bahdanau LSTM seq2seq with additive attention** (Bahdanau, Cho, Bengio 2015), fitted to this single-stream harness. The disease it was invented to cure is the fixed-length-context bottleneck of the plain recurrent encoder-decoder: an RNN reads the source left to right, $h_t = f(x_t, h_{t-1})$, and the older design hands the decoder only the final state $c = h_T$ as one fixed-width summary of the entire input. Everything the decoder will ever need has to be squeezed through that single vector, and a constant-size vector cannot hold an unbounded amount of content, so the model degrades on long inputs — worst beyond the lengths it trained on. The cure is to keep *all* the encoder's per-position annotations $h_1, \dots, h_T$ and let the decoder, at each output step, search the source for the position it should be reading and pull a content-weighted blend from there. With weights that are nonnegative and sum to one, the decoder reads
$$c_i = \sum_j \alpha_{ij}\, h_j,$$
a per-step context recomputed every output step rather than one summary frozen at the start. That removes the bottleneck: a long source is no harder than a short one, because the decoder only ever consumes one position's worth of blended content at a time.

The weights come from a content match. Before emitting $y_i$ the decoder knows its recurrent state $s_{i-1}$ — what it has produced and what it intends next — and each source position is represented by its annotation $h_j$, so the score $e_{ij} = a(s_{i-1}, h_j)$ measures how well position $j$ matches what the decoder is about to do. Because the query and key vectors need not live in the same space, the natural cheap scorer is a one-hidden-layer additive MLP,
$$e_{ij} = v_a^\top \tanh(W_a\, s_{i-1} + U_a\, h_j), \qquad \alpha_{ij} = \operatorname{softmax}_j(e_{ij}).$$
The softmax over positions is what keeps the read length-invariant in *scale*: $c_i$ is a convex combination of annotations, so its magnitude does not grow with the number of source positions. I use $s_{i-1}$ rather than $s_i$ as the query because $s_i$ depends on $c_i$, which would be circular — and the pre-step state is also the right semantics for "where should I look now." This content-based, non-monotonic, differentiable soft alignment is the load-bearing machinery; the non-monotonicity is what lets `reverse` be expressed at all, since the alignment can run backward through the source.

The part that matters most for this trajectory is how that method has to be expressed in *this* harness, because it is not the generic translation network. The scaffold gives me a single token stream $[\text{BOS}]\,x_1\dots x_T\,[\text{SEP}]\,y_1\dots y_M\,[\text{EOS}]$ and a fixed `forward(tokens) -> [B,T,V]` API — there is no separate source and target tensor — so I have to manufacture the encoder-decoder split inside `forward`. The `SEP` token is the seam: I find `sep_pos` per row, treat the content between `BOS` and `SEP` as the source, and treat everything from `SEP` onward as the decoding region. I embed the source slice, run a *bidirectional* LSTM over it so each annotation $h_j$ sees both sides of position $j$ (forward direction summarizes $x_1..x_j$, backward $x_j..x_T$, concatenated to width $d_{\text{model}}$), and mask the padded positions to $-\infty$ before the softmax so attention never reads them. The decoder is an `LSTMCell` initialized from a $\tanh$ projection of the masked-mean encoder summary; it steps over *every* position of the stream, but a per-row boolean `active = pos >= sep_pos` gates both the state update (the cell's `h, c` are held frozen on inactive positions) and the logit write, so predictions are produced only on the target region — exactly where the loss mask lives. The decoder reads `tokens[:, pos]` as its input embedding at every step (there is no teacher-forced separate target tensor, so the prefix flows through too, but its logits are discarded by the gate), and because the whole thing is one `forward` over a $[B,T]$ tensor with attention recomputed at all $T$ positions, this is by far the slowest baseline on the clock.

Two simplifications are forced by what the harness exposes, and I keep them deliberate. There is no bidirectional GRU with reset/update gates and a maxout deep output here — I use a plain bidirectional `LSTM` encoder and a single `LSTMCell` decoder (the lighter gated unit with the same long-range-gradient property) and a bare linear readout instead of maxout; there is no beam search (evaluation is greedy autoregressive decoding driven by the fixed loop); and there is no fertility/NULL machinery. The `build_positional_scheme` hook is returned as an empty placeholder with all three callables `None`, because the recurrence *is* the positional mechanism: the encoder's two directional passes stamp order into the annotations and the decoder's sequential stepping stamps order into the output, so there is nothing for the Transformer's positional hooks to do. `build_model` returns the recurrent module directly.

What I expect this floor to look like is sharp and falsifiable. At training lengths every variant should be easy — short source, no bottleneck — so content-based soft alignment plus recurrence should give near-perfect in-distribution exact match across copy, repeat, and reverse; the alignment merely has to learn to point at the right source position (the same index for `delim`, the mirrored index for `reverse`, the modular index for `repeat`). The OOD split is where I am genuinely uncertain. Recurrence and content attention have no *hard* length cap the way the learned table does, so in principle the *rule* of copying should transfer — but the encoder's hidden state is still a fixed-width channel that crowds more content through the same annotations at length 30–40, and the decoder state, seeded from a single masked-mean summary, accumulates drift over a longer output with no positional anchor to correct it. So I expect the sequence-level exact match to be the brittle metric: one wrong symbol anywhere fails the whole sequence, and over 30–40 steps a single drift is likely, while token accuracy degrades more gracefully because most symbols can still be aligned. Perfect ID, zero OOD exact match, non-trivial OOD token accuracy — that is the floor this rung should establish, and it is exactly the shape the explicit-position Transformer rungs above will have to diagnose and beat.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- step 1: Bahdanau LSTM seq2seq attention
class BahdanauSeq2SeqAttention(nn.Module):
    """LSTM encoder-decoder with additive attention."""

    def __init__(self, config: TaskConfig):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even for the bidirectional encoder")
        self.config = config
        self.vocab = vocab_size(config)
        self.token_embed = nn.Embedding(self.vocab, config.d_model, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.init_h = nn.Linear(config.d_model, config.d_model)
        self.init_c = nn.Linear(config.d_model, config.d_model)
        self.attn_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_v = nn.Linear(config.d_model, 1, bias=False)
        self.decoder = nn.LSTMCell(2 * config.d_model, config.d_model)
        self.out = nn.Linear(2 * config.d_model, self.vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        device = tokens.device
        sep_pos = tokens.eq(SEP_ID).long().argmax(dim=1)
        src_lens = (sep_pos - 1).clamp(min=1)
        max_src = int(src_lens.max().item())
        src = tokens[:, 1 : 1 + max_src].clone()
        src_idx = torch.arange(max_src, device=device).unsqueeze(0)
        src_mask = src_idx < src_lens.unsqueeze(1)
        src = torch.where(src_mask, src, torch.full_like(src, PAD_ID))

        enc_emb = self.token_embed(src)
        enc_emb = enc_emb * src_mask.unsqueeze(-1).to(enc_emb.dtype)
        enc_out, _ = self.encoder(enc_emb)
        enc_out = enc_out * src_mask.unsqueeze(-1)
        denom = src_lens.to(enc_out.dtype).unsqueeze(1)
        summary = enc_out.sum(dim=1) / denom
        h = torch.tanh(self.init_h(summary))
        c = torch.tanh(self.init_c(summary))
        keys = self.attn_key(enc_out)
        logits = enc_out.new_zeros(B, T, self.vocab)

        for pos in range(T):
            active = pos >= sep_pos
            query = self.attn_query(h).unsqueeze(1)
            scores = self.attn_v(torch.tanh(keys + query)).squeeze(-1)
            scores = scores.masked_fill(~src_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
            dec_in = torch.cat([self.token_embed(tokens[:, pos]), context], dim=-1)
            new_h, new_c = self.decoder(dec_in, (h, c))
            active_f = active.unsqueeze(1)
            h = torch.where(active_f, new_h, h)
            c = torch.where(active_f, new_c, c)
            step_logits = self.out(torch.cat([h, context], dim=-1))
            logits[:, pos, :] = torch.where(active_f, step_logits, logits[:, pos, :])
        return logits


def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """The LSTM ignores the scheme; return an empty placeholder."""
    return PositionalScheme(
        name="lstm_none",
        token_embedding_extra=None,
        attn_bias=None,
        rotary=None,
        extra_modules=nn.ModuleList(),
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Bahdanau-style recurrent encoder-decoder with additive attention."""
    return BahdanauSeq2SeqAttention(config)
```
