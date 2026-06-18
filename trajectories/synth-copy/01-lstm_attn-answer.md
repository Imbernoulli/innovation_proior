**Problem.** A small causal sequence model must copy / repeat / reverse a symbol string and keep working
at content lengths longer than it trained on. The single design lever is how token order enters the
model. The scaffold default (a learned absolute positional table) has no entry past the training range,
so it cannot even represent an OOD length — I want a rung-one reference whose order handling is
structural, not table-bound.

**Key idea.** Encode order through *recurrence* with no explicit positional code: the classical
additive-attention seq2seq of Bahdanau, Cho, Bengio (2015). A bidirectional LSTM reads the source
content (the slice between `BOS` and `SEP`) into per-position annotations `h_j`; the decoder, an
`LSTMCell` stepping over the target region, reads a content-weighted blend
`c_i = Σ_j α_{ij} h_j` of those annotations at each step, with weights
`α_{ij} = softmax_j v_a^⊤ tanh(W_a s_{i-1} + U_a h_j)`. No bottleneck vector, no positional encoding —
the recurrence supplies order.

**Why.** Content-based soft alignment is differentiable, non-monotonic (so `reverse` is expressible),
and length-invariant in scale (the read is a convex combination). Recurrence generalises the *rule* of
copying without a hard length cap, so it is the right structural floor against which the Transformer
positional schemes above will be judged. Its weakness is that the hidden state is still a fixed-width
channel and the decoder has no positional anchor, so long outputs drift.

**Scaffold edit.** Replace the editable block (lines 301–332). `build_positional_scheme` returns an
empty placeholder (recurrence is the positional mechanism); `build_model` returns the Bahdanau module
directly. The harness exposes only a single token stream and a fixed `forward(tokens) -> [B,T,V]`, so the
encoder/decoder split is recovered from `SEP_ID` inside `forward`, an `active = pos >= sep_pos` gate
restricts updates and logit writes to the target region, and a plain LSTM/LSTMCell + linear readout
replace the paper's GRU/maxout/beam-search.

**Hyperparameters.** Bidirectional LSTM encoder (1 layer, hidden `d_model//2` per direction), `LSTMCell`
decoder (hidden `d_model = 128`), additive attention of width `d_model`; all other training settings are
fixed by the loop (AdamW lr `5e-4`, wd `1e-2`, batch 256, 6000 steps).

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
