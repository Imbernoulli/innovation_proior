# T5: the Text-to-Text Transfer Transformer

## Problem it solves

Transfer learning for NLP (pre-train on unlabeled text, fine-tune on a task) had
fragmented into incompatible objectives, architectures, corpora, and fine-tuning recipes,
each task carrying its own output head. There was no controlled way to vary one factor at
a time. T5 removes the obstacle by giving every task a single interface, then uses that
interface both to run a systematic study and to build a strong model.

## Key idea

Cast **every** text task as **text → text**: feed the model an input string (with a short
text prefix naming the task) and train it to generate an output string. Classification
emits the label word; regression emits a number rounded to a string; QA, summarization,
and translation emit their targets directly. This yields one loss (teacher-forced
cross-entropy), one decoding procedure (autoregressive, greedy at test), and makes the
remaining design axes orthogonal knobs.

The choices that the unified study lands on:

- **Architecture**: original-style **encoder-decoder** Transformer. Bidirectional
  (fully-visible) encoder over the input; causal decoder that cross-attends into the
  encoder. At matched FLOPs it beats decoder-only and prefix-LM variants. Sharing
  encoder/decoder parameters halves the parameter count at almost no quality cost.
- **Pre-training objective**: **span corruption**. Corrupt 15% of tokens in contiguous
  spans (mean span length 3), replace each span with a single unique **sentinel** token,
  and train the model to generate only the dropped spans, each prefixed by its sentinel
  and ending with a final sentinel. Denoising beats causal LM; among denoising variants
  quality is flat, so the cheapest (shortest targets) wins.
- **Transformer simplifications**: RMSNorm (scale only, no mean-subtraction, no bias),
  **pre-norm** placement (normalize the sublayer input, residual outside), no biases in
  projections, feed-forward width d_ff = 4·d_model with ReLU.
- **Position**: a learned **scalar** relative-position **bias** per (offset-bucket, head)
  added directly to the attention logit; 32 total buckets, split by sign for
  bidirectional self-attention, exact for small offsets and **logarithmic** up to
  distance 128 with a catch-all beyond; **shared across all layers**.
- **No 1/√d_k attention scaling** in the forward pass — folded into the weight
  initialization (query-projection init std ∝ (d_model·d_kv)^(−1/2)).
- **Embeddings tied** across encoder input, decoder input, and the output softmax.
- **Optimizer**: Adafactor; inverse-square-root LR schedule with 10⁴ warmup.
- **Vocabulary**: 32k SentencePiece word-pieces trained on a 10:1:1:1 En/De/Fr/Ro mix
  (the fixed vocab must cover translation targets), plus sentinel IDs.
- **Data**: **C4** — Common Crawl heuristically cleaned (terminal-punctuation lines,
  length filters, bad-words/JavaScript/lorem-ipsum/curly-brace removal, citation
  stripping, boilerplate removal, three-sentence-span dedup, langdetect-English ≥ 0.99),
  ~750 GB of clean English.

## Final objective and pipeline

Pre-train: sample a span of C4 text, apply span corruption to get (input, target),
minimize teacher-forced cross-entropy of the target. Fine-tune: convert each downstream
task to (input, target) text via a task prefix and minimize the same loss. Infer: greedy
autoregressive decoding; for classification/regression, parse the generated string back to
a label/number (any out-of-set output counts as wrong).

## Code

Architecture (PyTorch, mirroring the canonical implementation):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class T5LayerNorm(nn.Module):
    """RMSNorm: scale by root-mean-square only; no mean-subtraction, no bias."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.type_as(self.weight)


class T5LayerFF(nn.Module):
    """Pre-norm position-wise feed-forward, d_ff = 4 * d_model, ReLU, no biases."""
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.wo(self.dropout(F.relu(self.wi(self.layer_norm(x)))))
        return x + self.dropout(h)


class T5Attention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, is_decoder,
                 has_relative_bias=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.d_kv, self.n_heads = d_kv, n_heads
        self.inner = d_kv * n_heads
        self.is_decoder = is_decoder
        self.num_buckets, self.max_distance = num_buckets, max_distance
        # No biases; no explicit 1/sqrt(d_k) -- it is folded into the init.
        self.q = nn.Linear(d_model, self.inner, bias=False)
        self.k = nn.Linear(d_model, self.inner, bias=False)
        self.v = nn.Linear(d_model, self.inner, bias=False)
        self.o = nn.Linear(self.inner, d_model, bias=False)
        self._reset_parameters(d_model)
        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _reset_parameters(self, d_model):
        nn.init.normal_(self.q.weight, mean=0.0,
                        std=(d_model * self.d_kv) ** -0.5)
        nn.init.normal_(self.k.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.v.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.o.weight, mean=0.0, std=self.inner ** -0.5)

    @staticmethod
    def _bucket(rel_pos, bidirectional, num_buckets, max_distance):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (rel_pos > 0).long() * num_buckets
            rel_pos = rel_pos.abs()
        else:
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact
        large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        large = torch.min(large, torch.full_like(large, num_buckets - 1))
        return ret + torch.where(is_small, rel_pos, large)

    def compute_bias(self, q_len, k_len, device):
        ctx = torch.arange(q_len, device=device)[:, None]
        mem = torch.arange(k_len, device=device)[None, :]
        buckets = self._bucket(mem - ctx, not self.is_decoder,
                               self.num_buckets, self.max_distance)
        return self.relative_attention_bias(buckets).permute([2, 0, 1]).unsqueeze(0)

    def forward(self, hidden, kv=None, mask=None, position_bias=None):
        B, qlen = hidden.shape[:2]
        kv_in = hidden if kv is None else kv
        klen = kv_in.shape[1]
        shape = lambda t: t.view(B, -1, self.n_heads, self.d_kv).transpose(1, 2)
        q, k, v = shape(self.q(hidden)), shape(self.k(kv_in)), shape(self.v(kv_in))
        scores = torch.matmul(q, k.transpose(3, 2))          # unscaled
        if position_bias is None:
            if self.has_relative_bias:
                position_bias = self.compute_bias(qlen, klen, hidden.device)
            else:
                position_bias = torch.zeros((1, self.n_heads, qlen, klen),
                                            device=hidden.device, dtype=scores.dtype)
            if mask is not None:
                position_bias = position_bias + mask
        scores = scores + position_bias
        attn = F.softmax(scores.float(), dim=-1).type_as(scores)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.inner)
        return self.o(out), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, is_decoder, has_relative_bias, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, d_kv, n_heads, is_decoder, has_relative_bias)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_bias=None):
        y, position_bias = self.attn(self.layer_norm(x), mask=mask,
                                     position_bias=position_bias)
        return x + self.dropout(y), position_bias


class T5LayerCrossAttention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, d_kv, n_heads, True, has_relative_bias=False)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv, mask=None, position_bias=None):
        y, position_bias = self.attn(self.layer_norm(x), kv=kv, mask=mask,
                                     position_bias=position_bias)
        return x + self.dropout(y), position_bias


class T5Block(nn.Module):
    def __init__(self, d_model, d_ff, d_kv, n_heads, is_decoder,
                 has_relative_bias, dropout):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = T5LayerSelfAttention(d_model, d_kv, n_heads, is_decoder,
                                              has_relative_bias, dropout)
        if is_decoder:
            self.cross_attn = T5LayerCrossAttention(d_model, d_kv, n_heads, dropout)
        self.ff = T5LayerFF(d_model, d_ff, dropout)

    def forward(self, x, mask=None, position_bias=None,
                enc_hidden=None, enc_mask=None, enc_position_bias=None):
        x, position_bias = self.self_attn(x, mask=mask, position_bias=position_bias)
        if self.is_decoder and enc_hidden is not None:
            x, enc_position_bias = self.cross_attn(x, enc_hidden, mask=enc_mask,
                                                   position_bias=enc_position_bias)
        x = self.ff(x)
        return x, position_bias, enc_position_bias


class T5Stack(nn.Module):
    def __init__(self, embed, n_layers, d_model, d_ff, d_kv, n_heads,
                 is_decoder, dropout):
        super().__init__()
        self.embed = embed
        self.blocks = nn.ModuleList([
            T5Block(d_model, d_ff, d_kv, n_heads, is_decoder,
                    has_relative_bias=(i == 0), dropout=dropout)
            for i in range(n_layers)
        ])
        self.final_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None, enc_hidden=None, enc_mask=None):
        x = self.dropout(self.embed(input_ids))
        position_bias = enc_position_bias = None
        for block in self.blocks:
            x, position_bias, enc_position_bias = block(
                x, mask=mask, position_bias=position_bias,
                enc_hidden=enc_hidden, enc_mask=enc_mask,
                enc_position_bias=enc_position_bias)
        return self.dropout(self.final_norm(x))


class T5ForConditionalGeneration(nn.Module):
    def __init__(self, vocab_size, d_model=768, d_ff=3072, d_kv=64,
                 n_heads=12, n_layers=12, dropout=0.1, pad_id=0, decoder_start_id=0):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5Stack(self.shared, n_layers, d_model, d_ff, d_kv,
                               n_heads, is_decoder=False, dropout=dropout)
        self.decoder = T5Stack(self.shared, n_layers, d_model, d_ff, d_kv,
                               n_heads, is_decoder=True, dropout=dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight                 # weight tying
        self.d_model, self.pad_id, self.decoder_start_id = d_model, pad_id, decoder_start_id

    def _shift_right(self, labels):
        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1].clone()
        shifted[..., 0] = self.decoder_start_id
        shifted[shifted == -100] = self.pad_id
        return shifted

    @staticmethod
    def _causal_mask(seq_len, device):
        m = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return (1.0 - m) * torch.finfo(torch.float32).min

    def forward(self, input_ids, attention_mask=None, labels=None):
        enc_mask = None
        if attention_mask is not None:
            enc_mask = (1.0 - attention_mask[:, None, None, :].float()) \
                * torch.finfo(torch.float32).min
        enc_hidden = self.encoder(input_ids, mask=enc_mask)
        if labels is None:
            raise ValueError("labels are required for teacher-forced training")
        dec_input = self._shift_right(labels)
        dec_self_mask = self._causal_mask(dec_input.size(1), dec_input.device)
        dec_hidden = self.decoder(dec_input, mask=dec_self_mask,
                                  enc_hidden=enc_hidden, enc_mask=enc_mask)
        logits = self.lm_head(dec_hidden * (self.d_model ** -0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)
        return loss, logits
```

Span-corruption pre-training transform and the text-to-text task interface:

```python
def span_corrupt(token_ids, sentinels, noise_density=0.15, mean_span_len=3.0):
    """Corrupt ~15% of tokens in contiguous spans; replace each span with one
    unique sentinel. Target = dropped spans, each prefixed by its sentinel, with a
    final sentinel appended. Both input and target are short."""
    n = len(token_ids)
    n_noise = max(1, round(n * noise_density))
    n_spans = max(1, round(n_noise / mean_span_len))
    mask = _random_spans_noise_mask(n, n_noise, n_spans)   # contiguous noise spans
    inp, tgt, s, i = [], [], 0, 0
    while i < n:
        if mask[i]:
            sent = sentinels[s]; s += 1
            inp.append(sent); tgt.append(sent)
            while i < n and mask[i]:
                tgt.append(token_ids[i]); i += 1
        else:
            inp.append(token_ids[i]); i += 1
    tgt.append(sentinels[s])
    return inp, tgt


def to_text_to_text(task, example):
    """Every task -> (input_text, target_text); a short prefix selects the task."""
    if task == "mnli":
        return (f"mnli premise: {example['premise']} hypothesis: {example['hypothesis']}",
                example["label_text"])
    if task == "stsb":
        return (f"stsb sentence1: {example['s1']} sentence2: {example['s2']}",
                f"{round(example['score'] * 5) / 5:.1f}")
    if task == "summarize":
        return (f"summarize: {example['article']}", example["highlights"])
    if task == "translate_en_de":
        return (f"translate English to German: {example['en']}", example["de"])
    if task == "squad":
        return (f"question: {example['question']} context: {example['context']}",
                example["answer"])
    raise ValueError(task)
```

The model sizes scale this base configuration (d_model=768, 12+12 layers, 12 heads,
d_kv=64, d_ff=3072, ~220M params) up to the 11B-parameter variant by widening d_ff and
d_model, adding heads, and deepening the stacks.
