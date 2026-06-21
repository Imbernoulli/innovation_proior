Transfer learning for NLP has become both very good and very messy at the same time. The recipe everyone agrees on — pre-train a large network on a pile of unlabeled text, then fine-tune on a downstream task — works remarkably well, but every new result changes three or four things at once: it swaps the pre-training objective (next-token prediction for masked reconstruction), and the architecture (a decoder for an encoder), and the corpus (Wikipedia for a web crawl), and the fine-tuning recipe, and reports a number that beats last month's. So which of those moves actually did the work? Nobody can say, because the methods cannot even be lined up against each other. They do not share an interface: BERT hands me a classification head, GPT a left-to-right decoder, a span model a start/end pointer, a translation model a full seq2seq decoder. You cannot hold "everything else fixed" when "everything else" includes the shape of the output. Before I can ask any clean scientific question — does objective $X$ beat objective $Y$, does architecture $A$ beat architecture $B$ — I have to remove the one structural obstacle that makes the comparison impossible: the per-task output interface, which welds the loss and the architecture to the task and forbids free substitution of any other component. Prior unification attempts circled this. Casting everything as question answering (decaNLP) mandates simultaneous multi-task training and a rigid Q/A format. Casting tasks as zero-shot language modeling (GPT-2) uses a decoder-only model that never gets to encode its context bidirectionally and never fine-tunes. Casting tasks as span extraction (Keskar et al.) cannot express generative tasks, because you cannot enumerate every possible German sentence. Each had the right instinct and stopped short.

The method is T5, the Text-to-Text Transfer Transformer, and the load-bearing move is to give every task — without exception — the same interface: a string of text in, a string of text out. Classification does not emit a class index through a softmax over a fixed label set; it emits the word, "entailment". Question answering emits the answer text; summarization and translation already are text-to-text. Even regression bends: STS-B asks for a similarity in $[1,5]$, and since the human annotations cluster on increments of $0.2$, I round to the nearest $0.2$ and emit the number as a literal string, "$3.8$", parsing it back at test time — which quietly turns regression into a $21$-way classification with no special-casing, because it is just text out. The only thing the model must be told is which task it is doing, and even that is text: a short prefix on the front of the input ("translate English to German: ", "summarize: ", "stsb sentence1: ..."), which is essentially a hyperparameter whose exact wording barely matters. The moment every task looks identical, the head disappears and three things fall out at once: one loss for everything (teacher-forced maximum likelihood, plain cross-entropy over the output token sequence, the same loss for unsupervised pre-training and every downstream task), one decoding procedure (autoregressive generation, greedy at test time), and — this is the real prize — objective, architecture, corpus, and fine-tuning recipe become orthogonal knobs that can each be varied while the other three are held fixed, because none of them touches the interface anymore. The unification is not a trick to win a benchmark; it is what makes a controlled study possible at all.

With that testbed in hand, the pieces get chosen by experiment. The structural choice is really a choice of attention mask over the same self-attention brick: a fully-visible mask lets every position see every other, a causal mask forbids looking right ($j > i$ gets weight zero), and a hybrid is fully-visible over a prefix and causal after. Every text-to-text task gives the model a context and asks for an output that depends on all of it — in translation, the German word I emit can depend on any English word, including ones to its right — so a fully causal encoding of the input is needlessly hobbled, exactly the old complaint about unidirectional RNN encoders. That favors either the encoder-decoder (bidirectional encoder, causal decoder that cross-attends) or the prefix LM (one stack, fully-visible over the input, causal over the target), and rules out the pure decoder-only LM. To compare them fairly I have to confront a subtlety: an $L+L$ encoder-decoder has roughly $2P$ parameters versus a depth-$L$ decoder-only LM's $P$, but the encoder's layers only run over the input and the decoder's only over the output, whereas the decoder-only LM runs all $L$ layers over the concatenation — so the two do nearly identical FLOPs despite the parameter gap (confirmed by near-identical step times). I therefore cannot match on both parameters and compute, so I fix FLOPs $M$ and sweep the parameter budget. The encoder-decoder wins on every task; sharing parameters between encoder and decoder (same compute, half the parameters) ties it; halving the depth to buy back parameters hurts, so depth is what matters and sharing is the cheap way to keep it; and the shared encoder-decoder beats the prefix LM, which says the explicit cross-attention is worth something beyond mere bidirectionality. So: original-style encoder-decoder.

The objective is the heart, because it is the channel through which general knowledge enters. Through the text-to-text interface I compare a prefix-LM objective, a BERT-style denoising objective, and a deshuffling objective; denoising wins, prefix-LM trails except on translation, deshuffling is clearly worst. Then I redesign the denoising objective for a generative decoder rather than inheriting BERT's, which was built for an encoder predicting at masked positions. BERT corrupts $15\%$ of tokens with a mix of $\texttt{[MASK]}$, random, and unchanged substitutions; the random-and-unchanged heuristics exist only to soften the encoder's pretrain/finetune $\texttt{[MASK]}$ mismatch, which is not my problem since my target is generated — dropping them costs nothing, so they go. The naive port reconstructs the entire original sequence, but that makes the decoder spend most of its effort copying the uncorrupted majority, which carries no learning signal and lengthens the target. So I predict only the corrupted tokens, and I do it by replacing each contiguous run of corrupted tokens with a single unique sentinel token — $\langle X\rangle, \langle Y\rangle, \langle Z\rangle$, fresh vocabulary IDs that are no real word piece — making the target the concatenation of the dropped spans, each prefixed by its sentinel and ended with a final sentinel. "Thank you for inviting me to your party last week ." with "for inviting" and "last" corrupted becomes input "Thank you $\langle X\rangle$ me to your party $\langle Y\rangle$ week ." and target "$\langle X\rangle$ for inviting $\langle Y\rangle$ last $\langle Z\rangle$"; because a whole run collapses into one sentinel, both input and target come out short. Sweeping the corruption rate over $10/15/25/50\%$ barely moves quality until $50\%$ degrades it (and lengthens targets), so $15\%$ stays. The decisive refinement is that i.i.d. corruption rarely produces long runs, so the collapse-into-one-sentinel shortening seldom fires; deliberately corrupting contiguous spans makes every span one sentinel and the sequences genuinely shorter, with the side benefit that span-level masking was already known to help. Parametrizing by corruption rate and span count fixes the mean span length, and a sweep of mean lengths $2/3/5/10$ lands on $3$ — slightly better than i.i.d. on most non-translation tasks and faster. The loudest finding is the negative one: the big gap is denoising-versus-LM, and within the denoising family quality is remarkably flat, so the tiebreaker is computational cost — shortest targets — not a magic objective.

A few Transformer internals are re-derived rather than copied. Standard LayerNorm re-centers and re-scales with a learned gain and bias; since the mean-subtraction buys little and the scale normalization does the stabilizing work, I drop both the mean-subtraction and the bias and normalize by root-mean-square only, $x \cdot \big(\mathrm{mean}(x^2)+\epsilon\big)^{-1/2}$ times a learned per-feature scale — cheaper, fewer parameters, just as stable. I place this norm on the input of each sublayer with the residual path left un-normalized (pre-norm), so the identity signal flows clean through and deep stacks train stably; each block is normalize, sublayer, dropout, add residual. Biases are stripped from every dense and attention projection. The feed-forward sublayer projects up to $d_{ff} = 4\,d_{model}$, applies ReLU, and projects back, kept wide because attention only mixes information across positions while this position-wise MLP is where most per-token computation lives. Attention uses $12$ heads of key/value dimension $64$, so the inner dimension is $12 \times 64 = 768 = d_{model}$. The usual $1/\sqrt{d_k}$ pre-softmax scaling exists because a dot product of two $d_k$-dimensional vectors grows like $\sqrt{d_k}$ and would saturate the softmax; rather than divide in the forward pass, I bake the factor into initialization by setting the query-projection init std to scale like $(d_{model}\cdot d_{kv})^{-1/2}$ instead of $d_{model}^{-1/2}$ — that extra $d_{kv}^{-1/2}$ is exactly the $\sqrt{d_k}$ I would have divided by — while keys and values init at $d_{model}^{-1/2}$ and the output projection at $(n_{heads}\cdot d_{kv})^{-1/2}$, so the logits come out correctly scaled and one operation leaves the hot loop.

Position needs explicit injection because self-attention is a set operation. Absolute position embeddings tie everything to absolute indices and generalize badly to unseen lengths, whereas what matters for language is relative offset. I take the relative-position idea to its lightest form: since the position signal ultimately just nudges the attention logit between a query and a key, let it be a single learned scalar per offset, added straight onto the logit, with a different scalar per head. To avoid learning a scalar for every offset, I bucket: $32$ buckets total, half assigned exactly to small offsets where fine resolution matters, the other half growing logarithmically up to a max distance of $128$, with everything at or beyond $128$ collapsing into one catch-all bucket — sharp distinctions nearby, a coarse sense of "far" at distance, and graceful generalization to longer sequences because an unseen huge offset lands in the same far bucket as other huge offsets. For the bidirectional encoder the buckets split by sign (key to the left versus right); for the causal decoder only non-positive offsets survive masking. Any single layer is blind beyond $128$, but stacked layers compose local position information into longer-range sensitivity. These position parameters are shared across all layers — relative relationships are the same kind of thing at every depth, so reuse costs almost nothing (just $32 \times n_{heads}$ scalars per stack) — computed once and threaded into every layer's logits. The remaining choices serve scale: the input embedding, decoder input embedding, and output softmax projection are one tied matrix (saving parameters, acting as a regularizer); the optimizer is Adafactor, which factorizes Adam's per-parameter second moment to fit $11$B parameters in memory; the learning rate follows an inverse-square-root schedule (constant $0.01$ through $10^4$ warmup steps, then $1/\sqrt{\text{step}}$ decay) so it needs no knowledge of the total step count, which varies across experiments; the vocabulary is $32$k SentencePiece word-pieces trained on a $10{:}1{:}1{:}1$ English/German/French/Romanian mix so the fixed vocabulary can emit the translation targets, plus the sentinel IDs. And the data is C4, the Colossal Clean Crawled Corpus: Common Crawl filtered by blunt heuristics that each kill a failure mode — keep only lines ending in terminal punctuation, drop short pages and lines, drop pages with bad words, remove "JavaScript"/"lorem ipsum"/curly-brace (a cheap high-precision code detector) lines, strip citation markers and boilerplate policy lines, deduplicate repeated three-sentence spans, and keep only langdetect-English $\geq 0.99$ — yielding roughly $750$ GB of clean English, pre-trained on about $2^{35}\approx 34$ billion tokens with no data repeated, since a corpus small enough to be repeated many times degrades downstream performance.

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
