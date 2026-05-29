# BERT: pre-training deep bidirectional Transformers

## The problem

Pre-training on unlabeled text and transferring is the dominant way to overcome
label scarcity in NLP, but the standard pre-training objective — a language model
that predicts the next token — is intrinsically unidirectional. The
representations it yields encode left context only, which is sub-optimal for
sentence-level tasks and outright wrong for token-level tasks like span-extraction
question answering, where the disambiguating context lies on both sides of a
token. BERT (Bidirectional Encoder Representations from Transformers) pre-trains a
deep Transformer encoder whose every layer is jointly conditioned on both left and
right context, then fine-tunes the whole network on each downstream task with a
single added output layer.

## The key idea

A standard left-to-right LM is welded to unidirectionality: in a multi-layer
network with unrestricted attention, predicting token *i* from a representation
that may attend to position *i* lets the token "see itself" directly, or through
neighboring states in a deeper stack, so the objective collapses. BERT breaks the
coupling by changing the *objective*, not the architecture:

- **Masked LM (MLM).** Select a random 15% of input positions as prediction
  targets and predict their original token IDs from the full bidirectional
  context. Most selected targets are removed from the input, so full
  bidirectional attention no longer makes token prediction trivial. Loss is
  computed only on the selected positions.
- **80/10/10 masking.** The `[MASK]` symbol never appears at fine-tuning time, so
  to avoid a train/test mismatch, of the chosen 15% of positions: 80% are
  replaced with `[MASK]`, 10% with a random token, 10% left unchanged (still
  predicted). Some queried positions now look like ordinary input tokens, so
  `[MASK]` is not a complete cue for where useful representations are needed.
- **Next Sentence Prediction (NSP).** A binary task — is segment B the true
  continuation of segment A (IsNext) or a random segment (NotNext)? — read off a
  pooled `[CLS]` vector. It injects the inter-sentence signal that pure token
  modeling misses but QA/NLI/paraphrase need.
- **Unified input.** `[CLS] A [SEP] B [SEP]`, with each token embedding = token +
  segment(A/B) + learned-position embedding. Self-attention over the packed pair
  is bidirectional cross-attention. Single-sentence tasks use a degenerate
  text-∅ pair. Fine-tuning = swap inputs/head, tune all parameters end-to-end.

## The final objective and architecture

- **Architecture:** a multi-layer bidirectional Transformer encoder (Vaswani et
  al. 2017) with no causal mask. BASE: L=12, H=768, A=12, 110M params (matched to
  OpenAI GPT for comparison). LARGE: L=24, H=1024, A=16, 340M. FFN width 4H, GELU
  activation, WordPiece vocabulary of 30k.
- **Pre-training loss:** `L_MLM = - |M|^{-1} Σ_{i∈M} log p(x_i | corrupted input)`;
  `L_NSP = - N^{-1} Σ_n log p(y_n | C_n)`; `L = L_MLM + L_NSP`, implemented as
  mean cross-entropy over the MLM target set plus mean binary softmax
  cross-entropy for NSP.
  Corpus: BooksCorpus + English Wikipedia (document-level, for long spans).
  Adam, lr 1e-4, β=(0.9,0.999), weight decay 0.01, warmup 10k + linear decay,
  dropout 0.1, batch 128k tokens, 1M steps; 90% of steps at length 128, last 10%
  at 512.
- **Fine-tuning:** initialize from the pre-trained weights, add one head.
  Classification: softmax(C·Wᵀ) on the `[CLS]` vector C. SQuAD span: learned
  start/end vectors S, E; `P_start(i)=softmax_i(S·Tᵢ)` and
  `P_end(j)=softmax_j(E·T_j)`; span score S·Tᵢ + E·T_j over j ≥ i; training
  minimizes `-log P_start(i*) - log P_end(j*)`. SQuAD 2.0:
  no-answer span at `[CLS]`; with margin τ, predict non-null iff the best
  non-null span score exceeds the null score by τ. Equivalently, using a
  null-score-difference threshold δ, predict null iff `s_null - s_best > δ`.

## Working code

Grounded in the canonical `google-research/bert` implementation (`modeling.py`,
`run_pretraining.py`), expressed in PyTorch.

```python
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB, MAXLEN, H, L, A = 30000, 512, 768, 12, 12
FFN = 4 * H

class EncoderLayer(nn.Module):
    """Standard Transformer encoder block — bidirectional (no causal mask)."""
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(H, A, dropout=0.1, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, FFN), nn.GELU(), nn.Linear(FFN, H))
        self.drop = nn.Dropout(0.1)
    def forward(self, x, key_padding_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)  # both directions
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))

class InputEmbedding(nn.Module):
    """token + segment(A/B) + learned position, summed, LayerNorm + dropout."""
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, H, padding_idx=0)
        self.seg = nn.Embedding(2, H)
        self.pos = nn.Embedding(MAXLEN, H)
        self.ln, self.drop = nn.LayerNorm(H), nn.Dropout(0.1)
    def forward(self, ids, seg_ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.drop(self.ln(self.tok(ids) + self.seg(seg_ids) + self.pos(pos)))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = InputEmbedding()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(L))
        self.pooler = nn.Linear(H, H)
    def forward(self, ids, seg_ids, pad_mask):
        x = self.embed(ids, seg_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        pooled = torch.tanh(self.pooler(x[:, 0]))     # [CLS] aggregate representation
        return x, pooled

class MaskedLMHead(nn.Module):
    """transform (dense+GELU+LN) then project through tied input embeddings."""
    def __init__(self, tok_embed):
        super().__init__()
        self.transform = nn.Linear(H, H)
        self.act, self.ln = nn.GELU(), nn.LayerNorm(H)
        self.decoder = nn.Linear(H, VOCAB, bias=True)
        self.decoder.weight = tok_embed.weight        # weight tying
    def forward(self, seq):
        return self.decoder(self.ln(self.act(self.transform(seq))))

class NextSentenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(H, 2)                     # 0 = IsNext, 1 = NotNext
    def forward(self, pooled):
        return self.cls(pooled)

class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mlm = MaskedLMHead(self.encoder.embed.tok)
        self.nsp = NextSentenceHead()
    def forward(self, ids, seg_ids, pad_mask):
        seq, pooled = self.encoder(ids, seg_ids, pad_mask)
        return self.mlm(seq), self.nsp(pooled)

def pretrain_loss(model, ids, seg_ids, pad_mask, mlm_labels, nsp_labels):
    mlm_logits, nsp_logits = model(ids, seg_ids, pad_mask)
    mlm = F.cross_entropy(mlm_logits.reshape(-1, VOCAB), mlm_labels.reshape(-1),
                          ignore_index=-100)          # only the 15% selected positions
    nsp = F.cross_entropy(nsp_logits, nsp_labels.reshape(-1))
    return mlm + nsp                                   # joint objective

# --- data: 15% selected MLM targets with 80/10/10, plus the 50/50 next-sentence draw ---
MASK_ID, MASK_PROB, CLS, SEP = 103, 0.15, 101, 102
def make_example(span_a, span_b_true, random_span, vocab_size):
    if torch.rand(1).item() < 0.5:
        span_b, nsp = span_b_true, 0
    else:
        span_b, nsp = random_span, 1
    ids = [CLS] + span_a + [SEP] + span_b + [SEP]
    seg = [0]*(len(span_a)+2) + [1]*(len(span_b)+1)
    labels = [-100]*len(ids)
    candidates = [i for i, tok in enumerate(ids) if tok not in (CLS, SEP)]
    num_to_predict = min(len(candidates), max(1, int(round(len(ids) * MASK_PROB))))
    for j in torch.randperm(len(candidates)).tolist()[:num_to_predict]:
        i = candidates[j]
        labels[i] = ids[i]
        r = torch.rand(1).item()
        if   r < 0.8: ids[i] = MASK_ID
        elif r < 0.9: ids[i] = torch.randint(0, vocab_size, (1,)).item()
        # else 10%: keep the original token, still predicted
    return ids, seg, labels, nsp

# --- fine-tuning: reuse the encoder, attach one task head, tune everything ---
class ForClassification(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(H, num_labels)
    def forward(self, ids, seg_ids, pad_mask):
        _, pooled = self.encoder(ids, seg_ids, pad_mask)
        return self.head(pooled)                        # softmax(C W^T)

class ForSpanQA(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.qa = nn.Linear(H, 2)                       # start vector S and end vector E
    def forward(self, ids, seg_ids, pad_mask):
        seq, _ = self.encoder(ids, seg_ids, pad_mask)
        start, end = self.qa(seq).split(1, dim=-1)      # S·T_i and E·T_i per token
        return start.squeeze(-1), end.squeeze(-1)

def span_qa_loss(start_logits, end_logits, start_positions, end_positions):
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    return start_loss + end_loss                        # -log P_start - log P_end

def squad2_predict(start_logits, end_logits, max_answer_len=30,
                   null_score_diff_threshold=0.0):
    # batch size 1: score(i,j) = S·T_i + E·T_j, with j >= i.
    best = (float("-inf"), 0, 0)
    for i in range(1, start_logits.size(1)):
        max_j = min(start_logits.size(1), i + max_answer_len)
        for j in range(i, max_j):
            score = (start_logits[0, i] + end_logits[0, j]).item()
            if score > best[0]:
                best = (score, i, j)
    null_score = (start_logits[0, 0] + end_logits[0, 0]).item()  # [CLS]
    score_diff = null_score - best[0]
    return None if score_diff > null_score_diff_threshold else (best[1], best[2])
```
