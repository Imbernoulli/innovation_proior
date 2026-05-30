# RoBERTa: a robustly optimized BERT pretraining recipe

## Problem

Post-BERT pretraining methods each report a new objective or architecture and a
new state of the art, but every comparison changes several things at once —
objective, data size, batch size, training length — so it is unclear whether the
gains come from the modeling idea or from an under-tuned BERT baseline being
compared against well-resourced successors. RoBERTa isolates the training recipe:
holding BERT's architecture and masked-language-model objective fixed, how good
is the model when the recipe is tuned carefully?

## Key idea

BERT was significantly undertrained. With the architecture and MLM objective
unchanged, a handful of recipe changes make masked-LM pretraining competitive
with (or better than) every method published after BERT:

1. **Dynamic masking** — generate the masking pattern freshly each time a
   sequence is fed, instead of a single static mask fixed at preprocessing time.
2. **Drop next-sentence prediction; use FULL-SENTENCES input** — pack contiguous
   full sentences up to 512 tokens (crossing document boundaries with an extra
   separator), train with only the masked-token loss. BERT's "NSP helps" result
   conflated removing the loss with shortening the input; with long contiguous
   input, NSP is redundant.
3. **Large batches** — train with 8K sequences per batch (with a scaled-up
   learning rate), which improves both MLM perplexity and end-task accuracy.
   Stabilize large-batch Adam by setting **β₂ = 0.98** (not 0.999) and turning
   gradient clipping off.
4. **Byte-level BPE, 50K units** — a universal tokenizer that encodes any text
   with no unknown tokens and no language-specific preprocessing.
5. **More data, longer training** — scale from 16GB to 160GB across five corpora
   (BookCorpus+Wikipedia, CC-News, OpenWebText, Stories) and from 100K to 500K
   steps, with the architecture frozen.

## Configuration

Architecture is identical to BERT. RoBERTa-large: 24 layers, hidden 1024, FFN
inner 4096, 16 heads, head size 64, 355M params (+~20M from the larger vocab).
RoBERTa-base: 12 layers, hidden 768, FFN inner 3072, 12 heads.

Pretraining: dropout 0.1, attention dropout 0.1; Adam β₁=0.9, β₂=0.98, ε=1e-6,
weight decay 0.01; peak LR 4e-4 (large) / 6e-4 (base) with linear warmup (30K /
24K steps) then linear decay; batch 8K; up to 500K steps; gradient clipping 0.0;
mixed precision. Masking: 15% of tokens selected; 80%→[MASK], 10% unchanged,
10% random, dynamic per feed.

Finetuning (large): GLUE — LR ∈ {1e-5, 2e-5, 3e-5}, batch ∈ {16, 32}, weight
decay 0.1, 10 epochs, warmup ratio 0.06, median over 5 seeds. SQuAD — LR 1.5e-5,
batch 48, weight decay 0.01, 2 epochs (v2.0 adds a binary answerability classifier
summed with the span loss). RACE — LR 1e-5, batch 16, weight decay 0.1, 4 epochs;
concatenate each candidate answer with question+passage, encode four sequences,
classify from the first-token representations.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- tokenizer: byte-level BPE, ~50K units, no preprocessing, no UNK ----
class ByteLevelBPE:
    def __init__(self, encoder_json, vocab_bpe):
        self.bpe = load_byte_bpe(encoder_json, vocab_bpe)
    def encode(self, text):
        return self.bpe.encode(text)

# ---- FULL-SENTENCES packing (no segment pairs, no NSP) ----
def build_full_sentences(documents, max_len=512, sep_id=2):
    seqs, buf = [], []
    for doc in documents:
        for sent in doc:
            if len(buf) + len(sent) > max_len:
                seqs.append(buf); buf = []
            buf += sent
        buf += [sep_id]                       # separator between documents
    if buf: seqs.append(buf)
    return seqs

# ---- dynamic masking: fresh each feed; 80/10/10 split over 15% ----
def dynamic_mask(tokens, mask_id, vocab_size, p=0.15):
    out, labels = tokens.clone(), torch.full_like(tokens, -100)
    sel = torch.rand_like(tokens, dtype=torch.float) < p
    labels[sel] = tokens[sel]
    r = torch.rand_like(tokens, dtype=torch.float)
    out[sel & (r < 0.8)] = mask_id                                  # 80% -> [MASK]
    rand = sel & (r >= 0.9)
    out[rand] = torch.randint(vocab_size, (int(rand.sum()),), device=tokens.device)  # 10% random
    return out, labels                                              # 10% unchanged

# ---- masked-LM head ----
class MLMHead(nn.Module):
    def __init__(self, hidden, vocab_size, embed_weight):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.weight = embed_weight                       # tied to input embeddings
        self.bias = nn.Parameter(torch.zeros(vocab_size))
    def forward(self, x):
        x = self.layer_norm(F.gelu(self.dense(x)))
        return x @ self.weight.t() + self.bias

def mlm_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                           labels.view(-1), ignore_index=-100)

# ---- finetuning head over the first token ----
class ClassificationHead(nn.Module):
    def __init__(self, hidden, inner, n_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, inner)
        self.out_proj = nn.Linear(inner, n_classes)
    def forward(self, features):
        x = self.dropout(features[:, 0, :])              # <s> representation
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))

# ---- large-batch-stabilized Adam ----
def make_optimizer(params, peak_lr=4e-4):
    return torch.optim.Adam(params, lr=peak_lr, betas=(0.9, 0.98),
                            eps=1e-6, weight_decay=0.01)  # beta2=0.98; clipping off
```

The model body is a standard bidirectional Transformer encoder (same as BERT); the
RobertaLMHead reprojects masked positions to the vocabulary with a tied embedding,
and a small classification head over the first token handles finetuning.
