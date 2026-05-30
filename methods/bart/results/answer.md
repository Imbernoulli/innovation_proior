# BART: denoising sequence-to-sequence pretraining

## Problem

Masked language models build strong bidirectional representations for
understanding but cannot generate autoregressively; left-to-right language models
generate fluently but condition only on left context, weakening understanding.
The deeper limitation of masked models is that they corrupt text *in place*, so
the corruption must preserve length and position — ruling out deletion, reordering,
and length-changing noise. BART seeks one pretraining objective strong across
classification, span extraction, summarization, dialogue, abstractive QA, and
machine translation.

## Key idea

Pretrain a standard sequence-to-sequence Transformer — a bidirectional encoder
plus a left-to-right autoregressive decoder with cross-attention — as a
**denoising autoencoder**: corrupt a document with an arbitrary noising function,
feed it to the encoder, and train the decoder to reconstruct the *original*
document under token-level cross-entropy. Decoupling the encoder input from the
decoder output frees the corruption to be anything (it need not preserve length or
order). The bidirectional encoder preserves BERT-style understanding; the
autoregressive decoder gives GPT-style generation, and reconstruction is itself
generation. A language model is the special case where the corruption destroys all
source information.

**Noising functions** (composable): token masking; token deletion (model must
locate gaps); **text infilling** — replace whole spans, with span length ~
Poisson(λ=3) including length 0, by a *single* mask token, so the model must
predict how many tokens are missing; sentence permutation; document rotation. The
large model uses text infilling (masking 30% of tokens) plus permuting all
sentences.

**Architecture.** Standard seq2seq Transformer with GeLU activations and N(0,0.02)
init. Base 6+6 layers, large 12+12, hidden 1024. Versus a masked encoder: each
decoder layer cross-attends to the encoder; the extra pre-prediction feed-forward
network is dropped (the decoder already builds rich features). ~10% more
parameters than a same-sized masked model.

**Finetuning.** Classification: feed the same input to encoder and decoder; append
a class token at the *end* and read its decoder hidden state (a causal decoder's
last position has attended to everything). Token tasks (SQuAD): per-token top
decoder states with start/end classifiers. Generation: encoder reads input,
decoder generates output autoregressively (label-smoothed cross-entropy 0.1, beam
5 with length tuning). Machine translation: keep all of BART as a fixed target-side
English denoiser and replace its source embedding with a small new randomly-init
source encoder (own foreign vocabulary) that maps foreign tokens to a
BART-denoisable representation; train in two steps — first freeze most of BART and
update only the new encoder, BART's positional embeddings, and the first encoder
layer's self-attention input projection, then briefly unfreeze all.

## Configuration

Large pretraining: batch 8000, 500K steps, GPT-2 byte-level BPE, 160GB
news/books/stories/web; dropout disabled for the final 10% of steps.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- noising functions (encoder/decoder decoupled => arbitrary corruption) ----
def token_masking(tokens, mask_id, p=0.15):
    out = tokens.clone()
    sel = torch.rand_like(tokens, dtype=torch.float) < p
    out[sel] = mask_id
    return out

def token_deletion(tokens, p=0.15):
    return tokens[torch.rand_like(tokens, dtype=torch.float) >= p]

def text_infilling(tokens, mask_id, lam=3.0, frac=0.30):
    out, i, masked, target = [], 0, 0, int(frac*len(tokens))
    while i < len(tokens):
        if masked < target and torch.rand(1) < 0.2:
            span = int(torch.poisson(torch.tensor(lam)))   # 0 => insertion
            out.append(mask_id); masked += span; i += span # one mask hides whole span
        else:
            out.append(tokens[i]); i += 1
    return torch.tensor(out)

def sentence_permutation(sentences):
    return [sentences[k] for k in torch.randperm(len(sentences))]

def document_rotation(tokens):
    k = int(torch.randint(len(tokens), (1,)))
    return torch.cat([tokens[k:], tokens[:k]])

# ---- pretraining: reconstruct the ORIGINAL document ----
def pretraining_loss(model, document, mask_id):
    noised = flatten(sentence_permutation(split_sentences(document)))
    noised = text_infilling(noised, mask_id)
    logits = model(src_tokens=noised, prev_output_tokens=shift_right(document))
    return F.cross_entropy(logits.view(-1, logits.size(-1)), document.view(-1))

# ---- classification readout at the END of the (causal) decoder ----
class ClassificationHead(nn.Module):
    def __init__(self, d_model, inner, n_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, n_classes)
    def forward(self, dec_out, eos_mask):
        sent = dec_out[eos_mask, :].view(dec_out.size(0), -1, dec_out.size(-1))[:, -1, :]
        x = torch.tanh(self.dense(self.dropout(sent)))
        return self.out_proj(self.dropout(x))

# ---- MT: swap source embedding for a new foreign encoder ----
class MTSourceEncoder(nn.Module):
    def __init__(self, src_vocab, d_model):
        super().__init__()
        self.embed = nn.Embedding(src_vocab, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=16, batch_first=True), 6)
    def forward(self, src):
        return self.layers(self.embed(src))
```

BART is a seq2seq Transformer with BERT-style init and a classification head that
selects the last EOS position of the decoder output
(`x[src_tokens.eq(eos)] ... [:, -1, :]`) before a dense→tanh→projection.
