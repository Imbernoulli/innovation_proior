# ALBERT

## Problem

Bigger Transformer encoders generally improve language understanding, but scaling
runs into two walls: hardware memory and a communication cost proportional to the
parameter count, and — under a fixed optimization budget — simply widening a large
BERT *degrades* downstream accuracy with no overfitting (it becomes harder to
train, not over-regularized). ALBERT reorganizes the parameters so a model with
far fewer parameters than large-BERT trains more stably and can be scaled much
wider.

## Key idea

Three changes to BERT, holding the Transformer-encoder backbone (GELU, FFN inner
size 4H, A = H/64 heads):

1. **Factorized embedding parameterization.** Untie the embedding dimension E from
   the hidden size H. The token embedding is a context-*independent* lookup and
   needn't be as wide as the context-*dependent* hidden states; want H ≫ E. Project
   one-hot tokens into a small E-dim space, then up to H. Parameters drop from
   O(V·H) to O(V·E + E·H). Fix **E = 128**.

2. **Cross-layer parameter sharing (share all).** Use one Transformer block applied
   L times instead of L independent blocks, so the parameter count is independent
   of depth. Measured input/output embedding distances *oscillate* (don't collapse
   to a fixed point), and sharing smooths layer-to-layer transitions, acting as a
   stabilizer. Once shared, depth past ~12 gives no gain.

3. **Sentence-order prediction (SOP) replacing NSP.** NSP's negatives (segments
   from different documents) are separable by topic alone — redundant with MLM and
   too easy. SOP keeps two *consecutive* segments from the same document; the
   positive is true order, the negative is the same two segments *swapped*. Topic
   is identical on both sides, so only discourse coherence distinguishes them.

Together these give ~18× fewer parameters than large-BERT at comparable
configuration, letting H scale to 2048/4096 while still having fewer parameters
than the original. For the large runs, dropout is removed (the model underfits).

## Configuration

Backbone Transformer encoder, GELU, FFN inner 4H, heads = H/64, vocabulary 30K
(SentencePiece), E = 128. Configs: base L12 H768 (12M params); large L24 H1024
(18M); xlarge L24 H2048 (60M); xxlarge L12 H4096 (235M). MLM target uses n-gram
masking, length n sampled with p(n) ∝ 1/n capped at n=3. Pretraining: LAMB
optimizer, batch 4096, learning rate 0.00176, 125K steps (longest runs to
1M–1.5M), max sequence length 512, dropout removed for the largest runs.
Downstream: per-task learning rates 1e-5 to 5e-5, batch 16–128, classifier
dropout 0.1; SQuAD v2.0 adds a jointly-trained answerability classifier.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- embedding stem: lookup in E, add type/position in E, project E -> H ----
class EmbeddingStem(nn.Module):
    def __init__(self, vocab_size, hidden, embedding_width=128,
                 max_positions=512, type_vocab_size=2, dropout=0.0):
        super().__init__()
        self.word = nn.Embedding(vocab_size, embedding_width)
        self.position = nn.Embedding(max_positions, embedding_width)
        self.token_type = nn.Embedding(type_vocab_size, embedding_width)
        self.norm = nn.LayerNorm(embedding_width)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(embedding_width, hidden)
                     if embedding_width != hidden else nn.Identity())

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        x = self.word(input_ids) + self.position(positions) + self.token_type(token_type_ids)
        return self.proj(self.drop(self.norm(x)))

# ---- one encoder block ----
class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden); self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Linear(4*hidden, hidden))

    def forward(self, x, padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

# ---- cross-layer sharing: one block applied n_layers times ----
class EncoderStack(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.block = EncoderBlock(hidden, n_heads, dropout)

    def forward(self, x, padding_mask=None):
        for _ in range(self.n_layers):
            x = self.block(x, padding_mask)
        return x

# ---- sentence-order prediction ----
def inter_sentence_examples(seg_a, seg_b):                      # consecutive segments, same doc
    if torch.rand(()) < 0.5:
        return (seg_a, seg_b), 0                                # correct order
    return (seg_b, seg_a), 1                                    # swapped

class SentencePairHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, sequence_output):
        pooled = torch.tanh(self.pooler(sequence_output[:, 0, :]))
        return self.classifier(pooled)

# ---- n-gram masking: p(n) proportional to 1/n, n in 1..3 ----
def sample_ngram_length(max_n=3):
    w = torch.tensor([1.0/n for n in range(1, max_n+1)])
    return 1 + int(torch.multinomial(w / w.sum(), 1))

def gather_positions(sequence_output, positions):
    batch, seq_len, width = sequence_output.shape
    offsets = torch.arange(batch, device=sequence_output.device).unsqueeze(1) * seq_len
    flat_positions = (positions + offsets).reshape(-1)
    return sequence_output.reshape(batch * seq_len, width).index_select(0, flat_positions)

# ---- MLM head: project H -> E, then tie logits to the input embedding table ----
class MLMHead(nn.Module):
    def __init__(self, hidden, embedding_width, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden, embedding_width)
        self.ln = nn.LayerNorm(embedding_width)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, sequence_output, positions, embedding_table):
        x = gather_positions(sequence_output, positions)
        x = self.ln(F.gelu(self.dense(x)))
        return x @ embedding_table.weight.t() + self.bias

def total_loss(mlm_logits, mlm_labels, mlm_weights, pair_logits, pair_labels):
    flat_labels = mlm_labels.reshape(-1)
    flat_weights = mlm_weights.reshape(-1).float()
    mlm_each = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)),
                               flat_labels, reduction="none")
    mlm_loss = (mlm_each * flat_weights).sum() / (flat_weights.sum() + 1e-5)
    pair_loss = F.cross_entropy(pair_logits, pair_labels.reshape(-1))
    return mlm_loss + pair_loss
```

In TensorFlow the factorization is an `embedding_hidden_mapping_in` dense
projection from E to H, and the sharing is a single `transformer` variable scope
with `reuse=tf.AUTO_REUSE` so all layers read the same variables. The masked-LM
head gathers only the masked positions, projects H back to E, and reuses the
input word-embedding table as the output weights with a separate vocabulary bias.
