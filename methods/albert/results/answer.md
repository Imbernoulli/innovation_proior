# ALBERT: A Lite BERT for self-supervised language representation learning

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
import torch, torch.nn as nn, torch.nn.functional as F

# ---- factorized embedding: lookup in E, project E -> H ----
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden):          # embed_dim E << hidden H
        super().__init__()
        self.word = nn.Embedding(vocab_size, embed_dim)         # V x E
        self.proj = nn.Linear(embed_dim, hidden)                # E x H
    def forward(self, ids):
        return self.proj(self.word(ids))

# ---- one encoder block ----
class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden); self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Linear(4*hidden, hidden))
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

# ---- cross-layer sharing: one block applied n_layers times ----
class SharedEncoder(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.block = EncoderBlock(hidden, n_heads, dropout)     # the ONLY block
    def forward(self, x, mask=None):
        for _ in range(self.n_layers):
            x = self.block(x, mask)                             # same weights reused
        return x

# ---- sentence-order prediction ----
def sop_example(seg_a, seg_b):                                  # two consecutive segments, same doc
    if torch.rand(1) < 0.5:
        return (seg_a, seg_b), 1                                # correct order
    return (seg_b, seg_a), 0                                    # swapped

class SOPHead(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.cls = nn.Linear(hidden, 2)
    def forward(self, features):
        return self.cls(features[:, 0, :])                      # over [CLS]

# ---- n-gram masking: p(n) proportional to 1/n, n in 1..3 ----
def sample_ngram_length(max_n=3):
    w = torch.tensor([1.0/n for n in range(1, max_n+1)])
    return 1 + int(torch.multinomial(w / w.sum(), 1))

# ---- MLM head (projects back toward embedding space) ----
class MLMHead(nn.Module):
    def __init__(self, hidden, embed_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        return self.decoder(self.ln(F.gelu(self.dense(x))))

def total_loss(mlm_logits, mlm_labels, sop_logits, sop_labels):
    return (F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)),
                            mlm_labels.view(-1), ignore_index=-100)
            + F.cross_entropy(sop_logits, sop_labels))          # MLM + SOP
```

In TensorFlow the factorization is an `embedding_hidden_mapping_in` dense
projection from E to H, and the sharing is a single `transformer` variable scope
with `reuse=tf.AUTO_REUSE` so all layers read the same variables.
