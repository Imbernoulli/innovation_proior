Bigger bidirectional Transformer encoders have consistently improved language understanding, but simply widening a large BERT under a fixed optimization budget makes it worse, not better, and the model does not overfit — it merely underfits and becomes harder to train. At the same time, the parameter count is hitting memory limits and inflating distributed communication, so adding capacity the naive way is blocked by both trainability and hardware walls. The real question is therefore how to reorganize the parameters so that a model with far fewer weights can still scale wider and train more stably.

The main parameter pools are easy to identify. The token embedding matrix is conventionally tied to the hidden size, so widening the context-dependent hidden representation forces the entire vocabulary lookup to widen too, even though most rows are updated sparsely. Meanwhile, the per-layer Transformer weights are replicated independently across every layer, so depth multiplies the same attention and feed-forward blocks over and over again. Both conventions waste parameters without adding modeling power where it matters.

The method I propose is ALBERT, which stands for A Lite BERT. It makes three core changes to the standard Transformer-encoder pretraining recipe. First, it factorizes the embedding parameterization by untying the embedding dimension E from the hidden size H. Tokens are first projected into a small E-dimensional lookup space and then mapped up to H with a learned projection, cutting the embedding parameters from O(V times H) to O(V times E plus E times H). Since the lookup is context-independent and the hidden states are context-dependent, there is no reason they must share the same width; in practice E equals 128 is enough and performance does not improve with larger E. Second, ALBERT shares all Transformer-block weights across layers, applying a single attention and feed-forward block repeatedly rather than maintaining L independent copies. This makes the per-layer parameter count independent of depth, and measurements of layer-to-layer distances show the shared trajectory oscillates and smooths rather than collapsing to a degenerate fixed point, so the sharing acts as a useful stabilizer. Third, ALBERT replaces the standard next-sentence prediction objective with sentence-order prediction. The original NSP negative examples draw two segments from different documents, which makes the task solvable by topic detection alone and therefore redundant with masked language modeling; SOP instead keeps two consecutive segments from the same document and swaps their order for negatives, so topic is held constant and the model must learn real coherence cues.

With these changes, an ALBERT model configured similarly to BERT-large uses roughly eighteen times fewer parameters, yet it can be scaled to hidden sizes of 2048 or even 4096 while still remaining smaller than the original large model. Once the block is shared, depth beyond about twelve layers yields little additional gain, so the widest configuration uses only twelve layers. For the largest models, dropout is removed because the model is underfitting rather than overfitting. Training uses the LAMB optimizer with a large batch, an initial learning rate around 0.00176, span-based n-gram masking for the MLM target with length sampled proportional to one over n up to three, and the new sentence-order objective alongside masked language modeling. The code below implements the embedding stem, the shared encoder stack, the sentence-order head, the masked-LM head tied to the input embeddings, and the combined loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingStem(nn.Module):
    def __init__(self, vocab_size, hidden, embedding_width=128,
                 max_positions=512, type_vocab_size=2, dropout=0.0):
        super().__init__()
        self.word = nn.Embedding(vocab_size, embedding_width)
        self.position = nn.Embedding(max_positions, embedding_width)
        self.token_type = nn.Embedding(type_vocab_size, embedding_width)
        self.norm = nn.LayerNorm(embedding_width)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embedding_width, hidden) if embedding_width != hidden else nn.Identity()

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = self.word(input_ids) + self.position(positions) + self.token_type(token_type_ids)
        return self.proj(self.drop(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

    def forward(self, x, padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))


class EncoderStack(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.block = EncoderBlock(hidden, n_heads, dropout)

    def forward(self, x, padding_mask=None):
        for _ in range(self.n_layers):
            x = self.block(x, padding_mask)
        return x


class SentencePairHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, sequence_output):
        pooled = torch.tanh(self.pooler(sequence_output[:, 0, :]))
        return self.classifier(pooled)


def inter_sentence_examples(seg_a, seg_b):
    if torch.rand(()) < 0.5:
        return (seg_a, seg_b), 0
    return (seg_b, seg_a), 1


def sample_ngram_length(max_n=3):
    w = torch.tensor([1.0 / n for n in range(1, max_n + 1)])
    return 1 + int(torch.multinomial(w / w.sum(), 1))


def gather_positions(sequence_output, positions):
    batch, seq_len, width = sequence_output.shape
    offsets = torch.arange(batch, device=sequence_output.device).unsqueeze(1) * seq_len
    flat_positions = (positions + offsets).reshape(-1)
    return sequence_output.reshape(batch * seq_len, width).index_select(0, flat_positions)


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
    mlm_each = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), flat_labels, reduction="none")
    mlm_loss = (mlm_each * flat_weights).sum() / (flat_weights.sum() + 1e-5)
    pair_loss = F.cross_entropy(pair_logits, pair_labels.reshape(-1))
    return mlm_loss + pair_loss
```
