The central problem is how to learn a general-purpose language representation from unlabeled text when labeled data for specific tasks is scarce. Pre-training has become the standard answer: train a model on raw text and transfer it. The dominant pre-training signal is a left-to-right language model, which predicts the next token from everything before it. That objective works, but it imposes a directional cost. A token's hidden state can only depend on preceding tokens, so the representation encodes left context only. For sentence-level tasks this is merely suboptimal, but for token-level tasks like extractive question answering it is a serious limitation: the words that disambiguate a candidate answer often lie on both sides of it, and a unidirectional representation has literally never looked right. ELMo offers a partial fix by training separate forward and backward language models and concatenating their hidden states, but no single unit at any internal layer is jointly conditioned on both directions; the two streams only meet at the top. What is missing is a representation that is deeply bidirectional at every layer and can be fine-tuned end-to-end with minimal task-specific architecture.

The method I propose is BERT, short for Bidirectional Encoder Representations from Transformers. The insight is that the obstacle is not the Transformer architecture itself, which is naturally all-to-all, but the next-token objective that is welded to unidirectionality. If I drop the causal mask and let every position attend to every other position, and then train the model to predict a token from its own hidden state, the objective collapses because the target token can leak into its own representation through direct or indirect paths. The fix is to change the objective so that the prediction target is never present in the context used to predict it. BERT therefore pre-trains a deep Transformer encoder with a masked language model objective: randomly select 15% of input positions, replace most of them with a special [MASK] symbol, and train the model to recover the original tokens from the full bidirectional context. Because the target token is absent from the input, the model is forced to use surrounding context on both sides, and full bidirectionality becomes safe rather than degenerate.

To make this practical, the chosen 15% of positions are handled with an 80-10-10 split. Eighty percent are replaced with [MASK], ten percent are replaced with a random token, and ten percent are left unchanged but still predicted. This is important because [MASK] never appears during fine-tuning; if every prediction target were masked, the model could learn to treat [MASK] as a special cue and would underperform on ordinary text. By occasionally asking the model to predict an unchanged or randomly substituted token, the representation is pushed to be useful for normal-looking inputs too. The model only computes the cross-entropy loss on the selected prediction positions, not on the entire sequence, so it does not waste capacity re-copying visible tokens.

BERT also adds a second pre-training task called next sentence prediction. Many downstream tasks require reasoning about the relationship between two sentences, such as entailment or paraphrase detection, but a masked language model only ever models a single token stream. The next-sentence task takes two text segments A and B, and half the time B is the real sentence that follows A, while half the time it is a random sentence from elsewhere in the corpus. The model must classify whether B is the true continuation. The two segments are packed into a single sequence in the format [CLS] A [SEP] B [SEP]. A special classification token [CLS] is prepended, and its final hidden state serves as the aggregate sequence representation for the binary classifier. Separator tokens mark segment boundaries, and a learned segment embedding tells the model whether each token belongs to A or B. The input representation at each position is therefore the sum of token, segment, and learned position embeddings. This unified format handles both pair tasks and single-sentence tasks with the same architecture.

The architecture itself is a standard Transformer encoder with no causal mask. It uses multi-head self-attention with 1/sqrt(d_k) scaling, feed-forward layers with inner dimension four times the hidden size, residual connections, layer normalization, and GELU activations. The masked language model head applies a small transform of dense layer, GELU, and layer normalization before projecting back to the vocabulary. The output projection shares its weights with the input token embedding matrix, saving parameters and regularizing the model. Pre-training optimizes the sum of the masked language model loss and the next sentence prediction loss using Adam with warmup and linear decay. Fine-tuning is straightforward: initialize the pre-trained encoder, attach a small task-specific head, and update all parameters end-to-end.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, MAXLEN, H, L, A = 30000, 512, 768, 12, 12
FFN = 4 * H

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(H, A, dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, FFN), nn.GELU(), nn.Linear(FFN, H))
        self.drop = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))

class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, H, padding_idx=0)
        self.seg = nn.Embedding(2, H)
        self.pos = nn.Embedding(MAXLEN, H)
        self.ln = nn.LayerNorm(H)
        self.drop = nn.Dropout(0.1)

    def forward(self, ids, seg_ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        e = self.tok(ids) + self.seg(seg_ids) + self.pos(pos)
        return self.drop(self.ln(e))

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
        pooled = torch.tanh(self.pooler(x[:, 0]))
        return x, pooled

class MaskedLMHead(nn.Module):
    def __init__(self, tok_embed):
        super().__init__()
        self.transform = nn.Linear(H, H)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(H)
        self.decoder = nn.Linear(H, VOCAB, bias=True)
        self.decoder.weight = tok_embed.weight

    def forward(self, seq):
        h = self.ln(self.act(self.transform(seq)))
        return self.decoder(h)

class NextSentenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(H, 2)

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
    mlm = F.cross_entropy(mlm_logits.reshape(-1, VOCAB), mlm_labels.reshape(-1), ignore_index=-100)
    nsp = F.cross_entropy(nsp_logits, nsp_labels.reshape(-1))
    return mlm + nsp

MASK_ID, MASK_PROB, CLS, SEP = 103, 0.15, 101, 102

def make_example(span_a, span_b_true, random_span, vocab_size):
    if torch.rand(1).item() < 0.5:
        span_b, nsp = span_b_true, 0
    else:
        span_b, nsp = random_span, 1
    ids = [CLS] + span_a + [SEP] + span_b + [SEP]
    seg = [0] * (len(span_a) + 2) + [1] * (len(span_b) + 1)
    labels = [-100] * len(ids)
    candidates = [i for i, tok in enumerate(ids) if tok not in (CLS, SEP)]
    num_to_predict = min(len(candidates), max(1, int(round(len(ids) * MASK_PROB))))
    for j in torch.randperm(len(candidates)).tolist()[:num_to_predict]:
        i = candidates[j]
        labels[i] = ids[i]
        r = torch.rand(1).item()
        if r < 0.8:
            ids[i] = MASK_ID
        elif r < 0.9:
            ids[i] = torch.randint(0, vocab_size, (1,)).item()
    return ids, seg, labels, nsp
```
