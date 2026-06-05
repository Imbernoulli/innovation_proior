## Research question

Transformer-based masked-language models pre-trained on large corpora are the engine of
modern NLP, and the prevailing way to improve them is to scale data and parameters. But
there is a structural question underneath the scaling: how should a Transformer encode
*word position*, and how should position interact with *content* when computing
attention? In the standard recipe, each input token is represented by a single vector
formed by *adding* its content embedding and its position embedding, and attention is
computed on these summed vectors. The pain points: (i) adding content and position
entangles two things that play different roles — the attention between two words depends
on their contents *and* on their relative positions, and on how those two interact, but
a single summed vector blurs this; (ii) existing relative-position schemes inject only a
*partial* slice of the content–position interaction; and (iii) absolute position, which
is genuinely needed to disambiguate some predictions, has to be supplied somewhere — and
the usual place (the input layer) may interfere with learning relative structure. Can a
better parameterization of content and position — and of where absolute position enters
— improve pre-training efficiency and downstream accuracy at a fixed model scale?

## Background

**The Transformer encoder and single-head self-attention.** A model is a stack of
blocks, each a multi-head self-attention layer followed by a position-wise feed-forward
network, with residual connections and layer normalization. For hidden states
`H ∈ R^{N×d}`, single-head self-attention projects `Q = HW_q`, `K = HW_k`, `V = HW_v`,
forms `A = QKᵀ/√d`, and outputs `softmax(A)V`. The `1/√d` scaling keeps the logits from
growing with dimension so the softmax does not saturate — important for stable training,
especially at large scale.

**Why position must be injected, and the two prevailing ways.** Self-attention is
permutation-invariant: by itself it cannot tell word order. So position information is
added. (i) *Absolute* position embeddings (Vaswani et al. 2017; Radford et al. 2019;
Devlin et al. 2018): a per-position vector is summed into each input token's embedding
at the bottom of the network. (ii) *Relative* position representations (Shaw et al.
2018; Huang et al. 2018; Dai et al. 2019, Transformer-XL): a learned bias depending on
the *distance* between query and key is added into the attention score. It has been
shown empirically that relative position representations are more effective for both
understanding and generation tasks than absolute ones.

**What existing relative schemes actually compute.** Decomposing the attention score
between a query at position `i` and a key at position `j` into content and position
parts, the full interaction has four pieces: content-to-content, content-to-position,
position-to-content, and position-to-position. Existing relative-position methods add a
bias that corresponds to only the content-to-position piece (a query-content vector
dotted with a relative-position key vector). This is the diagnostic that motivates the
method: the symmetric *position-to-content* piece — how a key's content matters relative
to the query's position — is omitted, even though the attention of a word pair plainly
depends on this interaction too.

**Masked language modeling.** Pre-training corrupts a sequence `X` into `X̃` by masking
~15% of tokens and trains `θ` to maximize `Σ_{i∈C} log p_θ(x̃_i = x_i | X̃)` over the
masked index set `C`. The standard masking splits the masked positions: 80% replaced
with `[MASK]`, 10% replaced with a random token, 10% kept unchanged. The unchanged 10%
exists to reduce the pre-train/fine-tune mismatch (`[MASK]` never appears downstream).

**Why absolute position is still sometimes necessary.** Relative position plus content
is not always enough to predict a masked word. Consider "a new **store** opened beside
the new **mall**" with *store* and *mall* masked: both follow *new* at the same relative
offset and have similar local context, yet they play different syntactic roles (the
subject is *store*). Disambiguating them requires their *absolute* positions in the
sentence. So absolute position carries real signal — the open design question is *where*
to add it without harming the learning of relative structure.

**Virtual adversarial training (Miyato et al. 2018; Jiang et al. 2019).** A
regularizer: perturb the input slightly to form an adversarial example and require the
model to produce the same output distribution on the perturbation as on the clean input.
For text, the perturbation is applied to word embeddings. A known difficulty: embedding
vector norms vary widely across words and grow with model size, which destabilizes the
perturbation scale — a pre-method fact that any embedding-space adversarial scheme must
contend with.

## Baselines

**Absolute-position MLM encoder (Devlin et al. 2018; Liu et al. 2019, RoBERTa).** Sum
content + absolute-position embeddings at the input, stack Transformer blocks, pre-train
with MLM (RoBERTa drops next-sentence prediction, uses dynamic masking, larger batches,
more data). Core idea: a single strong general encoder. Gaps: content and position are
entangled by summation at the input; position information is absolute, which is
empirically weaker than relative for many tasks.

**Relative-position Transformers (Shaw et al. 2018; Huang et al. 2018; Transformer-XL,
Dai et al. 2019).** Add a learned relative-position bias into the attention logits,
depending on the query–key distance. Core idea: model order by distance, which
generalizes across positions and sequence lengths. Gap: the bias captures only the
content-to-position interaction; the position-to-content interaction is missing. Storing
a distinct relative-position embedding per query naively costs `O(N²d)` memory.

**ALBERT (Lan et al. 2019).** Reduce parameters via cross-layer weight sharing and
factorized embeddings. Core idea: parameter efficiency. Gap: orthogonal to how content
and position interact in attention; does not address the entanglement or the missing
interaction term.

## Evaluation settings

Pre-training corpus: English Wikipedia (12GB), BookCorpus (6GB), OpenWebText (38GB), and
STORIES (a CommonCrawl subset, 31GB) — ~78GB after deduplication, with 5% held out for
validation. Tokenization follows the standard BPE vocabulary; dynamic batching, with
span masking (spans up to length 3) added to token masking. Architectures span a base
config (12 layers, hidden 768, 12 heads, head size 64, FFN inner 3072) and a large
config (24 layers, hidden 1024, 16 heads, FFN inner 4096), maximum relative distance
`k = 512`. Optimizer: Adam with decoupled weight decay; warmup 10k steps, linear decay,
batch size 2k, up to 1M steps. Downstream benchmarks (the natural yardsticks, all
pre-existing): GLUE (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B); SQuAD v1.1 and
v2.0 reading comprehension (passages from ~500 Wikipedia articles, with unanswerable
questions in v2.0); RACE; and for generation, language-model perplexity on WikiText-103
and the SuperGLUE benchmark. Fine-tuning uses task-specific hyperparameter search over
small grids of learning rate, batch size, and warmup. No outcome numbers are part of
these settings.

## Code framework

The harness is a standard MLM Transformer encoder: embeddings, a stack of blocks each
with self-attention and a feed-forward network, and an output projection that decodes
masked tokens against the vocabulary. The self-attention's score computation, the way
position is represented and where absolute position enters the decoder, and the
fine-tuning regularizer are the empty slots.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.h = d, n_heads
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d) for _ in range(3))
    def forward(self, H, attn_mask, **position_inputs):
        # TODO: design how the attention score is computed from content and position.
        pass

class FeedForward(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.attn, self.ff = SelfAttention(d, n_heads), FeedForward(d, d_ff)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, H, attn_mask, **pos):
        H = self.ln1(H + self.attn(H, attn_mask, **pos))
        return self.ln2(H + self.ff(H))

class Encoder(nn.Module):
    def __init__(self, vocab, d=768, L=12, n_heads=12, d_ff=3072):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks  = nn.ModuleList(EncoderBlock(d, n_heads, d_ff) for _ in range(L))
        # TODO: position representation(s) used by attention.
    def forward(self, input_ids, attn_mask):
        H = self.tok_emb(input_ids)
        for blk in self.blocks:
            H = blk(H, attn_mask)   # TODO: pass position inputs
        return H

def mlm_decode(encoder_out, mask_positions, vocab_proj, **absolute_position):
    # TODO: decode masked tokens; decide whether/where absolute position enters.
    pass

def adversarial_finetune_step(model, batch, eps):
    # TODO: virtual-adversarial regularizer for fine-tuning; handle embedding-norm scale.
    pass
```
