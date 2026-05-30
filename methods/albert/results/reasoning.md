Let me start from the thing that doesn't add up. The whole story of pretrained language representations has been "bigger is better" — wider hidden size, more layers, more heads, and the downstream numbers climb. So I take a large bidirectional Transformer encoder and just double its hidden size, keeping the optimization budget fixed — same steps, same learning rate — expecting a free lift. Instead the bigger model does *worse* on downstream tasks. And when I look at its training curves there's no overfitting; the larger model just has lower masked-LM accuracy. So it's not that the bigger model memorized and failed to generalize. It's that the bigger model is *harder to train* — I can't even get it to fit as well. That's the degradation phenomenon, the one where adding capacity makes optimization worse rather than generalization. Capacity isn't my problem; trainability and the sheer parameter count are.

And the parameter count is a real wall, not just an annoyance. These models already run to hundreds of millions, heading to billions, of parameters. Memory caps how wide I can go. And in distributed training the communication cost scales with the number of parameters, so every extra parameter taxes every synchronization step. People have attacked the *memory* side — recompute activations to trade compute for memory, reconstruct each layer's activations from the next so you don't store them, or split the model across devices. But none of those reduce the parameter *count*, so the communication tax stays, and the degradation-on-widening stays. I want to actually reduce parameters — ideally in a way that *also* makes the wide model trainable.

So let me ask where the parameters even are, because if I'm going to cut them I should cut where it's cheap. Two pools dominate. One: the token embedding matrix, V rows by E columns, V being the vocabulary, tens of thousands of rows. Two: the per-layer Transformer weights — attention projections plus a feed-forward block whose inner dimension is conventionally four times the hidden width — and these are *replicated independently across every layer*, so the depth multiplies that pool L-fold.

Take the embedding pool first. The convention is to tie the embedding dimension E to the hidden size H, so E ≡ H. Stare at that for a second — why should those two be equal? They're doing different jobs. The embedding is a lookup: it maps a token id to a vector with no context, a context-*independent* representation of the word type. The hidden states deep in the network are context-*dependent* — and the entire power of these models comes from context refining each token's meaning. Those are different capacities. There's no reason the context-free lookup needs to be as wide as the contextual representation. If anything I'd want H ≫ E: spend the width where context lives, not in the lookup table.

And the practical cost of tying them is brutal. The embedding matrix is V×E, and if E ≡ H, then every time I widen H I widen this V-row matrix proportionally — tens of thousands of rows getting wider, most of which see only sparse gradient updates because most tokens are rare. That's a huge, sparsely-trained block of parameters whose size is dictated by my hidden width for no modeling reason. So untie them. Instead of projecting one-hot tokens straight into the H-dimensional hidden space, project first into a low-dimensional embedding space of size E, then project E up to H with a small matrix. The parameter count goes from order V×H to order V×E + E×H. When H ≫ E that's a big reduction — the V-row matrix is now only V×E, and the only thing that scales with H is the tiny E×H projection. I'll use a single shared E for every wordpiece; unlike whole-word vocabularies where frequent and rare words might want different dimensions, wordpieces are distributed fairly evenly, so one embedding size is fine. That decouples my embedding budget from my hidden width: I can now grow H without dragging the embedding matrix along.

Now the bigger pool — the per-layer weights replicated L times. Why does every layer need its *own* attention and feed-forward weights? The idea of reusing the same weights across layers has been tried for translation; a recurrent weight-shared Transformer even beat a vanilla one on some tasks, and another line showed a shared-layer Transformer can settle into a fixed point. If I share parameters across all layers, the per-layer pool stops scaling with depth entirely — the parameter count becomes independent of L. That's the second big cut, and it directly attacks the communication tax.

But sharing is a strong constraint — am I about to cripple the model? Let me think about what could go wrong and what the options are. I could share only the feed-forward weights, only the attention weights, or everything. And I'm a little worried about the fixed-point story: if the shared layer drives input and output embeddings to coincide, the network collapses to doing nothing after the first application. Let me check what actually happens by measuring, layer to layer, the distance between each layer's input and output representation — both the L2 distance and the angle between them. If they were converging to a fixed point those would head to zero. What I see instead is that they *oscillate* — they shrink relative to the unshared model and the layer-to-layer transitions become much smoother, but even after twenty-four layers they don't collapse to zero. So sharing isn't forcing a degenerate fixed point; it's *stabilizing* the parameters, smoothing the trajectory through depth, which reads as a regularizer. That's encouraging — it might be part of why the shared model is easier to train than the naively-widened one.

Where does sharing cost accuracy, though? When I compare the variants: sharing everything hurts, but the hurt is milder when E is small than when E is large, and almost all of the damage comes from sharing the *feed-forward* weights — sharing only the attention weights costs essentially nothing. So the principled choice would be share-attention-only. But I'm optimizing for the maximum parameter reduction and for a model whose count is flat in depth, and all-sharing gives the cleanest, largest saving plus the strongest stabilization. The accuracy cost of all-sharing is small, especially at small E, and it's the configuration that lets me scale H aggressively, so I'll make all-shared the default and pair it with the small embedding.

Let me pick E. Sweeping the embedding dimension under all-sharing, performance peaks around E = 128 and doesn't improve with larger E — consistent with the lookup not needing much width. So E = 128, fixed, and now H is free to grow.

With both cuts in place the arithmetic is striking: a configuration matching large-BERT's depth and width has on the order of eighteen times fewer parameters, because the embedding matrix shrank and the per-layer weights are counted once instead of twenty-four times. And because the count is flat in depth and small, I can push H far past where the unshared model degraded — to 2048, even 4096 hidden — while *still* having fewer parameters than the original large model. The very widening that broke the dense model is now trainable, because sharing stabilizes it and the parameter budget no longer explodes.

One subtlety about depth once everything is shared. The shared block is just applied repeatedly, so "more layers" means "apply the same function more times." When I sweep depth at fixed width, going from one to three applications helps a lot, then returns diminish quickly — a twelve-layer shared model is already close to a twenty-four-layer one, and going deeper can even decline. For the widest configuration I'll therefore use only twelve layers; deeper gives no real gain once the layer is shared, and costs compute.

Now the auxiliary objective, which I've been suspicious of. Alongside masked language modeling, the standard recipe adds next-sentence prediction: classify whether two segments were consecutive, with positives being adjacent segments and negatives being segments from two *different* documents. It was supposed to help sentence-pair reasoning, but it's been found unreliable and quietly dropped. Why would it be so weak? Look at how the negative is built — two segments from different documents. To call that "not consecutive," the model only has to notice the two segments are about different *topics*. Topic is exactly the kind of thing masked language modeling already captures from word co-occurrence. So next-sentence prediction collapses into topic detection, which is easy and redundant with the main objective; it teaches almost nothing about how sentences cohere in order.

I still believe inter-sentence structure matters for understanding — but the task has to actually be about *coherence*, not topic. So construct the negative differently: keep the two segments from the *same* document and *consecutive*, but swap their order. Positive is the two consecutive segments in their true order; negative is the identical two segments reversed. Now topic is identical on both sides — same document, same content — so topic gives the model nothing. The only way to win is to read the discourse cues that tell you which segment comes first. Call it sentence-order prediction. It's strictly harder, and it can't be shortcut through topic.

Does that distinction bear out? If I take a model trained only with the old next-sentence objective and test it on the order task, it scores at chance — confirming the old objective never learned order, only topic shift. A model trained on the order objective, conversely, can still do the old next-sentence task reasonably *and* the order task well, because coherence cues subsume the topic ones. And the order objective is what helps the multi-sentence downstream tasks. So I replace next-sentence prediction with sentence-order prediction.

A couple of training details follow from the regime I'm now in. For the masked-LM target, rather than masking isolated subword pieces I'll mask short spans — choose an n-gram length randomly with probability proportional to 1/n, capped at three, so the target is a whole word or short phrase like "White House correspondents." Predicting a coherent span is a harder, more linguistically meaningful target than predicting one subword given its neighbors. And on regularization: when I look at the largest model's curves it's *underfitting* — the training loss is high and there's no overfitting even after a million steps. Dropout's whole purpose is to fight overfitting, and there isn't any here; it's just throttling capacity I can't afford to waste. So for the large runs I remove dropout, and masked-LM accuracy goes up.

Let me now write this as code. The model body is a standard Transformer encoder; the contribution is the factorized embedding, the single shared block applied L times, and the order-based inter-sentence objective.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- factorized embedding: lookup in E, then project E -> H ---
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden):   # embed_dim E << hidden H
        super().__init__()
        self.word = nn.Embedding(vocab_size, embed_dim)          # V x E (small)
        self.proj = nn.Linear(embed_dim, hidden)                 # E x H
    def forward(self, ids):
        return self.proj(self.word(ids))                         # [B, T, H]

# --- one encoder block (attention + 4H feed-forward, GELU) ---
class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0):    # dropout off for large runs
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden); self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Linear(4*hidden, hidden))
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

# --- cross-layer sharing: ONE block, applied n_layers times ---
class SharedEncoder(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.block = EncoderBlock(hidden, n_heads, dropout)      # the ONLY block
    def forward(self, x, mask=None):
        for _ in range(self.n_layers):                          # reuse same weights
            x = self.block(x, mask)
        return x

# --- sentence-order prediction: positive = true order, negative = swapped ---
def sop_examples(doc_segments):
    seg_a, seg_b = doc_segments                                 # two CONSECUTIVE segments, same doc
    if torch.rand(1) < 0.5:
        return (seg_a, seg_b), 1                                # correct order
    else:
        return (seg_b, seg_a), 0                                # swapped: only coherence distinguishes

class SOPHead(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.cls = nn.Linear(hidden, 2)
    def forward(self, features):
        return self.cls(features[:, 0, :])                      # over [CLS]

# --- n-gram masking for the MLM target: p(n) ∝ 1/n, n in 1..3 ---
def sample_ngram_length(max_n=3):
    w = torch.tensor([1.0/n for n in range(1, max_n+1)])
    return 1 + int(torch.multinomial(w / w.sum(), 1))           # length with prob (1/n)/sum_k(1/k)

class MLMHead(nn.Module):
    def __init__(self, hidden, embed_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden, embed_dim)               # project back toward embedding space
        self.ln = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        return self.decoder(self.ln(F.gelu(self.dense(x))))

def total_loss(mlm_logits, mlm_labels, sop_logits, sop_labels):
    return (F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)),
                            mlm_labels.view(-1), ignore_index=-100)
            + F.cross_entropy(sop_logits, sop_labels))          # MLM + SOP

# config: E=128 fixed; xxlarge = L12, H4096, A=64; LAMB, batch 4096, lr 0.00176
```

So the chain is: naively widening the dense model made it *worse* and untrainable, and the parameter count was hitting memory and communication walls — so I had to reduce parameters in a way that also stabilizes training. The embedding dimension was needlessly tied to the hidden size though it does a context-free job, so I factorized it (V×E + E×H instead of V×H) and fixed E small at 128. The per-layer weights were replicated across depth, so I shared one block across all layers; measuring input-output embedding distances showed the shared model oscillates rather than collapsing, confirming it stabilizes rather than degenerates, and the count is now flat in depth — letting me push the hidden size far past where the dense model broke while still having fewer parameters, with twelve applications enough once the block is shared. The next-sentence objective was solvable by topic alone and thus redundant with masked LM, so I replaced it with sentence-order prediction — same segments, swapped order — which can only be won by reading coherence. And because the large model underfits, dropout was throttling needed capacity, so I dropped it.
