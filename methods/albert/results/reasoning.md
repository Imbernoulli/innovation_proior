Let me start from the thing that doesn't add up. The whole story of pretrained language representations has been "bigger is better" — wider hidden size, more layers, more heads, and the downstream numbers climb. So I take a large bidirectional Transformer encoder and just double its hidden size, keeping the optimization budget fixed — same steps, same learning rate — expecting a free lift. Instead the bigger model does *worse* on downstream tasks. And when I look at its training curves there's no overfitting; the larger model just has lower masked-LM accuracy. So it's not that the bigger model memorized and failed to generalize. It's that the bigger model is *harder to train* — I can't even get it to fit as well. That's the degradation phenomenon, the one where adding capacity makes optimization worse rather than generalization. Capacity isn't my problem; trainability and the sheer parameter count are.

And the parameter count is a real wall, not just an annoyance. These models already run to hundreds of millions, heading to billions, of parameters. Memory caps how wide I can go. And in distributed training the communication cost scales with the number of parameters, so every extra parameter taxes every synchronization step. People have attacked the *memory* side — recompute activations to trade compute for memory, reconstruct each layer's activations from the next so you don't store them, or split the model across devices. But none of those reduce the parameter *count*, so the communication tax stays, and the degradation-on-widening stays. I want to actually reduce parameters — ideally in a way that *also* makes the wide model trainable.

So let me ask where the parameters even are, because if I'm going to cut them I should cut where it's cheap. Let me put real numbers on it for large-BERT: vocabulary V ≈ 30K, hidden H = 1024, depth L = 24. The token embedding matrix is V×H = 30000·1024 ≈ 30.7M. Each Transformer layer is attention (the four H×H projections, 4H²) plus a feed-forward block with inner width 4H (two H×4H matrices, 8H²), so about 12H² per layer ignoring biases and norms — 12·1024² ≈ 12.6M, times 24 layers ≈ 302M. So the per-layer pool is ten times the embedding pool, and it's the depth multiplier L that makes it dominate. Adding the two gives ≈ 333M, which lands right on the reported 334M for large-BERT — so this accounting captures essentially all the parameters. Two pools, then: the V-row embedding table, and the per-layer weights replicated L-fold.

Take the embedding pool first. The convention is to tie the embedding dimension E to the hidden size H, so E ≡ H. Stare at that for a second — why should those two be equal? They're doing different jobs. The embedding is a lookup: it maps a token id to a vector with no context, a context-*independent* representation of the word type. The hidden states deep in the network are context-*dependent* — and the entire power of these models comes from context refining each token's meaning. Those are different capacities. There's no reason the context-free lookup needs to be as wide as the contextual representation. If anything I'd want H ≫ E: spend the width where context lives, not in the lookup table.

And the practical cost of tying them is brutal. The embedding matrix is V×E, and if E ≡ H, then every time I widen H I widen this V-row matrix proportionally — tens of thousands of rows getting wider, most of which see only sparse gradient updates because most tokens are rare. That's a huge, sparsely-trained block whose size is dictated by my hidden width for no modeling reason. So untie them. Instead of projecting one-hot tokens straight into the H-dimensional hidden space, project first into a low-dimensional embedding space of size E, then project E up to H with a small matrix. The parameter count goes from V×H to V×E + E×H. Let me check this is actually a win and not just shuffling cost around. Pick E = 128 and run the two expressions across the hidden sizes I care about:

- H = 1024: tied V·H ≈ 30.7M, factorized V·E + E·H ≈ 3.97M — an 87% cut.
- H = 2048: tied ≈ 61.4M, factorized ≈ 4.1M — 93%.
- H = 4096: tied ≈ 122.9M, factorized ≈ 4.4M — 96%.

The thing I want to notice is not just that it's smaller but that the saving *grows* with H. The factorized table is V·E ≈ 3.84M plus a tiny E·H projection that barely moves as H quadruples; the tied table grows linearly in H without bound. So the wider I go, the more this matters — exactly the regime I'm trying to reach. That settles a worry I had, that the E×H projection might eat the saving; at E = 128 it's ≈ 0.5M even at H = 4096, negligible. I'll use a single shared E for every wordpiece: unlike whole-word vocabularies where frequent and rare words might want different dimensions, wordpieces are distributed fairly evenly, so one embedding size is fine. That decouples my embedding budget from my hidden width: I can now grow H without dragging the embedding matrix along.

Now the bigger pool — the per-layer weights replicated L times, the 302M from above. Why does every layer need its *own* attention and feed-forward weights? The idea of reusing the same weights across layers has been tried for translation; a recurrent weight-shared Transformer even beat a vanilla one on some tasks, and another line showed a shared-layer Transformer can settle into a fixed point. If I share parameters across all layers — one block applied L times — the per-layer pool stops scaling with depth entirely: it's counted once. At H = 1024 that's the ≈ 12.6M of a single block instead of ≈ 302M for twenty-four of them. The parameter count becomes independent of L. That directly attacks the communication tax, since the tax is proportional to count.

But sharing is a strong constraint — am I about to cripple the model? Let me think about what could go wrong. I could share only the feed-forward weights, only the attention weights, or everything. And I'm worried about the fixed-point story I just cited: if a shared layer drives input and output embeddings to coincide, the network collapses to doing nothing after the first application, and stacking it twelve or twenty-four times buys nothing. That's a concrete failure mode, so let me actually measure it rather than assume either way. I'll track, layer to layer, the distance between each layer's input and output representation — both the L2 distance and the cosine angle between them. If the shared block were converging to a fixed point, those would decay monotonically toward zero with depth. What I see instead is that they *oscillate*: they're smaller than in the unshared model and the layer-to-layer transitions are much smoother, but even after twenty-four applications they don't fall to zero — the input and output of the block stay distinct at every depth. So the fixed-point collapse I feared doesn't happen. The shared block keeps transforming its input all the way down; sharing is *stabilizing* the parameters and smoothing the trajectory through depth, which reads as a regularizer. That smoother trajectory is plausibly part of why the shared model is easier to train than the naively-widened dense one — though I'd want to confirm that link directly rather than lean on it.

Where does sharing cost accuracy, though — because "doesn't collapse" isn't the same as "free." When I compare the variants: sharing everything hurts, but the hurt is milder when E is small than when E is large, and almost all of the damage comes from sharing the *feed-forward* weights — sharing only the attention weights costs essentially nothing. So on pure accuracy the principled choice would be share-attention-only. But the FFN is two thirds of the per-layer weights, so share-attention-only leaves the bulk of the depth-multiplied pool in place and the count is no longer flat in L. I'm optimizing for the maximum parameter reduction and a count that's flat in depth, and all-sharing gives the cleanest, largest saving plus the strongest stabilization. Since the accuracy cost of all-sharing is small at small E — and I've already committed to small E from the factorization — I'll make all-shared the default and pair it with the small embedding. The two cuts reinforce each other: small E is what makes all-sharing cheap.

Let me pick E. Sweeping the embedding dimension under all-sharing, performance peaks around E = 128 and doesn't improve with larger E — consistent with the lookup not needing much width, and consistent with the factorization math where larger E only inflates V·E for no return. So E = 128, fixed, and now H is free to grow.

With both cuts in place let me total it up for the same depth-and-width as large-BERT. Embedding pool: V·E + E·H ≈ 4.0M instead of 30.7M. Per-layer pool: one block ≈ 12.6M instead of 302M. Sum ≈ 16.6M against the original 334M — about a 20× reduction by my biases-omitted count, and the published figure with all the small terms in is 18M, ≈ 18× fewer. Both pools moved: the embedding matrix shrank and the layer weights are counted once instead of twenty-four times. And because the count is now flat in depth and small, I can push H far past where the dense model degraded — to 2048, even 4096 hidden — while *still* having fewer parameters than the original large model: at H = 4096 the embedding is ≈ 4.4M and one block is ≈ 200M, so even the widest configuration sits near the original 334M rather than the ~5B a dense L24-H4096 would be. The very widening that broke the dense model is now reachable, because sharing stabilizes it and the parameter budget no longer explodes with H.

One subtlety about depth once everything is shared. The shared block is just applied repeatedly, so "more layers" means "apply the same function more times." When I sweep depth at fixed width, going from one to three applications helps a lot, then returns diminish quickly — a twelve-layer shared model is already close to a twenty-four-layer one, and going deeper can even decline. This fits the oscillation picture: the block keeps moving the representation but the useful work is mostly done in the first several applications. For the widest configuration I'll therefore use only twelve layers; deeper gives no real gain once the layer is shared, and costs compute on every application.

Now the auxiliary objective, which I've been suspicious of. Alongside masked language modeling, the standard recipe adds next-sentence prediction: classify whether two segments were consecutive, with positives being adjacent segments and negatives being segments from two *different* documents. It was supposed to help sentence-pair reasoning, but it's been found unreliable and quietly dropped. Why would it be so weak? Look at how the negative is built — two segments from different documents. To call that "not consecutive," the model only has to notice the two segments are about different *topics*. Topic is exactly the kind of thing masked language modeling already captures from word co-occurrence. So next-sentence prediction risks collapsing into topic detection, which is easy and redundant with the main objective; it would teach almost nothing about how sentences cohere in order.

I still believe inter-sentence structure matters for understanding — but the task has to actually be about *coherence*, not topic. So construct the negative differently: keep the two segments from the *same* document and *consecutive*, but swap their order. Positive is the two consecutive segments in their true order; negative is the identical two segments reversed. Now topic is identical on both sides — same document, same content — so topic gives the model nothing. The only way to win is to read the discourse cues that tell you which segment comes first. I can keep the old binary-classification convention: uncorrupted order is label 0, corrupted order is label 1. Sentence-order prediction. It's strictly harder, and by construction it can't be shortcut through topic.

The shortcut hypothesis makes a testable prediction I can check, at least in principle. Take a model trained only with the old next-sentence objective and evaluate it on this new task — recover the order of two same-document consecutive segments. If next-sentence prediction really is just a topic detector, it has no signal when both alternatives contain the same text, so it should sit near 50% on the order task; if it had genuinely learned coherence, it should be well above chance. The prediction is sharp: near-chance order accuracy from an NSP-trained model would be direct evidence that NSP never learned coherence and that the redundancy argument is right. I can't run that million-step pretraining here, so I'll hold it as the prediction the design rides on rather than claim the result — but the logic is what motivates replacing NSP with SOP, and the swap is cheap to make either way.

A couple of training details follow from the regime I'm now in. For the masked-LM target, rather than masking isolated subword pieces I'll mask short spans — choose an n-gram length randomly with probability proportional to 1/n, capped at three. Let me make that concrete so I know what I'm sampling: weights (1, 1/2, 1/3) sum to 11/6, so p(1) = 6/11 ≈ 0.545, p(2) = 3/11 ≈ 0.273, p(3) = 2/11 ≈ 0.182. So single tokens still dominate at a bit over half, but roughly 45% of the time I mask a two- or three-gram — enough to make whole words and short phrases like "White House correspondents" a regular target without the mask spans running long. Predicting a coherent span is a harder, more linguistically meaningful target than predicting one subword given its neighbors. And on regularization: when I look at the largest model's curves it's *underfitting* — the training loss is high and there's no overfitting even after a million steps. Dropout's whole purpose is to fight overfitting, and there isn't any here; it's just throttling capacity I can't afford to waste. So for the large runs I remove dropout, and masked-LM accuracy goes up — which is itself a small confirmation that the binding constraint was underfitting, not variance.

Let me now write this as code. The model body is a standard Transformer encoder; the pieces that matter are the embedding stem, the encoder stack, the masked-LM head tied back to the input embeddings, and the order-based inter-sentence objective.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- embedding stem: lookup in E, add type/position in E, project E -> H ---
class EmbeddingStem(nn.Module):
    def __init__(self, vocab_size, hidden, embedding_width=128,
                 max_positions=512, type_vocab_size=2, dropout=0.0):
        super().__init__()
        self.word = nn.Embedding(vocab_size, embedding_width)       # V x E
        self.position = nn.Embedding(max_positions, embedding_width)
        self.token_type = nn.Embedding(type_vocab_size, embedding_width)
        self.norm = nn.LayerNorm(embedding_width)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(embedding_width, hidden)             # E x H
                     if embedding_width != hidden else nn.Identity())

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        x = self.word(input_ids) + self.position(positions) + self.token_type(token_type_ids)
        return self.proj(self.drop(self.norm(x)))                   # [B, T, H]

# --- one encoder block (attention + 4H feed-forward, GELU) ---
class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0):              # dropout off for large runs
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden); self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Linear(4*hidden, hidden))

    def forward(self, x, padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

# --- cross-layer sharing: one block, applied n_layers times ---
class EncoderStack(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.block = EncoderBlock(hidden, n_heads, dropout)        # the only block

    def forward(self, x, padding_mask=None):
        for _ in range(self.n_layers):                             # reuse same weights
            x = self.block(x, padding_mask)
        return x

# --- sentence-order prediction: label 0 = true order, label 1 = swapped ---
def inter_sentence_examples(seg_a, seg_b):                         # consecutive segments, same doc
    if torch.rand(()) < 0.5:
        return (seg_a, seg_b), 0                                   # correct order
    return (seg_b, seg_a), 1                                       # swapped

class SentencePairHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, sequence_output):
        pooled = torch.tanh(self.pooler(sequence_output[:, 0, :]))  # [CLS] pooler
        return self.classifier(pooled)

# --- n-gram masking for the MLM target: p(n) ∝ 1/n, n in 1..3 ---
def sample_ngram_length(max_n=3):
    w = torch.tensor([1.0/n for n in range(1, max_n+1)])
    return 1 + int(torch.multinomial(w / w.sum(), 1))             # (1/n) / sum_k (1/k)

def gather_positions(sequence_output, positions):
    batch, seq_len, width = sequence_output.shape
    offsets = torch.arange(batch, device=sequence_output.device).unsqueeze(1) * seq_len
    flat_positions = (positions + offsets).reshape(-1)
    return sequence_output.reshape(batch * seq_len, width).index_select(0, flat_positions)

class MLMHead(nn.Module):
    def __init__(self, hidden, embedding_width, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden, embedding_width)           # H -> E
        self.ln = nn.LayerNorm(embedding_width)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, sequence_output, positions, embedding_table):
        x = gather_positions(sequence_output, positions)
        x = self.ln(F.gelu(self.dense(x)))
        return x @ embedding_table.weight.t() + self.bias          # tied to input embeddings

def total_loss(mlm_logits, mlm_labels, mlm_weights, pair_logits, pair_labels):
    flat_labels = mlm_labels.reshape(-1)
    flat_weights = mlm_weights.reshape(-1).float()
    mlm_each = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)),
                               flat_labels, reduction="none")
    mlm_loss = (mlm_each * flat_weights).sum() / (flat_weights.sum() + 1e-5)
    pair_loss = F.cross_entropy(pair_logits, pair_labels.reshape(-1))
    return mlm_loss + pair_loss

# config: E=128 fixed; xxlarge = L12, H4096, A=64; LAMB, batch 4096, lr 0.00176
```
