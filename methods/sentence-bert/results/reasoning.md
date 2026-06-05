Let me start from the wall I keep hitting. I have a pre-trained Transformer that is the best thing going for sentence-pair tasks — semantic similarity, NLI, paraphrase — when I feed it both sentences at once, joined by `[SEP]`, and let full self-attention chew across all the tokens of both. The attention between the two sentences at every layer is precisely why it's so accurate. But it means there is no per-sentence object I can compute once and reuse: the prediction only exists for a *pair*, jointly encoded. So if I want the most similar pair in a collection of 10,000 sentences, I have to encode all ~50 million pairs — about 65 hours on one GPU. Clustering, semantic search, dedup: all infeasible. The cost is quadratic in pairs, and the unit of computation is a full forward pass.

What I actually want is obvious: a *single-sentence* fixed-size vector, computed once per sentence, such that semantically similar sentences are close under a cheap metric like cosine. Then a collection of 10,000 sentences is 10,000 encodes (~seconds) plus near-free vector comparisons, and with an index, nearest-neighbor search is milliseconds. The expensive joint encoding gets replaced by precomputable vectors.

So just take the pre-trained encoder, push one sentence through, and read off a vector — the classification-token output, or the mean of the token outputs; people already do this. Let me check whether it actually works before I build on it. Score the cosine similarity of these vectors against human similarity judgments (Spearman correlation on STS). The result is bad — embarrassing, even. Averaging the encoder's token outputs lands *below* averaging plain GloVe word vectors, and the classification-token vector is worse still. So the encoder, out of the box, maps sentences into a space whose *geometry* is wrong for cosine similarity. The information is in there, but the angles between sentence vectors don't track meaning.

Why would that be? The encoder was pre-trained with masked-language-modeling and a next-sentence objective, and used downstream by fine-tuning a classifier on top that can re-weight and recombine dimensions however it likes. Cosine similarity has no such freedom — it weights every dimension equally and only sees the angle. A representation can be perfectly good for a downstream classifier and useless under cosine, because nothing ever forced "semantically close" to mean "small angle." Indeed, when I evaluate the same naive embeddings with a logistic-regression probe (which *can* re-weight dimensions), they look fine — it's specifically the cosine/Euclidean geometry that's broken. So the fix isn't a better readout trick; the cosine geometry has to be *trained in*.

That reframes the problem: I need to fine-tune the encoder so that a single-sentence embedding sits in a space where small angle = semantic closeness. And I want to do this *without* throwing away the pre-trained knowledge — start from the pre-trained encoder, not from scratch.

For that I need a training structure that produces *comparable* vectors for two sentences and a loss defined on those vectors. The classical answer is a siamese network: pass each sentence through the *same* encoder with *tied* weights, so both land in one shared space and are directly comparable, then define the loss on the two resulting vectors. Tied weights are the whole point — if the two towers had separate parameters, their vectors wouldn't live in the same geometry. So: shared encoder, pooling to a fixed vector, loss on the pooled vectors.

First, pooling — how to turn the variable number of token outputs into one fixed vector. Options: the classification-token output, the mean of the token outputs, or a max-over-time. I already saw the classification token is a poor standalone summary. Mean pooling averages the contextual token vectors, which is a sensible default summary of the whole sentence and parallels the averaged-word-embedding baselines that already beat the naive encoder vectors. I'll take mean pooling as the default and keep the others as things to ablate; I expect mean to be at least as good as the classification token, and max to be more fragile.

Now the loss, and it should match the data I have. Two regimes.

The data I have most of is NLI — SNLI plus MultiNLI, a million labeled pairs tagged entailment / contradiction / neutral. That's a 3-way classification over a *pair*, not a similarity score, so I can't regress cosine directly. I need to turn the two sentence vectors `u` and `v` into features for a softmax classifier. The natural features: the two vectors themselves, and something that captures how they *differ*. So form a feature vector and feed it to a softmax over the 3 labels, training with cross-entropy:
`o = softmax(W_t · [u ; v ; |u − v|])`, with `W_t ∈ R^{3n × k}` for embedding dimension `n` and `k` labels.
Why `|u − v|`? The element-wise absolute difference measures, per dimension, how far apart the two embeddings are; including it pushes the classifier to make "different label" correspond to "large coordinate-wise difference," which is exactly the kind of pressure that shapes the *distance* structure of the space — and distance structure is what I'll exploit at inference with cosine. Why concatenate `u` and `v` themselves too, rather than only the difference? The raw vectors let the classifier condition on *what* the sentences are, not just how they differ, which the entailment/contradiction asymmetry needs.

Should I throw in more interaction terms — say the element-wise product `u·v`, as the from-scratch siamese baselines do, giving `[u ; v ; |u−v| ; u·v]`? Let me reason about what each term buys. The difference `|u−v|` is the load-bearing one because it directly shapes distances. The product is another interaction, but it doesn't have a clean distance interpretation and adds 2n more input dimensions to `W_t`. I'd want to actually compare configurations, but my expectation is that the difference term is essential, the raw vectors help, and the product is at best neutral and may hurt by diluting the difference signal — so the default is `[u ; v ; |u−v|]`. Crucially, all of these concatenation features matter *only for training the classifier*: at inference I discard `W_t` entirely and compare sentences by cosine of `u` and `v` directly. The classifier is scaffolding to bend the embedding space into shape.

The other regime is when I have actual similarity scores (the STS benchmark gives pairs labeled 0–5). Then I don't need a classifier at all — I can train the geometry I want *directly*. Compute the cosine similarity of `u` and `v` and regress it onto the gold score with mean-squared error. This is the most direct possible objective: make cosine equal the target similarity. So: classification objective with `[u;v;|u−v|]` cross-entropy when the data is categorical (NLI), regression objective with MSE on cosine when the data is graded (STS). And I can do both in sequence — pre-train the embeddings on NLI, then fine-tune on STS — to combine broad semantic signal with task-specific calibration.

A third structure for when the data comes as triplets — an anchor, a positive (should be close), a negative (should be far), as in the Wikipedia-section dataset. Then use a triplet margin loss: make the anchor closer to the positive than to the negative by at least a margin `ε`:
`max( ‖s_a − s_p‖ − ‖s_a − s_n‖ + ε, 0 )`,
with `‖·‖` Euclidean distance and a margin `ε` (I'll use `ε = 1`). The margin forces a gap rather than just any ordering, so the positive is pulled at least `ε` closer than the negative; the relu zeroes out triplets that already satisfy the gap so they stop contributing gradient.

Let me make sure the inference path is dead simple and consistent across all three: encode each sentence once → mean-pool → get `u`. Compare two sentences by cosine `(u·v)/(‖u‖‖v‖)`. For a collection: precompute all embeddings, then retrieval/clustering is plain vector math, optionally with an approximate-nearest-neighbor index. No pair ever goes through the encoder jointly again.

Code. The shared encoder with mean pooling first — mean over the real tokens only, masking out padding:

```python
import torch, torch.nn as nn, torch.nn.functional as F

def mean_pool(token_embeddings, attention_mask):
    # average token output vectors over non-pad positions
    mask = attention_mask.unsqueeze(-1).float()                 # [B, L, 1]
    summed = (token_embeddings * mask).sum(dim=1)               # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                    # avoid div-by-0
    return summed / counts

def embed(encoder, input_ids, attention_mask):
    tok = encoder(input_ids, attention_mask)                    # [B, L, H]
    return mean_pool(tok, attention_mask)
```

The NLI classification objective — note the encoder is shared, so `u` and `v` come from the same weights, and the classifier dimension is `3n` because we concatenate `[u; v; |u−v|]`:

```python
class SoftmaxObjective(nn.Module):
    def __init__(self, encoder, sent_dim, num_labels,
                 use_rep=True, use_diff=True, use_mul=False):
        super().__init__()
        self.encoder = encoder                                  # tied weights for both sentences
        self.use_rep, self.use_diff, self.use_mul = use_rep, use_diff, use_mul
        n_vec = (2 if use_rep else 0) + (1 if use_diff else 0) + (1 if use_mul else 0)
        self.classifier = nn.Linear(n_vec * sent_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, a, b, labels):
        u = embed(self.encoder, a.input_ids, a.attention_mask)
        v = embed(self.encoder, b.input_ids, b.attention_mask)
        feats = []
        if self.use_rep:  feats += [u, v]
        if self.use_diff: feats += [torch.abs(u - v)]           # the load-bearing distance feature
        if self.use_mul:  feats += [u * v]
        logits = self.classifier(torch.cat(feats, dim=1))       # W_t over [u; v; |u-v|]
        return self.loss_fct(logits, labels)
```

The STS regression objective — train cosine directly:

```python
class CosineRegressionObjective(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_fct = nn.MSELoss()

    def forward(self, a, b, gold):                              # gold scaled to [0, 1]
        u = embed(self.encoder, a.input_ids, a.attention_mask)
        v = embed(self.encoder, b.input_ids, b.attention_mask)
        cos = F.cosine_similarity(u, v)
        return self.loss_fct(cos, gold)
```

And the triplet objective:

```python
class TripletObjective(nn.Module):
    def __init__(self, encoder, margin=1.0):
        super().__init__()
        self.encoder, self.margin = encoder, margin

    def forward(self, anc, pos, neg):
        a = embed(self.encoder, anc.input_ids, anc.attention_mask)
        p = embed(self.encoder, pos.input_ids, pos.attention_mask)
        n = embed(self.encoder, neg.input_ids, neg.attention_mask)
        d_ap = (a - p).norm(p=2, dim=1)
        d_an = (a - n).norm(p=2, dim=1)
        return F.relu(d_ap - d_an + self.margin).mean()         # anchor at least `margin` closer to pos
```

Inference is the same for all of them: encode once, mean-pool, compare by cosine.

```python
@torch.no_grad()
def encode(encoder, batches):
    return torch.cat([embed(encoder, b.input_ids, b.attention_mask) for b in batches])

def most_similar(query_vec, bank):                              # bank: [N, H] precomputed
    return F.cosine_similarity(query_vec.unsqueeze(0), bank).argmax()
```

Training is short — fine-tune the pre-trained encoder for about one epoch on NLI with the softmax objective, batch size 16, Adam at 2e-5 with a linear warmup over the first 10% of steps, mean pooling. Then optionally continue on STS with the cosine-regression objective.

So the chain: the cross-encoder is accurate but doesn't factorize, so collection-level similarity is quadratic-in-pairs and infeasible; naive single-sentence vectors from the encoder are worse than averaged GloVe under cosine because the pre-training never shaped the space for an equal-weight angular metric; so fine-tune the encoder in a siamese (tied-weight) structure that yields comparable per-sentence vectors, with mean pooling, training the *geometry* via a `[u; v; |u−v|]` softmax on NLI (the `|u−v|` term shaping distances) or MSE-on-cosine on graded STS data or a margin triplet loss — and at inference discard the heads and just compare precomputed embeddings by cosine, turning 65 hours of pairwise encoding into seconds of encoding plus near-free vector comparison.
