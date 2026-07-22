Let me start from the wall I keep hitting. I have a pre-trained Transformer that is the best thing going for sentence-pair tasks — semantic similarity, NLI, paraphrase — when I feed it both sentences at once, joined by `[SEP]`, and let full self-attention chew across all the tokens of both. The attention between the two sentences at every layer is precisely why it's so accurate. But it means there is no per-sentence object I can compute once and reuse: the prediction only exists for a *pair*, jointly encoded. So if I want the most similar pair in a collection of 10,000 sentences, I have to encode all ~50 million pairs — about 65 hours on one GPU. Clustering, semantic search, dedup: all infeasible. The cost is quadratic in pairs, and the unit of computation is a full forward pass.

What I actually want is obvious: a *single-sentence* fixed-size vector, computed once per sentence, such that semantically similar sentences are close under a cheap metric like cosine. Then a collection of 10,000 sentences is 10,000 encodes (~seconds) plus near-free vector comparisons, and with an index, nearest-neighbor search is milliseconds. The expensive joint encoding gets replaced by precomputable vectors.

The tempting workaround is to push one sentence through the pre-trained encoder and read off a vector — the classification-token output, or the mean of the token outputs; people already do this. The diagnostic I have to take seriously says that this is not enough: when cosine similarity of those vectors is compared with human STS judgments by Spearman correlation, averaging the encoder's token outputs lands *below* averaging plain GloVe word vectors, and the classification-token vector is worse still. So the encoder, out of the box, maps sentences into a space whose *geometry* is wrong for cosine similarity. The information is in there, but the angles between sentence vectors don't track meaning.

Why would that be? The encoder was pre-trained for token- and sentence-pair use, then usually adapted by fine-tuning a classifier on top that can re-weight and recombine dimensions however it likes. Cosine similarity has no such freedom — it weights every dimension equally and only sees the angle. So a representation can be perfectly good for a downstream classifier and useless under cosine, because nothing ever forced "semantically close" to mean "small angle." Let me make that concrete enough to trust, because it's the assumption everything below rests on. Take two embeddings that point the same direction but differ in length, `u = (1,1)` and `v = (3,3)`. Their cosine is `(1·3+1·3)/(√2·√18) = 6/6 = 1` — cosine calls them identical, because it normalizes away magnitude. A downstream classifier with free weights could still tell them apart (one coordinate is three times larger), but cosine structurally cannot. That's the trap: the very thing pre-training is free to encode in magnitude, cosine throws away. The same naive vectors can still look usable to a logistic-regression probe, because that probe can learn which dimensions to emphasize and which to ignore. The failure is specifically the equal-weight cosine geometry. So a better readout trick won't save it; the cosine geometry has to be *put into the embedding by training*.

That reframes the problem: I need to fine-tune the encoder so that a single-sentence embedding sits in a space where small angle = semantic closeness. And I want to do this *without* throwing away the pre-trained knowledge — start from the pre-trained encoder, not from scratch.

For that I need a training structure that produces *comparable* vectors for two sentences and a loss defined on those vectors. The classical answer is a siamese network: pass each sentence through the *same* encoder with *tied* weights, so both land in one shared space and are directly comparable, then define the loss on the two resulting vectors. The tied weights are not a detail — if the two towers had separate parameters, "small angle between `u` and `v`" would be comparing coordinates that mean different things in the two spaces, and the cosine I compute at inference would be meaningless. One encoder, one space, comparable vectors. So: shared encoder, pooling to a fixed vector, loss on the pooled vectors.

I still need to turn a variable number of token outputs into one fixed vector. The available choices are the classification-token output, the mean of all token output vectors, or a max-over-time. The classification token has already failed as a standalone similarity vector in the naive diagnostic, so it's a poor default. Mean pooling keeps every contextual token vector in the sentence representation and parallels the averaged-word-embedding baseline that remains surprisingly strong. Max pooling is worth keeping as a variant, but as a coordinatewise selector it can throw away broad sentence evidence in favor of a few loud tokens. So I'll make MEAN the default and leave CLS and MAX as switches.

Before I build anything on mean pooling I want to be sure I can write it so that padding cannot leak into the vector — batches pad to the longest element, and if pad positions bleed into the mean, the *same sentence* would get a different embedding depending on what it was batched with, which would quietly poison the geometry I'm trying to train. So the mean has to be over real tokens only, weighted by the attention mask. Let me trace the arithmetic on a tiny case before trusting it. One sentence, two real tokens and one pad, hidden size 2: token outputs `[[1,2],[3,0],[100,100]]` with mask `[1,1,0]` (that last `[100,100]` is whatever garbage sits in the pad slot). Multiply each token by its mask: the pad row zeroes out, leaving `[[1,2],[3,0],[0,0]]`. Sum over the length axis: `(1+3+0, 2+0+0) = (4,2)`. Divide by the count of real tokens, which is `1+1+0 = 2`: `(2,1)`. That's exactly the mean of the two real tokens `(1,2)` and `(3,0)`, and the pad's `[100,100]` left no trace. Good — masking works and is independent of padding, which is the property I needed.

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

Now the loss, which has to match the supervision I actually have.

The data I have most of is NLI — SNLI plus MultiNLI, a million labeled pairs tagged entailment / contradiction / neutral. That's a 3-way classification over a *pair*, not a similarity score, so I can't regress cosine directly. I need to turn the two sentence vectors `u` and `v` into features for a softmax classifier. The natural features are the two vectors themselves and something that captures how they *differ*. So form a feature vector and feed it to a softmax over the 3 labels, training with cross-entropy:
`o = softmax(W_t([u ; v ; |u − v|]))`, where `W_t` is a trainable linear map from the 3n-dimensional concatenation to `k` labels.

Why include `|u − v|`? The element-wise absolute difference measures, per dimension, how far apart the two embeddings are; including it lets the classifier make "different label" correspond to "large coordinate-wise difference," which puts pressure on the *distance* structure of the space. But I should be honest with myself about what kind of distance that is, because it's tempting to wave my hands and say "and therefore cosine works." It does *not* directly shape cosine. `|u − v|` is the coordinate-wise gap, the ingredient of *Euclidean* distance — and Euclidean distance and cosine are not the same geometry. The very example from before makes this sharp: `u=(1,1)`, `v=(3,3)` have cosine `1` (cosine calls them identical) but `|u−v| = (2,2)`, a large Euclidean gap. So an objective that pushes `|u−v|` apart is pushing Euclidean distance, which cosine partly ignores. Why do I still expect this to help cosine at inference? Because the cross-entropy isn't free to push only magnitude: separating labels by enlarging `|u−v|` while the encoder also has to keep entailed pairs *together* tends to spread the classes apart in direction as well as length, and once the classes occupy different directions, cosine sees it too. That's a plausible mechanism, not a guarantee — I'd want to confirm it empirically by checking that cosine STS correlation actually rises after NLI fine-tuning, rather than assuming the `|u−v|` term hands me cosine for free. I'll treat "train Euclidean-flavored separation, read off cosine" as a hypothesis to be measured, not an identity.

Why concatenate `u` and `v` themselves too, rather than only the difference? The raw vectors let the classifier condition on *what* the sentences are, not just how they differ, which the entailment/contradiction asymmetry needs — entailment is directional, and `|u−v|` alone is symmetric and so blind to direction.

Should I throw in more interaction terms — say the element-wise product `u*v`, as the from-scratch siamese baselines do, giving `[u ; v ; |u−v| ; u*v]`? The difference is the term with the cleanest distance reading; the product is another interaction but without that interpretation, and it adds another n input dimensions to the classifier. I have no a-priori reason it helps and a concrete cost (more parameters, more to overfit), so I'll keep it off by default and behind a switch rather than spend the capacity speculatively. Crucially, all of these concatenation features matter *only for training the classifier*: at inference I discard `W_t` entirely and compare sentences by cosine of `u` and `v` directly. The classifier is scaffolding to bend the embedding space into shape, then thrown away.

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
        return self.loss_fct(logits, labels.view(-1))
```

The other regime is when I have actual similarity scores, like STS pairs labeled on a graded scale. Then I don't need a classifier at all — I can train the geometry I want *directly*, with no Euclidean-vs-cosine gap to worry about. Compute the cosine similarity of `u` and `v` and regress it onto the gold similarity with mean-squared error. There's a units check to do here: cosine lives in `[-1, 1]`, and the STS gold labels arrive on a `0`-to-`5` scale, so if I fed the loss raw labels against a cosine I'd be regressing a `[-1,1]` quantity toward targets up to `5` — the loss would chase an unreachable target and mostly just push every cosine toward its ceiling. Normalizing the labels to `0`-to-`1` first puts the target inside cosine's reachable range (the upper half of it), so the MSE is comparing comparable numbers. With that normalization in place this is the most direct possible objective: make cosine track the target similarity, end to end, with nothing thrown away at inference. So: classification with `[u;v;|u−v|]` cross-entropy when the data is categorical (NLI), regression with MSE on cosine when the data is graded (STS). And I can do both in sequence — first fine-tune on NLI for broad semantic signal, then on STS to calibrate cosine to the actual scale — which is appealing precisely because the NLI stage only trains cosine *indirectly* and the STS stage closes that gap directly.

```python
class CosineRegressionObjective(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_fct = nn.MSELoss()

    def forward(self, a, b, target):
        u = embed(self.encoder, a.input_ids, a.attention_mask)
        v = embed(self.encoder, b.input_ids, b.attention_mask)
        cos = F.cosine_similarity(u, v)
        return self.loss_fct(cos, target.view(-1).float())      # STS labels normalized to 0..1
```

A third structure for when the data comes as triplets — an anchor, a positive (should be close), a negative (should be far), as in the Wikipedia-section dataset. Then a triplet margin loss: make the anchor closer to the positive than to the negative by at least a margin `ε`:
`max( ‖s_a − s_p‖ − ‖s_a − s_n‖ + ε, 0 )`,
with `‖·‖` Euclidean distance and a margin `ε` (I'll use `ε = 1`). I want to be sure the relu and the margin behave the way I'm claiming, so let me run two triplets by hand with `ε = 1`. First, a triplet whose ordering is already correct but barely so: `d_ap = 0.3`, `d_an = 0.8`. Then `d_ap − d_an + ε = 0.3 − 0.8 + 1 = 0.5`, and relu leaves `0.5` — still a positive loss, even though the positive is already nearer, because the gap `0.5` is smaller than the margin `1`. So the margin does what I want: ordering alone isn't enough, the gap has to reach `ε`, and triplets that are merely-correct keep producing gradient that widens the gap. Second, a triplet where the negative is genuinely far: `d_ap = 0.3`, `d_an = 5`. Then `0.3 − 5 + 1 = −4.7`, relu clamps it to `0` — no loss, no gradient. So once a triplet already clears the margin it drops out and stops pulling, which is exactly the behavior I want: effort concentrates on the triplets still violating the gap. Both cases match the design.

```python
class TripletObjective(nn.Module):
    def __init__(self, encoder, margin=1.0):
        super().__init__()
        self.encoder, self.margin = encoder, margin

    def forward(self, anc, pos, neg):
        a = embed(self.encoder, anc.input_ids, anc.attention_mask)
        p = embed(self.encoder, pos.input_ids, pos.attention_mask)
        n = embed(self.encoder, neg.input_ids, neg.attention_mask)
        d_ap = F.pairwise_distance(a, p, p=2)
        d_an = F.pairwise_distance(a, n, p=2)
        return F.relu(d_ap - d_an + self.margin).mean()         # anchor at least `margin` closer to pos
```

Now the inference path, which has to be the same regardless of which loss trained the encoder: encode each sentence once → mean-pool → get `u`. Compare two sentences by cosine `(u·v)/(‖u‖‖v‖)`. For a collection: precompute all embeddings, then retrieval/clustering is plain vector math, optionally with an approximate-nearest-neighbor index. No pair ever goes through the encoder jointly again. Let me trace the retrieval on a tiny bank to be sure the cheap cosine comparison actually picks the right neighbor when there's a near-tie. Bank of three unit-ish vectors `[[1,0],[0,1],[0.9,0.1]]`, query `[1, 0.05]`. Cosines come out `0.9988`, `0.0499`, `0.9982`: the query is nearly parallel to both the first and third bank vectors, but it leans very slightly more toward `[1,0]`, and `argmax` returns index `0` over the close runner-up at index `2`. So the comparison is a single vectorized cosine plus an argmax — no forward pass, no pair encoding — and it resolves even a tight neighbor correctly. That is the whole payoff: the 65-hour pairwise cost collapses to one encode per sentence plus this.

```python
@torch.no_grad()
def encode(encoder, batches):
    return torch.cat([embed(encoder, b.input_ids, b.attention_mask) for b in batches])

def most_similar(query_vec, bank):                              # bank: [N, H] precomputed
    return F.cosine_similarity(query_vec.unsqueeze(0), bank).argmax()
```
