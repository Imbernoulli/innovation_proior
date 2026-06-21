A pre-trained Transformer is the strongest tool available for sentence-pair work — semantic textual similarity, natural language inference, paraphrase detection — but it earns that accuracy in a way that does not scale. It is a cross-encoder: you concatenate the two sentences with a `[SEP]` separator, run full multi-head self-attention across all tokens of both sentences through every layer, and read a prediction off the joint representation. The two sentences interact through attention at each layer, which is exactly why the score is good and exactly why it does not factorize. There is no per-sentence object you can compute once and reuse, because the prediction is only defined on a *pair*, jointly encoded. So the moment the task becomes "find the most similar pair, or the nearest neighbor, in a collection," the cost explodes: for $n = 10{,}000$ sentences there are $n(n-1)/2 = 49{,}995{,}000$ pairs, each one a full forward pass — on the order of $65$ hours on a single V100. Clustering, large-scale semantic search, and deduplication are simply out of reach.

What I actually want is obvious: a fixed-size vector for a *single* sentence, computed once, such that semantically similar sentences land close under a cheap metric like cosine. Then a collection of $10{,}000$ sentences is $10{,}000$ encodes — seconds — plus near-free vector comparisons, and with an index, nearest-neighbor search is milliseconds. The tempting shortcut is to push one sentence through the pre-trained encoder and read off a vector: the classification-token output, or the mean of the token outputs, as the popular encoder-as-a-service tooling offers. The diagnostic kills this. When the cosine similarity of those naive vectors is correlated against human STS judgments by Spearman rank correlation, averaging the encoder's token outputs lands *below* averaging plain GloVe word vectors, and the classification-token vector is worse still. Out of the box, the encoder maps sentences into a space whose *geometry* is wrong for cosine. The reason is precise: the encoder was pre-trained and then typically adapted by fine-tuning a classifier on top, and a classifier can re-weight and recombine dimensions however it likes. Cosine has no such freedom — it weights every dimension equally and sees only the angle. A representation can be perfectly informative for a downstream probe yet useless under cosine, because nothing in pre-training ever forced "semantically close" to mean "small angle" under an equal-weight metric. The failure is the equal-weight angular geometry specifically, so no better readout trick will save it; the cosine geometry has to be *trained in*. And it must be trained in starting from the pre-trained encoder, not from scratch, so the encoder's knowledge is not thrown away as the from-scratch InferSent and Universal Sentence Encoder approaches effectively do.

I propose Sentence-BERT (SBERT): fine-tune the pre-trained encoder inside a siamese (and, where the data demands it, triplet) structure, with a pooling step that collapses the token outputs into one fixed vector, under a loss that bends the *geometry* of the embedding space so that semantic closeness becomes small angle — and then at inference discard the training head entirely and compare precomputed embeddings by cosine. The siamese structure is load-bearing: each sentence in a pair passes through the *same* encoder with *tied* weights, so the two vectors necessarily live in one shared space and are directly comparable. If the two towers had independent parameters their vectors would not share a geometry, and a vector metric across them would be meaningless. For the pooling, the choices are the classification-token output, a coordinate-wise max-over-time, or the mean of the token outputs. The classification token has already failed as a standalone similarity vector in the naive diagnostic; max-pooling, as a coordinate-wise selector, can discard broad sentence-level evidence; mean pooling keeps every contextual token vector in the representation and parallels the surprisingly strong averaged-word-embedding baseline. So mean pooling is the default — averaged over the real tokens only, masking out padding so that sentence length and batch padding never perturb the vector — with CLS and MAX left as variants.

The objective is chosen by the supervision available, and there are three regimes. The data I have most of is NLI — SNLI plus MultiNLI, around a million pairs tagged entailment / contradiction / neutral. That is a three-way classification over a *pair*, not a graded score, so I cannot regress cosine directly; I have to turn the two sentence vectors $u$ and $v$ into features for a softmax classifier. The features are the two vectors themselves together with something that captures how they differ, so I form a $3n$-wide vector and feed it to a softmax over the $k$ labels, trained with cross-entropy,
$$o = \mathrm{softmax}\!\big(W_t\,[\,u\,;\,v\,;\,|u - v|\,]\big),$$
where $W_t$ is a trainable linear map from the $3n$-dimensional concatenation to $k$ labels. The element-wise absolute difference $|u - v|$ is the load-bearing term: per dimension it measures how far apart the two embeddings are, so including it pushes the classifier to make "different label" correspond to "large coordinate-wise difference," which is exactly the pressure that shapes the *distance* structure of the space — and distance structure is what cosine exploits at inference. I keep the raw $u$ and $v$ in the concatenation as well, rather than the difference alone, because the raw vectors let the classifier condition on *what* the sentences are and not merely on how they differ, which the asymmetry of entailment versus contradiction needs. I deliberately do *not* add the element-wise product $u \cdot v$ by default: it is another interaction term but it lacks the clean distance interpretation of $|u-v|$ and it adds another $n$ input dimensions, diluting the pressure I want concentrated on $[\,u\,;\,v\,;\,|u-v|\,]$. Crucially, every one of these concatenation features matters *only for training the classifier*. At inference $W_t$ is discarded outright and sentences are compared by the cosine of $u$ and $v$ directly; the classifier is scaffolding whose sole job is to bend the embedding space into shape.

When the data is graded similarity instead — STS pairs labeled on a continuous scale — I do not need a classifier at all, and I can train the target geometry *directly*. I compute the cosine similarity of $u$ and $v$ and regress it onto the gold similarity with mean-squared error. Since the STS labels arrive on a $0$–$5$ scale, I normalize them to $0$–$1$ before the loss ever sees them, so the numeric target is compatible with a cosine score. This is the most direct objective possible: make cosine track the desired similarity. And the two regimes compose — fine-tune first on NLI for broad semantic signal, then continue on STS for task-specific calibration. The third regime is triplet data — an anchor, a positive that should be close, and a negative that should be far, as in the Wikipedia-section dataset. There I use a triplet margin loss with Euclidean distance,
$$\max\!\big(\|s_a - s_p\| - \|s_a - s_n\| + \epsilon,\ 0\big),\qquad \epsilon = 1,$$
which forces the anchor to be at least a margin $\epsilon$ closer to the positive than to the negative; the margin demands a genuine gap rather than any mere ordering, and the relu zeroes out triplets that already satisfy the gap so they stop contributing gradient.

The inference path is identical across all three objectives and is deliberately trivial: encode each sentence once, mean-pool to get $u$, and compare two sentences by cosine $(u \cdot v)/(\|u\|\,\|v\|)$. For a collection, precompute all embeddings once, after which retrieval and clustering are plain vector math, optionally accelerated by an approximate-nearest-neighbor index. No pair is ever fed through the encoder jointly again, which is precisely what turns $65$ hours of pairwise encoding into seconds of encoding plus near-free comparison. Training itself is short: fine-tune the pre-trained encoder for about one epoch on SNLI+MultiNLI with the softmax objective, batch size $16$, Adam at learning rate $2\times10^{-5}$ with a linear warmup over the first $10\%$ of steps and linear decay thereafter, mean pooling throughout, optionally continuing on STS with the cosine-regression objective.

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F

def mean_pool(token_embeddings, attention_mask):
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed(encoder, batch):
    tok = encoder(batch.input_ids, batch.attention_mask)   # [B, L, H]
    return mean_pool(tok, batch.attention_mask)            # [B, H]

class SoftmaxObjective(nn.Module):
    def __init__(self, encoder, sent_dim, num_labels=3,
                 use_rep=True, use_diff=True, use_mul=False):
        super().__init__()
        self.encoder = encoder                              # tied weights for both sentences
        self.use_rep, self.use_diff, self.use_mul = use_rep, use_diff, use_mul
        n = (2 if use_rep else 0) + (1 if use_diff else 0) + (1 if use_mul else 0)
        self.classifier = nn.Linear(n * sent_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
    def forward(self, a, b, labels):
        u, v = embed(self.encoder, a), embed(self.encoder, b)
        feats = []
        if self.use_rep:  feats += [u, v]
        if self.use_diff: feats += [torch.abs(u - v)]
        if self.use_mul:  feats += [u * v]
        return self.loss_fct(self.classifier(torch.cat(feats, 1)), labels.view(-1))

class CosineRegressionObjective(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder; self.loss_fct = nn.MSELoss()
    def forward(self, a, b, target):                        # STS target normalized to 0..1
        u, v = embed(self.encoder, a), embed(self.encoder, b)
        return self.loss_fct(F.cosine_similarity(u, v), target.view(-1).float())

class TripletObjective(nn.Module):
    def __init__(self, encoder, margin=1.0):
        super().__init__(); self.encoder = encoder; self.margin = margin
    def forward(self, anc, pos, neg):
        a, p, n = (embed(self.encoder, anc), embed(self.encoder, pos), embed(self.encoder, neg))
        d_ap = F.pairwise_distance(a, p, p=2)
        d_an = F.pairwise_distance(a, n, p=2)
        return F.relu(d_ap - d_an + self.margin).mean()

# --- training ---
def linear_warmup_decay(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

def train(objective, data, epochs=1):
    total_steps = epochs * len(data)
    warmup_steps = math.ceil(0.1 * total_steps)
    opt = torch.optim.Adam(objective.parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )
    for _ in range(epochs):
        for batch in data:
            loss = objective(*batch)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()

# --- inference: encode once, compare by cosine ---
@torch.no_grad()
def encode_bank(encoder, batches):
    return torch.cat([embed(encoder, b) for b in batches])         # [N, H], precomputed once

def most_similar(query_vec, bank):
    return F.cosine_similarity(query_vec.unsqueeze(0), bank).argmax()
```
