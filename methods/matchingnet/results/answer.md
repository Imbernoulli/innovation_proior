# Matching Networks, distilled

Matching Networks map a small labeled support set to a non-parametric classifier with no
fine-tuning at test time. A deep backbone embeds support and query images; each query is
classified by a softmax-over-cosine-similarity weighted vote over the support labels. The
embeddings are made *fully conditional* on the whole support set (a bidirectional LSTM for
support, an attention LSTM for the query), and the whole model is trained *episodically* so
that the training task matches the one-shot test task.

## Problem it solves

N-way k-shot classification: given a support set of N classes never seen during training, with
k labeled examples each (k as small as 1), classify a query into one of the N classes — with no
weight updates at test time. Parametric deep nets need many gradient steps to absorb a class;
non-parametric kNN absorbs instantly but has no learned metric. Matching Networks get both:
deep learned features feeding a non-parametric, instantly-reconfigurable classifier.

## Key idea

Predict the query label as an attention-weighted combination of the support labels:

```
y_hat = sum_{i=1}^k  a(x_hat, x_i) * y_i        # y_i one-hot -> y_hat is a distribution over classes
a(x_hat, x_i) = softmax_i( c( f(x_hat), g(x_i) ) ),   c = cosine similarity
```

- With `a` a kernel this is **kernel density estimation**; zeroing all but the nearest support
  points recovers **(k−b)-nearest-neighbours**; reading the `y_i` as memories bound to the
  `x_i` makes it an **associative memory** that grows with the support set. So a freshly
  sampled support set *is* a new classifier with no retraining.
- **Cosine** (not raw dot or Euclidean): embeddings have uncontrolled magnitude, so compare
  *direction* — an atypically long support embedding can't dominate the softmax; cosine is
  bounded in [−1, 1], giving a well-behaved softmax scale.
- The **softmax** turns the hard nearest-neighbour into a smooth, convex, differentiable vote
  (the soft-neighbour trick of NCA, applied over the support set as a unit), so the whole model
  trains end to end.
- Output is **log-probabilities**, so the loss is **negative log-likelihood**, not
  cross-entropy on logits (that would softmax an already-normalized distribution twice).

## Training: episodic meta-learning ("test and train conditions must match")

```
theta = argmax_theta  E_{L ~ T} [ E_{S ~ L, B ~ L} [ sum_{(x,y) in B} log P_theta(y | x, S) ] ]
```

Sample a label set `L` from a task distribution `T` (e.g. 5 classes), sample a support set `S`
and a query batch `B` from `L`, and train the model to predict `B`'s labels conditioned on `S`,
resampling the classes each episode. This explicitly learns *to learn from a support set*; at
test time a support set `S'` of novel classes needs no fine-tuning. (Works while the test task
distribution resembles the training one; degrades when it diverges, e.g. broad-sampled training
vs. fine-grained test.)

## Full Context Embeddings (FCE)

Plain `g(x_i)` embeds each support example independently ("myopic"); and `S` should change how
the query is embedded. Make both embeddings functions of the whole support set.

Support, bidirectional LSTM with a skip connection:

```
h_fwd_i, c_fwd_i = LSTM(g'(x_i), h_fwd_{i-1}, c_fwd_{i-1})
h_bwd_i, c_bwd_i = LSTM(g'(x_i), h_bwd_{i+1}, c_bwd_{i+1})        # backward starts at i=|S|
g(x_i, S) = h_fwd_i + h_bwd_i + g'(x_i)
```

Query, attention LSTM (a Process block) with K read steps and a skip connection:

```
h_hat_k, c_k = LSTM( f'(x_hat), [h_{k-1}, r_{k-1}], c_{k-1} )
h_k          = h_hat_k + f'(x_hat)
r_{k-1}      = sum_i  a(h_{k-1}, g(x_i)) * g(x_i)
a(h_{k-1}, g(x_i)) = softmax_i( h_{k-1}^T g(x_i) )
f(x_hat, S)  = attLSTM(f'(x_hat), g(S), K) = h_K
```

The skip connections anchor each contextual embedding to its raw feature, so the LSTMs learn
only the context-dependent *correction*. K is a depth knob for the query attention; a natural
implementation choice is K = |S|.

## Why each choice

- **Non-parametric attention vote** — a new support set defines a new classifier with no weight
  update; the support set is the memory.
- **Cosine + softmax** — compare directions on a bounded scale; smooth, differentiable, convex
  weights -> a probability distribution per query.
- **Episodic training** — the only way to make features serve *comparison against a support
  set on unseen classes* is to train under exactly that task.
- **FCE bidirectional LSTM (support)** — lets near-colliding support examples adjust given each
  other; skip keeps it no worse than the myopic embedding.
- **attLSTM with K steps (query)** — conditions the query embedding on the support set with
  iterated ("deep") attention that can sharpen and selectively ignore support elements.
- **Adam @ 1e-3 with gradient clipping** — the bidirectional support LSTM and the multi-step
  query attention loop destabilize under a large plain-SGD rate and collapse the output softmax
  toward uniform; the smaller adaptive step keeps them trainable.

## Working code

Filling the comparison-rule slot of the episodic few-shot harness. The forward path: backbone
features -> contextualize support (bidirectional LSTM) -> contextualize query (attention LSTM)
-> cosine-softmax vote over support labels -> log-probabilities; trained with NLL.

```python
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MatchingNetworks(FewShotClassifier):
    """Matching Networks. Contextualize support and query
    features with LSTMs (full context embeddings), then classify each query by a
    cosine-similarity-weighted vote over support labels. Output is log-probabilities,
    so train with NLLLoss."""

    LR_OVERRIDE = 1e-3   # Adam @ 1e-3 (+ grad clip); SGD destabilizes the LSTMs.

    def __init__(self, *args, feature_dimension: int, **kwargs):
        super().__init__(*args, use_softmax=False, **kwargs)
        self.feature_dimension = feature_dimension

        # g(x_i, S): bidirectional LSTM over the support set.
        self.support_features_encoder = nn.LSTM(
            input_size=feature_dimension,
            hidden_size=feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # f(x_hat, S): attLSTM cell; input width 2d = [raw query feature ; read-out].
        self.query_features_encoding_cell = nn.LSTMCell(feature_dimension * 2, feature_dimension)
        self.softmax = nn.Softmax(dim=1)

        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)          # g'(x_i): (|S|, d)
        self.contextualized_support_features = self.encode_support_features(support_features)
        self.one_hot_support_labels = F.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)             # f'(x_hat)
        contextualized_query_features = self.encode_query_features(query_features)

        # softmax over cosine similarity from each query to each support example.
        # Normalize support only -> keep query vectors "sharp" so softmax stays peaked.
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                F.normalize(self.contextualized_support_features, dim=1).T
            )
        )
        # y_hat = sum_i a(x_hat, x_i) y_i ; log for NLL ; +1e-6 floors log(0).
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def encode_support_features(self, support_features: Tensor) -> Tensor:
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[0].squeeze(0)
        # g(x_i, S) = forward_h + backward_h + g'(x_i)   (skip connection)
        return (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features                       # h_0 = raw query feature
        cell_state = torch.zeros_like(query_features)
        # K = |S| processing steps.
        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(hidden_state.mm(self.contextualized_support_features.T))
            read_out = attention.mm(self.contextualized_support_features)            # r_{k-1}
            lstm_input = torch.cat((query_features, read_out), 1)                    # [f'(x_hat); r]
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features    # h_k = h_hat_k + f'(x_hat)
        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.nll_loss(scores, labels)                  # scores are log-probabilities
```
</content>
