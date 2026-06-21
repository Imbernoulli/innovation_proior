We want a classifier that works at test time from almost nothing: a support set of $N$ previously unseen classes with only $k$ labeled examples each, $k=1$ in the one-shot case, from which the learner must classify fresh queries into those $N$ labels. Random guessing is $1/N$, so this is not just recognition with fewer examples — the labels themselves did not exist during training, and the test-time interface only ever hands us a support set. A useful system therefore cannot afford to add output units, run gradient descent on the support examples, or fine-tune the backbone for each episode. What we need is a reusable procedure that takes a variable-size labeled set, compares a query against it, and returns a distribution over exactly the labels present in that set.

The two obvious families each fail in a structural way. Parametric deep classifiers store their knowledge in output weights that move slowly through many gradient steps; one support image gives almost no stable gradient signal, and aggressive fine-tuning on a handful of examples overfits and forgets — and in any case the test condition forbids fine-tuning. Non-parametric classifiers — nearest-neighbor, kernel, locally weighted — have the opposite profile: they assimilate a new example instantly because the stored examples *are* the model, and they handle a changing label set natively. But all of their intelligence sits in the distance function, and a fixed metric in pixel space, or features borrowed from a base-class softmax, is only accidentally aligned with the comparison the new task demands. The right move is to keep the instant-assimilation behavior of the non-parametric approach while *learning* the space and the comparison rule, and to make that comparison differentiable, task-shaped, and indifferent to which labels happen to appear in an episode.

I propose Matching Networks. The idea is to turn the support set itself into the classifier through a differentiable attention vote. For a query $\hat{x}$ and support set $S = \{(x_i, y_i)\}_{i=1}^k$ with one-hot labels $y_i$, the prediction is a weighted sum of the support labels,
$$\hat{y} = \sum_i a(\hat{x}, x_i)\, y_i, \qquad a_i = \frac{\exp\!\big(c(f(\hat{x},S),\, g(x_i,S))\big)}{\sum_j \exp\!\big(c(f(\hat{x},S),\, g(x_j,S))\big)}.$$
Because the $a_i$ are nonnegative and sum to one and each $y_i$ is one-hot, $\hat{y}$ is automatically a distribution over the episode's labels, and the probability of class $n$ is $P(y=n \mid \hat{x}, S) = \sum_{i:\, y_i = e_n} a_i$ — the *total* attention paid to support examples carrying that label. In one-shot this is a single weight per class; in multi-shot it is the summed attention mass across all support examples of that class. The same parameterization handles every shot count without ever adding a learned per-class output. This is exactly the memory-read interpretation: the support labels are values bound to support keys, and the query reads out a convex combination of those values.

The soft attention is the load-bearing borrowing from Neighbourhood Components Analysis. Hard nearest-neighbor classification is discontinuous in the metric, but the stochastic soft neighbor is smooth, and the sign matters — NCA uses *negative* squared distance so that nearer points receive larger mass, $p_{ij} \propto \exp(-\|Ax_i - Ax_j\|^2)$. I take that soft-neighbor trick and move it off a fixed training set onto the episode's own support set, and I replace the fixed linear map $A$ with deep embeddings $f$ and $g$ so we are no longer confined to a Mahalanobis metric. The similarity $c$ is cosine, which removes vector norm from the comparison so the vote depends on direction in feature space rather than on whichever embedding has the largest magnitude. One precision point I have to keep when I implement it: the reference normalizes the support embeddings but leaves the query embeddings unnormalized before the final matrix multiply, so the computed score is $f^\top(g/\|g\|) = \|f\|\cos(f,g)$ — a scaled cosine with a query-dependent softmax temperature, not literal symmetric cosine, and it should not be described as such in code.

The training objective has to match the test condition, or the embeddings revert to borrowed features. So I train on episodes shaped exactly like evaluation: draw a label set $L$ from a task distribution, draw a support set $S$ and a query batch $B$ from those labels, and maximize
$$\theta = \arg\max_\theta \ \mathbb{E}_{L \sim T}\ \mathbb{E}_{S \sim L,\, B \sim L} \sum_{(x,y)\in B} \log P_\theta(y \mid x, S).$$
The sign stays consistent: the derivation is an argmax of log-probability, and the implementation minimizes the negative log-likelihood on the query labels. Because the forward pass already emits log-probabilities (the vote is a genuine distribution, then logged), the matching loss is `NLLLoss`, not cross-entropy on logits.

At this point the classifier already works, but the embeddings are myopic: a support vector computed in isolation cannot react to another support example of a different class sitting nearby, and a query vector computed in isolation cannot emphasize the features that separate *this* episode's candidate classes. So I make both embeddings conditional on the support set — full context embeddings. For the support side I run a bidirectional LSTM over the raw support features and add a residual,
$$g(x_i, S) = \overrightarrow{h}_i + \overleftarrow{h}_i + g'(x_i).$$
The forward pass reads from the start up to $i$, the backward pass from the end down to $i$, so their sum gives each position context from the whole (orderless) support sequence, and the skip connection means the recurrence only has to learn a contextual *correction* while the raw feature is preserved. For the query side I use an attention LSTM that reads the support memory $K$ times. At step $k$ it carries a hidden state and a readout, computes
$$\hat{h}_k, c_k = \mathrm{LSTM}\big(f'(\hat{x}),\, [h_{k-1}, r_{k-1}],\, c_{k-1}\big), \quad h_k = \hat{h}_k + f'(\hat{x}), \quad r_{k-1} = \sum_i \mathrm{softmax}_i\big(h_{k-1}^\top g(x_i, S)\big)\, g(x_i, S),$$
and after $K$ reads sets $f(\hat{x}, S) = h_K$. The softmax inside this loop is over support index $i$ and produces the refinement readout; it is a separate mechanism from the final class vote. The original treats $K$ as a fixed processing depth; this port simply loops once per support example, while the Facebook reference takes $K$ as a constructor argument.

The full tensor path then reads cleanly: contextualize the support features with the bidirectional LSTM, one-hot the support labels, contextualize each query against that support memory with the attention LSTM, normalize the support embeddings, form the support-normalized scaled-cosine score matrix, softmax over support examples, multiply by the one-hot support-label matrix to aggregate per-class attention, and take a log before NLL. The matrix multiply by the labels *is* the multi-shot class aggregation; the log is valid precisely because the attention vote is already a distribution over classes. A small numerical floor of $1\text{e-}6$ is added before the log to avoid $\log 0$ when a class receives no mass — it slightly breaks exact normalization but is the safe choice. What makes the whole design work is the separation of duties: the learned parameters define the comparison machinery (the embeddings, the contextualizers, the metric), while the episode's support set supplies the actual class-specific memory, so installing a new classifier is just a matter of swapping the support set.

```python
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MatchingNetworks(FewShotClassifier):
    """Support-conditioned non-parametric classifier with full context embeddings.

    The forward pass returns log-probabilities, so train with NLL loss.
    This mirrors the local EasyFSL-style port and the cited Facebook reference:
    contextualize support with a bidirectional LSTM, contextualize queries with
    an attention LSTM, then vote over support labels using support-normalized
    scaled-cosine scores.
    """

    LR_OVERRIDE = 1e-3

    def __init__(self, *args, feature_dimension: int, **kwargs):
        super().__init__(*args, use_softmax=False, **kwargs)
        self.feature_dimension = feature_dimension
        self.support_features_encoder = nn.LSTM(
            input_size=feature_dimension,
            hidden_size=feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.query_features_encoding_cell = nn.LSTMCell(
            feature_dimension * 2,
            feature_dimension,
        )
        self.softmax = nn.Softmax(dim=1)
        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        self.contextualized_support_features = self.encode_support_features(
            support_features
        )
        self.one_hot_support_labels = F.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        contextualized_query_features = self.encode_query_features(query_features)

        # Reference behavior: normalize support embeddings only.
        support = F.normalize(self.contextualized_support_features, dim=1)
        attention = self.softmax(contextualized_query_features.mm(support.T))

        # Aggregate support-item attention into class probabilities.
        class_probabilities = attention.mm(self.one_hot_support_labels)
        return (class_probabilities + 1e-6).log()

    def encode_support_features(self, support_features: Tensor) -> Tensor:
        hidden_state = self.support_features_encoder(
            support_features.unsqueeze(0)
        )[0].squeeze(0)
        return (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        # Local port choice: K equals the number of support examples.
        for _ in range(len(self.contextualized_support_features)):
            support_attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = support_attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), dim=1)
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input,
                (hidden_state, cell_state),
            )
            hidden_state = hidden_state + query_features

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.nll_loss(scores, labels)
```
