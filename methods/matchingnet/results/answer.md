# Matching Networks

Matching Networks define a non-parametric classifier from a labeled support set. Given
`S = {(x_i, y_i)}_{i=1}^k`, with one-hot labels `y_i`, the query prediction is

```text
y_hat = sum_i a(x_hat, x_i) y_i
a_i   = exp(c(f(x_hat, S), g(x_i, S))) / sum_j exp(c(f(x_hat, S), g(x_j, S)))
```

For class `n`, this gives `P(y=n | x_hat, S) = sum_{i: y_i=e_n} a_i`. In one-shot this is
one attention weight per class; in multi-shot it sums the attention mass across all support
examples of the same class.

The paper-level similarity `c` is cosine. The reference PyTorch implementation used here
normalizes the support embeddings but not the query embeddings before the final vote, so the
implemented score is a scaled cosine, `f^T (g / ||g||) = ||f|| cos(f, g)`. That matches the
Facebook reference and the local port; it should not be described as symmetric cosine in code.

## Training Objective

Train on episodes shaped like evaluation:

```text
theta = argmax_theta E_{L ~ T} E_{S ~ L, B ~ L}
        sum_{(x,y) in B} log P_theta(y | x, S)
```

Implementation minimizes negative log-likelihood on the query labels. The output is already
log-probabilities, not logits, so the matching loss is `NLLLoss`, not cross-entropy on logits.

## Full Context Embeddings

Support context:

```text
g(x_i, S) = h_forward_i + h_backward_i + g'(x_i)
```

where the forward and backward states come from a bidirectional LSTM over support features.

Query context:

```text
h_hat_k, c_k = LSTM(f'(x_hat), [h_{k-1}, r_{k-1}], c_{k-1})
h_k          = h_hat_k + f'(x_hat)
r_{k-1}      = sum_i softmax_i(h_{k-1}^T g(x_i, S)) g(x_i, S)
f(x_hat, S)  = h_K
```

The paper treats `K` as a fixed processing depth. The local EasyFSL-style port loops
`len(support)` times; the fetched Facebook reference takes `K` as a constructor argument.

## Reference-Faithful Code

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
