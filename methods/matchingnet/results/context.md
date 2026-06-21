# Context: one-shot classification before the method

## Research Question

The target setting is extreme low-data classification. At test time, a learner is given a
small labeled support set containing `N` previously unseen classes and only `k` labeled
examples per class, then must classify new query examples into those `N` classes. In the
one-shot case `k = 1`; in the broader few-shot case `k` is still small. Random guessing is
`1/N`, so the problem is not merely recognizing familiar labels with fewer examples. The
learner must create a usable classifier for labels that did not exist during training.

The central question is how to build a classifier that generalizes to new classes given only
a small support set at test time, without modifying the model's parameters for each new task.

## Load-Bearing Prior Art

Metric learning supplies the first key ingredient. Neighbourhood Components Analysis replaces
hard leave-one-out nearest-neighbor decisions with stochastic neighbor assignments:

```text
p_ij = exp(-||A x_i - A x_j||^2) / sum_{l != i} exp(-||A x_i - A x_l||^2),  p_ii = 0
p_i  = sum_{j: c_j = c_i} p_ij
```

The objective maximizes `sum_i p_i`, the expected number of correctly classified training
points. This turns a discontinuous nearest-neighbor rule into a smooth objective for the
metric.

Content-based attention supplies the second ingredient. Sequence-to-sequence attention scores
a query state against each memory entry, softmax-normalizes those scores, and reads a weighted
sum of the entries. The mechanism is differentiable and lets the model decide where to read
from memory. Memory Networks and Neural Turing Machines make the same broad point: a neural
model can combine learned parameters with an external memory that is read by content.

Set processing supplies the third ingredient. A support set has no natural order, while a
plain recurrent network is order-sensitive. Work on sequence-to-sequence for sets shows that
one can read a memory with repeated attention steps: maintain a processing state, attend over
the set, read a weighted summary, and fold that readout back into the state. This gives a way
to do multiple rounds of computation over a set without treating a single arbitrary order as
the object of interest.

Siamese networks provide a contemporary few-shot baseline. They learn a pairwise same/different
embedding and then use nearest-neighbor matching. Memory-augmented meta-learners train a
recurrent model to absorb examples sequentially, storing and retrieving from an external
memory across episodes.

## Evaluation Frame

The standard test is episodic. An episode samples `N` labels, gives the learner `k` labeled
support examples per label, and evaluates on disjoint query examples from the same label set.
The labels used for evaluation are held out from ordinary representation training, so success
requires transfer to new classes rather than memorization of output units.

Omniglot is the small-scale visual benchmark: 1623 handwritten character classes from 50
alphabets, each with 20 examples. ImageNet provides the harder natural-image setting, where
held-out label subsets make it possible to test one-shot recognition of unseen classes.
Penn Treebank provides an analogous language setting: a query sentence with a blank must be
matched to one of the labels represented by support sentences. In all cases, the core
measurement is query classification accuracy within the episode, with random performance at
`1/N`.

## Code Scaffold

The implementation substrate is an episodic few-shot harness. A backbone maps images to
feature vectors. For each episode, the harness first calls `process_support_set(...)` so the
classifier can prepare task-specific state from the support images and labels. It then calls
`forward(...)` on query images and computes a loss against query labels. The open design
space is the support preparation, the query scoring rule, and the loss matched to that
scoring rule.

```python
import torch
from torch import Tensor, nn


class FewShotClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, use_softmax: bool = False):
        super().__init__()
        self.backbone = backbone
        self.use_softmax = use_softmax

    def compute_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def softmax_if_specified(self, scores: Tensor) -> Tensor:
        return scores.softmax(-1) if self.use_softmax else scores

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        pass

    def forward(self, query_images: Tensor) -> Tensor:
        pass

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        pass


def train_episode(model, support_images, support_labels, query_images, query_labels, optimizer):
    optimizer.zero_grad()
    model.process_support_set(support_images, support_labels)
    scores = model(query_images)
    loss = model.compute_loss(scores, query_labels)
    loss.backward()
    optimizer.step()
```
