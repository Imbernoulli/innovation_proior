# Context

## Research question

A large feedforward neural network with enough hidden units to model a complicated
input-output relationship, trained on a limited amount of labeled data, generalizes poorly.
When the network has more than enough capacity, there are typically many different weight
settings that fit the training set almost perfectly. Each of these settings makes different
predictions on held-out data, and almost all of them do worse on test data than on training
data: the feature detectors have been tuned to work well *together* on the training set, but
that joint tuning does not transfer.

The precise goal: reduce this overfitting in large nets without (a) acquiring more labeled
data, (b) hand-wiring task-specific structure, or (c) paying the cost of training and
evaluating a large ensemble of separate networks. A solution must improve generalization while
keeping training and test cost close to that of a single network.

## Background

A feedforward net stacks layers of non-linear hidden units between input and output. Backprop
(Rumelhart, Hinton & Williams 1986) learns the incoming weights of each hidden unit so the unit
becomes a feature detector that helps predict the correct output. Capacity grows fast with width
and depth, and with limited labels the optimizer has many equally-good-on-train solutions to
choose from.

The mechanism behind the poor generalization can be named more sharply. A hidden unit can become
useful *only in the context of* several specific other hidden units — it learns to fix up the
particular mistakes those collaborators make on the training cases. These complex co-adaptations
fit the training data but are brittle: the specific context a detector depends on need not recur
on test data. This is a more targeted description of overfitting than "weights too large."

Several lines of prior work bear directly on this.

*Reducing variance by averaging.* The standard, principled way to cut generalization error is to
average the predictions of many models. Averaging reduces the variance contribution to error.
The obstacle is purely computational: training many large nets, and running all of them at test
time, is expensive.

*Ways of combining predictions.* When models output probability distributions, two natural
combinations exist. The arithmetic mean (mixture) averages the probabilities. The normalized
geometric mean multiplies the probabilities and renormalizes — a product of experts. A useful
fact about products of experts and geometric-mean combinations (Hinton 2002): the combined
distribution assigns the correct answer a log-probability at least as high as the average of the
individual models' log-probabilities, with equality only when the models agree. For squared
error with linear outputs, the error of the averaged prediction is no worse than the average of
the individual errors (a Jensen / bias-variance fact).

*Generative pre-training.* Unsupervised, layer-wise pre-training initializes a deep net's
weights from data without using labels — a Deep Belief Network of stacked RBMs trained by
contrastive divergence (Hinton & Salakhutdinov 2006), or a Deep Boltzmann Machine
(Salakhutdinov & Hinton 2009). This finds useful feature detectors that supervised fine-tuning
can then sharpen, and it improves generalization in the small-label regime. It is a separate
phase and does not, on its own, prevent the subsequent discriminative fine-tuning from
co-adapting its units.

*Empirical observation about feature detectors.* When the first-layer features of a fully
connected net trained by plain backprop are visualized, they are typically messy and hard to
interpret — consistent with units that have specialized to correct each other rather than to
detect individually meaningful structure. This is a diagnostic about what plain backprop
produces, knowable by inspection of trained nets.

## Baselines

**Plain backpropagation with weight decay.** The default recipe: stochastic gradient descent on
a cross-entropy (or squared-error) objective, with an L2 penalty (½λ‖w‖²) added to keep weights
from growing. The penalty shrinks every weight toward zero uniformly. Two limitations: it is a
blunt, untargeted capacity control that does nothing about the *specific* failure of detectors
that are useful only together; and as a penalty it only weakly opposes an arbitrarily large
proposed weight update, so one cannot safely run a very large learning rate to search weight
space aggressively.

**Bagging (Breiman 1996) and Random Forests (Breiman 2001).** Train each model on a bootstrap
resample of the training cases; combine with equal weight. Variance reduction through diversity.
It is used overwhelmingly with cheap base learners such as decision trees, because the method's
cost scales with the number of separately-fit, separately-evaluated models. The component models
share nothing. Applying it to many large neural nets is expensive at both train and test time.

**Bayesian model averaging (Neal 1996).** Weight each model by its posterior probability given
the data. For model classes as complex as neural nets this requires Markov-chain Monte-Carlo
sampling of weight configurations. Principled and accurate, but computationally heavy and
impractical at the scales of interest.

**Mixture of experts (Jacobs, Jordan, Nowlan & Hinton 1991).** A gating network routes each
input to a small number of specialized expert networks. Because each expert is trained on only
the fraction of data routed to it, each parameter is adapted on a small slice of the training
set — statistically inefficient with limited data.

**Naive Bayes (extreme reference point).** Each input feature is trained on its own to predict
the label; at test time the per-feature predictive distributions are multiplied together. With
very little data this often beats logistic regression, which trains each feature to work in the
context of all the others — precisely because no feature in naive Bayes relies on any context.

## Evaluation settings

The natural yardsticks are standard supervised benchmarks where overfitting of large nets is the
binding constraint, evaluated by held-out test error.

- **MNIST**: 60,000 training and 10,000 test images, 28×28 grayscale handwritten digits, 10
  classes. The standard generalization probe for fully-connected nets. Protocol: train a
  feedforward classifier, report number of misclassified test images, with no data augmentation,
  weight-sharing, or pre-training when isolating the regularizer's effect; pre-trained variants
  (DBN, DBM) are a separate condition.
- **TIMIT**: clean read speech, small vocabulary, with phone-level transcriptions. A deep net
  acoustic model maps a window of consecutive feature frames (log filter-bank energies, 25 ms
  windows, 10 ms stride, mean/variance normalized per dimension) to a distribution over HMM
  states. Metrics: frame classification error on the core test set; phone recognition rate after
  Viterbi decoding with an HMM. Kaldi for feature extraction and alignment.
- **CIFAR-10**: 32×32 color images, 10 object classes, 50,000 train / 10,000 test. Convolutional
  net territory; report test classification error, no augmentation when isolating the
  regularizer.
- **ImageNet (2010 ILSVRC subset)**: ~1000 classes, ~1.3M training images, full-resolution
  (resized to 256×256); large-scale object recognition where even strong nets overfit. Report
  test error, optionally averaging predictions over crops.
- **Reuters / RCV1-v2**: newswire documents categorized into a class hierarchy; a subset of 50
  mutually exclusive second-level classes. Each document is a bag-of-counts over the 2000 most
  frequent non-stop words, count C mapped to log(1+C). Report test classification error.

Models are trained with stochastic gradient descent on mini-batches, momentum, cross-entropy
(multinomial logistic) objective, with architecture choices (depth, width, learning-rate
schedule) selected on a held-out validation set. For convolutional models the primitives are
convolutional layers, max/average pooling, local response normalization, and the
max-with-zero (ReLU) nonlinearity.

## Code framework

The available pieces are a data pipeline, a layered net with forward and backward passes, a
cross-entropy loss, and an SGD-with-momentum training loop. The empty slots are the points where
a new training rule can modify layer activations, activation gradients, inputs, schedules, or
post-update state.

```python
import numpy as np

# ---- existing primitives ----------------------------------------------------

def relu(z):                      # max-with-zero nonlinearity
    return np.maximum(0.0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_grad(probs, y):  # dL/dlogits for multinomial logistic loss
    g = probs.copy()
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


class Layer:
    """Fully-connected layer: a = f(x W + b)."""
    def __init__(self, n_in, n_out, last=False):
        self.W = (np.random.randn(n_in, n_out) * 0.01)
        self.b = np.zeros(n_out) + (0.0 if last else 1.0)  # positive bias for ReLU units
        self.last = last
        # TODO: store any per-layer training-rule state.

    def forward(self, x, train):
        self.x = x
        self.z = x @ self.W + self.b
        a = self.z if self.last else relu(self.z)
        # TODO: apply the layer-level training rule.
        return a

    def backward(self, grad_a):
        # TODO: apply the matching backward-pass rule.
        g = grad_a if self.last else grad_a * (self.z > 0)
        self.gW = self.x.T @ g
        self.gb = g.sum(axis=0)
        return g @ self.W.T

    def after_update(self):
        pass  # TODO: apply any post-update rule required by the training recipe


class Net:
    def __init__(self, sizes):
        self.layers = [Layer(a, b, last=(i == len(sizes) - 2))
                       for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        # TODO: store any input-level training-rule state.

    def forward(self, x, train):
        # TODO: apply the input-level rule.
        for L in self.layers:
            x = L.forward(x, train)
        return softmax(x)

    def backward(self, probs, y):
        g = cross_entropy_grad(probs, y)
        for L in reversed(self.layers):
            g = L.backward(g)


def train(net, X, Y, epochs, batch=100):
    vW = [np.zeros_like(L.W) for L in net.layers]   # momentum buffers
    vb = [np.zeros_like(L.b) for L in net.layers]
    for ep in range(epochs):
        lr  = lr_schedule(ep)        # TODO
        mom = momentum_schedule(ep)  # TODO
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            probs = net.forward(xb, train=True)
            net.backward(probs, yb)
            for k, L in enumerate(net.layers):
                vW[k] = mom * vW[k] - lr * L.gW
                vb[k] = mom * vb[k] - lr * L.gb
                L.W += vW[k]; L.b += vb[k]
                L.after_update()


def lr_schedule(ep):       pass  # TODO
def momentum_schedule(ep): pass  # TODO
```
