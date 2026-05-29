# Dropout

## Problem

A large feedforward neural network trained on limited labeled data overfits: many weight
settings fit the training set almost perfectly, but each generalizes differently and almost all
do worse on test data. The mechanism is **co-adaptation** — a hidden feature detector becomes
useful only in the presence of specific other detectors, learning to fix their particular
mistakes on the training set. That joint tuning does not transfer. The goal is to suppress
co-adaptation, and obtain the generalization benefit of model averaging, at the cost of a single
network.

## Key idea

On every presentation of every training case, randomly omit each hidden unit independently with
probability 0.5 (and, more gently, omit a fraction of input units). Forward- and back-propagate
through the surviving units only. A unit can no longer rely on any specific other unit being
present, so it must learn a feature that is useful across the combinatorial variety of contexts
in which it may operate.

Equivalently, each masked pass trains a different thinned subnetwork. With N hidden units there
are 2^N such subnetworks, all sharing weights. Dropout is therefore model averaging — extreme
bagging where each model is trained on essentially one case and every parameter is tied across
the whole exponential family. The weight-sharing is a stronger regularizer than shrinking weights
toward zero.

## Test time: the mean network

Run one deterministic pass with all units present and every unit's outgoing weights multiplied
by its keep probability q = 1 − drop_rate (halved when q = 0.5). Equivalently, scale each
layer's activations by q.

For a single hidden layer feeding a softmax, this is **exact**: the mean network computes the
normalized geometric mean of all 2^N subnetworks' predictive distributions. With logit
z_k(m) = Σ_i m_i w_{ik} h_i + b_k under mask m,

  G(k) ∝ exp( (1/2^N) Σ_m z_k(m) ) = softmax_k( Σ_i (½ w_{ik}) h_i + b_k ),

because the per-mask log-partition term is independent of k and cancels under normalization, and
each unit is kept in exactly half the masks so (1/2^N) Σ_m m_i = ½. With another drop rate d,
the probability-weighted geometric mean gives E[m_i] = 1 − d.

The log-probability guarantee is the geometric-mean superadditivity step. For nonnegative
q_m(j),

  (Π_m Σ_j q_m(j))^{1/M} ≥ Σ_j (Π_m q_m(j))^{1/M},

where M is the number of masks. Dividing by (Π_m Σ_j q_m(j))^{1/M} gives
Σ_j Π_m r_m(j)^{1/M} ≤ Π_m(Σ_j r_m(j))^{1/M} = 1 by Hölder, with
r_m(j) = q_m(j)/Σ_j q_m(j). Equality requires the normalized vectors r_m to be identical,
which means all subnetworks make the same prediction. Therefore the geometric mean assigns the
correct class a log-probability at least as high as the average of the individual subnetworks'
log-probabilities; for linear-output regression the mean network's squared error is at most the
average of the individuals'. For deeper nets the identity holds approximately, and keep-probability
scaling restores the expected input to each next layer.

## Supporting choices

- **Drop rate 0.5 for hidden units**: maximizes subnetwork variety (max-entropy mask) while
  still leaving a usable thinned network; keep_prob = 0.5 makes the test-time rule exactly
  "halve the weights."
- **Lighter input dropout (~20%)**: inputs carry signal; keep_prob around 0.8 behaves like
  input noise without deleting too much raw evidence.
- **Max-norm constraint instead of L2 penalty**: cap each hidden unit's incoming-weight squared length
  at l by multiplying its incoming vector by scale = min(1, sqrt(l / ||w||^2)) after each
  update. A hard cap bounds weights regardless of step size, so the
  learning rate can start very large and decay — a far more thorough search of weight space than
  small-weights-with-small-rate. (When fine-tuning a pre-trained net, drop the constraint and use
  a small rate to preserve learned features.)
- **High momentum ramped toward 0.99**: each gradient is for a different stochastic net;
  momentum averages gradient information over many of them. Scale the learning rate by
  (1 − momentum).
- **Apply dropout in all fully-connected hidden layers**: co-adaptation can occur throughout the
  stack. **Little/no dropout in convolutional layers**: weight sharing already limits their
  overfitting capacity.
- **ReLU (max-with-zero)** in conv nets, with weights initialized at enough variance and hidden
  biases at a small positive constant so units start active and avoid being permanently dead.

## Algorithm

```
train:
  for each minibatch:
    visible_mask = Uniform(0, 1) > input_drop_rate
    x = x * visible_mask
    for each layer L (hidden):
      a = nonlinearity(x W + b)
      mask = Uniform(0, 1) > L.drop_rate
      a = a * mask               # forward through survivors only
    logits -> softmax -> cross-entropy
    backprop: at each hidden layer, grad = grad * mask  # gradient through survivors only
    update with momentum
    project each hidden unit's incoming weights onto ||w||^2 <= l   # max-norm

test:
  one pass, no masks, scale input and hidden activations by keep_prob = 1 - drop_rate
```

## Code

The cuda-convnet dropout mechanics are: train uses `mask = Uniform(0, 1) > drop_rate` and
`acts *= mask`, test uses `acts *= 1 - drop_rate`, and backprop reuses the same mask. The
hidden-unit max-norm constraint follows the fully-connected MNIST recipe.

```python
import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_grad(probs, y):
    g = probs.copy()
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


class Layer:
    """Fully-connected layer with per-unit dropout and a max-norm weight constraint."""
    def __init__(self, n_in, n_out, drop_rate=0.5, max_sq_norm=15.0, last=False):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b = np.zeros(n_out) + (0.0 if last else 1.0)  # positive bias keeps ReLU units alive
        self.drop_rate = 0.0 if last else drop_rate
        self.keep_prob = 1.0 - self.drop_rate
        self.max_sq_norm = max_sq_norm
        self.last = last
        self.mask = None

    def forward(self, x, train):
        self.x = x
        self.z = x @ self.W + self.b
        a = self.z if self.last else relu(self.z)
        self.mask = None
        if self.last or self.drop_rate == 0.0:
            return a
        if train:
            self.mask = (np.random.rand(*a.shape) > self.drop_rate).astype(a.dtype)
            a = a * self.mask                  # thinned subnetwork
        else:
            a = a * self.keep_prob             # mean network scale
        return a

    def backward(self, grad_a):
        if not self.last:
            if self.mask is not None:
                grad_a = grad_a * self.mask    # same mask as the forward pass
            grad_a = grad_a * (self.z > 0)     # ReLU gate
        self.gW = self.x.T @ grad_a
        self.gb = grad_a.sum(axis=0)
        return grad_a @ self.W.T

    def after_update(self):
        if self.last:
            return
        sq = (self.W ** 2).sum(axis=0, keepdims=True)
        scale = np.ones_like(sq)
        too_large = sq > self.max_sq_norm
        scale[too_large] = np.sqrt(self.max_sq_norm / sq[too_large])
        self.W *= scale                        # project hidden-unit incoming weights


class Net:
    def __init__(self, sizes, drop_rate_hidden=0.5, drop_rate_input=0.2):
        self.drop_rate_input = drop_rate_input
        self.input_keep_prob = 1.0 - drop_rate_input
        self.input_mask = None
        self.layers = []
        n_layers = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = (i == n_layers - 1)
            drop = 0.0 if last else drop_rate_hidden
            self.layers.append(Layer(a, b, drop_rate=drop, last=last))

    def forward(self, x, train):
        self.input_mask = None
        if self.drop_rate_input > 0.0:
            if train:
                self.input_mask = (np.random.rand(*x.shape) > self.drop_rate_input).astype(x.dtype)
                x = x * self.input_mask
            else:
                x = x * self.input_keep_prob
        for L in self.layers:
            x = L.forward(x, train)
        return softmax(x)

    def backward(self, probs, y):
        g = cross_entropy_grad(probs, y)
        for L in reversed(self.layers):
            g = L.backward(g)


def lr_schedule(ep, eps0=10.0, f=0.998):
    return eps0 * (f ** ep)                          # large initial rate, geometric decay

def momentum_schedule(ep, p_i=0.5, p_f=0.99, T=500):
    return (ep / T) * p_f + (1 - ep / T) * p_i if ep < T else p_f


def train(net, X, Y, epochs=3000, batch=100):
    vW = [np.zeros_like(L.W) for L in net.layers]
    vb = [np.zeros_like(L.b) for L in net.layers]
    for ep in range(epochs):
        mom = momentum_schedule(ep)
        lr  = lr_schedule(ep) * (1 - mom)
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            probs = net.forward(xb, train=True)
            net.backward(probs, yb)
            for k, L in enumerate(net.layers):
                vW[k] = mom * vW[k] - lr * L.gW
                vb[k] = mom * vb[k] - lr * L.gb
                L.W += vW[k]; L.b += vb[k]
                L.after_update()
```
