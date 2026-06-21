A large feedforward network with more hidden units than labeled examples almost always overfits. The training error can be driven very low, but the test error stays high because the optimizer finds weight settings that fit the training set through fragile joint adjustments rather than robust individual features. The core failure is co-adaptation: a hidden unit learns to be useful only in the presence of specific other units, correcting their particular mistakes on the training cases. That conspiracy works on the data it was tuned for and breaks on held-out data. Plain weight decay only shrinks weights toward zero uniformly, which is a blunt capacity control that does not directly stop units from depending on specific partners. Bagging and Bayesian model averaging would fix the variance problem by averaging many models, but training and evaluating many large networks is prohibitively expensive. Mixtures of experts route each example to only a few specialists, which wastes limited data by splitting it across sub-models. Naive Bayes avoids co-adaptation entirely by making every feature independent, but that throws away useful cooperation. What is needed is a way to keep cooperation possible while forbidding dependence on any specific collaborator, and to obtain the averaging benefit of an ensemble without paying the cost of multiple separate networks.

The method is Dropout. During training, every hidden unit is randomly omitted on each forward pass with probability one half, and a smaller fraction of input units is omitted as well. The forward and backward passes use only the surviving units, and the mask is freshly sampled for every training case. Because any given partner unit may be absent, a unit cannot survive by learning to fix the mistakes of a specific small committee; it must learn features that are useful across the combinatorial variety of contexts in which it might find itself. This directly attacks co-adaptation. At the same time, each sampled mask defines a different thinned subnetwork, and with N hidden units there are 2^N such subnetworks. They all share weights, so a single training run implicitly trains an exponential family of networks and averages them through parameter tying. That is the variance-reduction benefit of model averaging woven into the cost of one network.

At test time, one deterministic pass is used. All units are kept, but each hidden activation is scaled by its keep probability, so the expected input to the next layer matches what was seen during training. For a single hidden layer followed by softmax, this scaling is exact: the deterministic network computes the normalized geometric mean of all 2^N subnetwork predictions. With logits z_k(m) = sum_i m_i w_ik h_i + b_k under mask m, the geometric mean over masks is proportional to exp of the average logit. Since each unit appears in half the masks, the average logit is sum_i (0.5 w_ik) h_i + b_k, which is exactly the network with weights halved. The geometric-mean construction also guarantees that the log-probability assigned to the true class is at least the average log-probability of the individual subnetworks, with equality only when all subnetworks agree. For deeper networks the exact identity becomes approximate, but keep-probability scaling still restores the expected activations layer by layer.

Dropout requires supporting training choices. Because every mini-batch optimizes a different stochastic network, gradients are noisier than usual, so momentum is ramped toward 0.99 to average information over many sampled architectures. The learning rate is scaled by one minus the momentum to keep the effective step size sane, and it starts large with geometric decay so the optimizer searches weight space aggressively. An L2 penalty is replaced by a per-hidden-unit max-norm constraint: after each update, the incoming weight vector of each hidden unit is projected onto an L2 ball of fixed squared radius. A hard constraint caps weights regardless of the proposed step size, which makes the large learning rate safe. The dropout rate of 0.5 in fully-connected hidden layers maximizes the variety of sampled subnetworks while leaving enough computation; a gentler rate around 0.2 is used on inputs because raw features carry irreplaceable signal. Convolutional layers usually need little or no dropout because weight sharing already limits their capacity. ReLU nonlinearities are used with small positive hidden biases so units start active and do not die under the masking.

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
        self.b = np.zeros(n_out) + (0.0 if last else 1.0)
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
            a = a * self.mask
        else:
            a = a * self.keep_prob
        return a

    def backward(self, grad_a):
        if not self.last:
            if self.mask is not None:
                grad_a = grad_a * self.mask
            grad_a = grad_a * (self.z > 0)
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
        self.W *= scale


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
    return eps0 * (f ** ep)

def momentum_schedule(ep, p_i=0.5, p_f=0.99, T=500):
    return (ep / T) * p_f + (1 - ep / T) * p_i if ep < T else p_f


def train(net, X, Y, epochs=3000, batch=100):
    vW = [np.zeros_like(L.W) for L in net.layers]
    vb = [np.zeros_like(L.b) for L in net.layers]
    for ep in range(epochs):
        mom = momentum_schedule(ep)
        lr = lr_schedule(ep) * (1 - mom)
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            probs = net.forward(xb, train=True)
            net.backward(probs, yb)
            for k, L in enumerate(net.layers):
                vW[k] = mom * vW[k] - lr * L.gW
                vb[k] = mom * vb[k] - lr * L.gb
                L.W += vW[k]
                L.b += vb[k]
                L.after_update()
```
