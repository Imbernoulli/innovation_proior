# Research question

Training a deep feedforward or convolutional network by stochastic gradient descent is, around 2014–2015, a fragile craft. The optimization minimizes

    Θ = argmin_Θ (1/N) Σ_i ℓ(x_i, Θ)

and each SGD step replaces the full-set gradient by a mini-batch estimate (1/m) Σ ∂ℓ(x_i,Θ)/∂Θ over a batch of size m. Mini-batches earn their keep twice over: the batch gradient is a lower-variance estimate of the full-set gradient than a single example would give (and improves as m grows), and a batch of m examples is far cheaper to evaluate on parallel hardware than m separate forward/backward passes. But the procedure works only under careful tuning of the learning rate and careful choice of the initial parameters, and deep networks built from saturating nonlinearities (sigmoid, tanh) are notoriously hard to train at all. Practitioners experience this as three separate annoyances — divergence at high learning rates, sensitivity to initialization, and dead saturating units — each with its own folk remedy.

The structural cause is that a layer's input is manufactured by every layer beneath it. Write a deep net as a composition ℓ = F₂(F₁(u, Θ₁), Θ₂). To the upper part F₂, the quantity x = F₁(u, Θ₁) is simply "the input," and a gradient step on Θ₂,

    Θ₂ ← Θ₂ − (α/m) Σ_i ∂F₂(x_i, Θ₂)/∂Θ₂,

is *identical* to the step a stand-alone network F₂ fed x would take. So everything known about training a learner — including the textbook fact that a learner converges faster when its input distribution is fixed (the same between train and test, and stable in time) — applies to *every internal sub-network*, not only to the data fed in at the bottom. The trouble is that every time the lower parameters Θ₁ update, the distribution of x moves: its mean wanders, its variance breathes, its shape changes. Each upper layer must then spend capacity continually re-adapting to its own drifting inputs, and the effect compounds with depth, because a small nudge to an early parameter is amplified as it propagates upward. With a saturating nonlinearity the drift is actively lethal: in z = g(Wu + b) with g the logistic sigmoid, as |Wu+b| grows g'(·) → 0, so once the drift pushes many coordinates of the pre-activation into the flat tails, the gradient flowing down to u vanishes and that layer stops learning.

A solution must keep the distribution of each layer's inputs stable *throughout* training — not merely at initialization, not merely at the data layer — while staying cheap enough to run at every single step. If it works, it should let practitioners raise the learning rate without divergence, make initialization far less delicate, make saturating nonlinearities trainable again, and ideally lighten the regularization burden.

# Background

Several strands of established knowledge bear directly on the question, and each leaves a specific gap the solution must close.

**Whitening accelerates convergence — and the cheap part of it suffices.** LeCun, Bottou, Orr & Müller ("Efficient BackProp", 1998) and Wiesler & Ney ("A convergence analysis of log-linear training", 2011) establish that a network converges faster when its inputs are *whitened*: linearly transformed to zero mean, unit variance, and decorrelated. A load-bearing detail in LeCun et al. is that per-coordinate mean/variance normalization helps *even when the features are not decorrelated* — the decorrelation is the expensive part and the least essential. The gap: this wisdom is applied once, statically, to the network's external inputs; it says nothing about internal activations that move during training.

**Covariate shift.** Shimodaira (2000) names *covariate shift*: the input distribution to a learning system differs between training and test, classically repaired by domain adaptation (Jiang, 2008, survey). The notion is defined for the whole system and for the train/test gap; it is not, in the prior art, applied to the parts of a network or to a shift that happens during training.

**Initialization theory — a static fix for a dynamic problem.** Glorot & Bengio (2010) derive initialization schemes (Xavier) that keep activation and gradient variances roughly constant across layers at the *start* of training. Saxe, McClelland & Ganguli (2013), for deep linear nets, show that orthogonal initialization with Jacobian singular values near 1 gives well-behaved gradient propagation — singular values near 1 preserve gradient magnitude through depth. Both pin only the starting distribution; as parameters move, the distribution drifts again, so they are static remedies for a dynamic problem.

**Saturation workarounds.** Nair & Hinton (2010) introduce the ReLU, max(x,0), whose non-saturating positive branch is the standard escape from vanishing gradients. Combined with small learning rates and careful init, ReLU is the accepted toolkit for getting depth to train — a treatment of the symptom (saturation) rather than its cause (distribution drift into the saturated regime).

**Normalization pushed inside training, and why it has been awkward.** Several groups are moving normalization or whitening *into* the optimization: mean-normalized SGD (Wiesler et al., 2014), the linear-transformation reparameterizations of Raiko, Valpola & LeCun (2012), natural-gradient methods (Povey et al., 2014), and natural neural networks (Desjardins & Kavukcuoglu). These either modify the optimizer or perform normalization as a side computation interleaved with the gradient steps. The pitfall this creates is sharp and is the central cautionary tale of the setting (made concrete under *Baselines*): if the gradient step is blind to the dependence of the normalization statistics on the parameters, the optimizer and the normalizer fight, and a parameter can drift unboundedly while the loss stays flat.

**Per-example normalization discards scale.** Lyu & Simoncelli (2008) use divisive normalization computed over a single example, or across feature maps at one spatial location. This stabilizes scale but discards the *absolute scale* of activations, changing what the network can represent — the normalization is computed from a single example, with no reference to the statistics of the rest of the data.

**Pre-activation is "more Gaussian".** Hyvärinen & Oja (2000, ICA) give the intuition that an affine mixture Wu+b of many inputs tends toward a symmetric, non-sparse, more-Gaussian distribution than the output of a preceding nonlinearity, whose distributional shape keeps changing during training. For a distribution far from Gaussian, matching only its first two moments need not pin its shape.

**The convolutional property.** Convolutional layers (LeCun et al., 1998a) share one filter across all spatial locations of a feature map: the same weights produce every location's value, which is what gives convolutions their translation equivariance and their parameter efficiency. Different locations of one feature map are produced by the same weights. An operation that treated each (channel, location) as an independent activation would break the weight-sharing symmetry and undo part of why convolutions work.

**Bias and variance of a sample variance.** It is a standard fact that the sample variance σ̂² = (1/m) Σ (x_i − x̄)² computed about the sample mean is a *biased* estimator of the population variance, underestimating it by a factor (m−1)/m because one degree of freedom is consumed by estimating the mean; the unbiased estimator multiplies by m/(m−1) (Bessel's correction).

# Baselines

The natural points of comparison are the prevailing recipes for training deep nets, and the normalization-inside-training attempts the question reacts against.

**Plain SGD with momentum / Adagrad on a strong conv net.** The reference regime: a state-of-the-art convolutional classifier (the Inception/GoogLeNet line of Szegedy et al., 2014 — many convolutional and pooling layers, ReLU nonlinearities, a softmax over 1000 classes, trained with momentum SGD, Sutskever et al. 2013, in a distributed framework à la Dean et al. 2012). The recipe: ReLU for non-saturation, Xavier/orthogonal init for a good starting distribution, a conservatively small learning rate, Dropout and L2 weight decay for regularization, exponential learning-rate decay. The gap: the input distribution to each internal layer still drifts during training, so the learning rate must stay small and saturating nonlinearities remain effectively untrainable; raising the learning rate risks the parameter scale exploding, because larger weights amplify gradients in backprop and the amplification feeds back on the weight growth.

**Dropout (Srivastava et al., 2014).** Randomly zeroes each unit's output with probability p at training time (rescaling the survivors), so the network cannot rely on any single unit; at test time the full network is used. An effective regularizer that prevents co-adaptation, but it is *only* a regularizer — it does nothing about input-distribution drift or saturation, and the noise it injects tends to slow convergence.

**Normalization as a separate step (the failure mode to beat).** Take a unit that adds a learned bias, x = u + b, then is centered by subtracting the dataset mean over X = {x₁…x_N}: x̂ = x − E[x], with E[x] = (1/N)Σx_i. If the gradient step ignores the dependence of E[x] on b and updates b ← b + Δb with Δb ∝ −∂ℓ/∂x̂, then the re-centered output is

    (u + b + Δb) − E[u + b + Δb] = u + b − E[u + b],

identical to before. The output, and hence the loss, do not change — yet b has moved, and will keep moving step after step (the loss gradient keeps requesting the same Δb), growing without bound while the loss stays perfectly flat. The model blows up; this is observed empirically when normalization statistics are computed outside the gradient step, and gets worse if the normalization also rescales by a variance. Treating normalization as a side computation, with the gradient step blind to how the statistics depend on the parameters, is the gap.

**Full joint whitening.** A stronger version would, for any parameter setting, always produce activations with the desired whitened distribution. Full whitening means forming the covariance Cov[x] = E[xxᵀ] − E[x]E[x]ᵀ (a d×d matrix), computing its inverse square root Cov[x]^{−1/2} to produce the whitened activations Cov[x]^{−1/2}(x − E[x]), and differentiating through *all* of it — including the matrix inverse-square-root, which in practice goes through an eigendecomposition or SVD at O(d³) cost per step, is not everywhere differentiable, and must be re-done over the whole training set after every update. The gap: correct but far too expensive; and with a mini-batch smaller than the number of activations being whitened (the usual case), the empirical covariance is singular and would require regularization just to invert. Joint whitening is thus unusable for per-step use.

**Gülçehre & Bengio (2013) standardization layer.** A pre-existing layer that standardizes activations, applied to the *output* of the nonlinearity, with no learned scale/shift, no convolutional handling, and no separate deterministic-inference rule. It is the nearest neighbor in the prior art.

# Evaluation settings

The natural yardsticks are:

**MNIST (LeCun et al., 1998a)** — handwritten-digit classification, 28×28 images, 10 classes. The controlled probe: a deliberately simple, non-state-of-the-art net (3 fully-connected hidden layers of 100 units each, sigmoid nonlinearities, small-Gaussian-initialized weights, a final 10-way layer with cross-entropy loss) trained ~50000 steps with 60 examples per mini-batch. The aim is a *comparison* between a baseline and a normalized variant, plus inspection of how the distribution of a typical sigmoid's input (tracked by its {15, 50, 85}th percentiles) evolves over training — a direct read on distribution drift. Metric: held-out test accuracy versus number of training steps.

**ImageNet / ILSVRC 2012 (Russakovsky et al., 2014)** — 1000-class image classification on the LSVRC2012 training set, evaluated on the 50000-image validation set (and a held-out test server). Host architecture: a deep Inception-style conv net (Szegedy et al., 2014) on the order of ten-plus million parameters, trained with momentum SGD, mini-batch size around 32, in a distributed setup. Metrics: top-1 and top-5 error; validation accuracy@1 (probability the single top prediction is correct, single-crop) tracked *as a function of training steps* — so "how many steps to reach a target accuracy" is itself a measurable quantity. Standard protocol levers available for tuning: initial learning rate, learning-rate decay schedule, Dropout probability, L2 weight, local response normalization, data shuffling, and the strength of photometric distortion in augmentation. Multi-crop (e.g. 144 crops) and model-ensembling are the standard ways to push final accuracy.

# Code framework

What already exists is a bare deep-net training harness with explicit per-layer forward/backward passes, implemented in NumPy. There is a generic layer abstraction, a few concrete layers that subclass it, a container that stacks layers, an SGD optimizer, a training loop, a loss, and an *input-only* preprocessing/whitening utility applied once to the data. The open engineering slot is one more generic layer that receives tensors, returns transformed tensors, caches backward state, and exposes any learnable parameters. Concretely:

```python
import numpy as np

# ---- the generic layer abstraction every layer implements ----
class Layer:
    """Base class. A concrete layer fills in forward/backward and (if it has
    learnable parameters) registers them so the optimizer can update them."""
    def forward(self, x):
        # returns out; stashes whatever backward needs on self.cache
        raise NotImplementedError   # TODO: defined by each concrete layer

    def backward(self, dout):
        # returns dx; sets self.grads for each learnable parameter
        raise NotImplementedError   # TODO: defined by each concrete layer

    def params_and_grads(self):
        # (name, value, grad) triples; empty for parameter-free layers
        return []

# ---- concrete layers that already exist (they fill in the stubs above) ----
class Affine(Layer):                      # fully-connected pre-activation producer Wu+b
    def __init__(self, W, b): self.W, self.b = W, b
    def forward(self, x):
        N = x.shape[0]; self.x = x
        return x.reshape(N, -1).dot(self.W) + self.b
    def backward(self, dout):
        x = self.x; xr = x.reshape(x.shape[0], -1)
        self.dW = xr.T.dot(dout); self.db = np.sum(dout, axis=0)
        return dout.dot(self.W.T).reshape(x.shape)
    def params_and_grads(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]

class Conv(Layer):                        # weight-shared filter over (N,C,H,W) tensors
    def __init__(self, W, b, conv_param): self.W, self.b, self.cp = W, b, conv_param
    def forward(self, x):  ...            # standard conv forward (omitted)
    def backward(self, dout):  ...        # standard conv backward (omitted)
    def params_and_grads(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]

class Pool(Layer):                        # max-pooling over (N,C,H,W)
    def forward(self, x):  ...
    def backward(self, dout):  ...

class ReLU(Layer):                        # nonlinearity max(0,x)
    def forward(self, x): self.x = x; return np.maximum(0, x)
    def backward(self, dout):
        dx = np.array(dout, copy=True); dx[self.x <= 0] = 0; return dx

# ====================================================================
# TODO: one new Layer subclass can occupy this slot. It obeys the same
# forward/backward contract and may register learnable parameters.
class NewLayer(Layer):
    def forward(self, x):
        pass  # TODO
    def backward(self, dout):
        pass  # TODO
    def params_and_grads(self):
        pass  # TODO, if it has parameters
# ====================================================================

# ---- container, optimizer, loss, training loop (all already exist) ----
class Net:                                # stacks Layers; runs forward then backward
    def __init__(self, layers): self.layers = layers
    def forward(self, x):
        for L in self.layers: x = L.forward(x)
        return x
    def backward(self, dout):
        for L in reversed(self.layers): dout = L.backward(dout)
    def params_and_grads(self):
        for L in self.layers:
            for t in L.params_and_grads(): yield L, t

class SGD:                                # vanilla SGD (+momentum) over all parameters
    def __init__(self, lr, mu=0.9): self.lr, self.mu, self.v = lr, mu, {}
    def step(self, net):
        for L, (name, val, grad) in net.params_and_grads():
            key = (id(L), name)
            v = self.v.get(key, np.zeros_like(val))
            v = self.mu * v - self.lr * grad
            self.v[key] = v; val += v       # in-place parameter update

def softmax_loss(scores, y):              # returns (loss, dscores); seeds the top gradient
    probs = np.exp(scores - scores.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    N = scores.shape[0]
    loss = -np.log(probs[np.arange(N), y]).sum() / N
    dscores = probs.copy(); dscores[np.arange(N), y] -= 1; dscores /= N
    return loss, dscores

def preprocess(X_train, X):               # INPUT-ONLY whitening, applied ONCE to the data
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0) + 1e-8
    return (X - mu) / sd                   # NOT applied to internal activations

def train(net, opt, data, steps, batch=60):
    Xtr, ytr = data
    for t in range(steps):
        idx = np.random.choice(len(Xtr), batch, replace=False)
        scores = net.forward(Xtr[idx])
        loss, dscores = softmax_loss(scores, ytr[idx])
        net.backward(dscores)
        opt.step(net)
```

The harness is complete except for whatever lives in the marked slot. Any new transform enters as one more `Layer` subclass implementing `forward`/`backward` (and registering any learnable parameters), stacked at the appropriate point in `Net` and updated by the existing `SGD` like every other layer. The `preprocess` utility only touches the data at the bottom; nothing currently stabilizes the distributions of the internal activations as the parameters below them move. A layer that needs different training and inference behavior, or that needs to accumulate state across steps, carries that state on the layer or in a small parameter dictionary.
