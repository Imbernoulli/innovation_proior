# Research question

We want a neural density estimator for general-purpose, continuous, tabular and image data that
is at once **flexible** (capable of representing complex, possibly multimodal densities) and
**tractable** (an exact, cheap-to-evaluate density and a cheap learning algorithm). The specific
emphasis is on tasks where the density itself is the object of interest — likelihood-free
inference, learned priors and proposals, surrogate models — so the estimator must, for *any*
externally provided datapoint x, return the exact p(x) in a single fast pass on a GPU, not via a
D-step sequential loop. The open question: can we take one of the two flexible-and-tractable
families we already have and make it markedly more flexible *without* losing the one-pass exact
density?

# Background

**Two flexible, tractable families.** Most neural density estimators that are both flexible and
tractable fall into two camps. *Autoregressive models* decompose the joint by the chain rule,
p(x) = ∏_i p(x_i | x_{1:i-1}), and model each one-dimensional conditional in turn; the
log-likelihood is exact and a sum of per-dimension terms. *Normalizing flows* (Rezende & Mohamed
2015) write p(x) = π_u(f^{-1}(x)) · |det(∂f^{-1}/∂x)| for an invertible, tractable-Jacobian map f
pushing a simple base density π_u (e.g. a standard Gaussian) onto the data; for the formula to be
usable f must be (a) easy to invert and (b) have an easy Jacobian determinant, and crucially these
properties survive composition, so a flow can be deepened by stacking copies of f.

**Autoregressive estimators and the order problem.** The Real-valued Neural Autoregressive Density
Estimator (RNADE; Uria et al. 2013) models each conditional as a mixture of Gaussians or Laplacians
with a linear hidden-state update; LSTM-based variants (Theis & Bethge 2015; van den Oord et al.
2016) update the hidden state more flexibly. A structural weakness of all of them is *order
sensitivity*: the chosen variable ordering matters, and for a given conditional family one order
may represent a density exactly while another cannot — yet which of the factorially many orders is
best is unknown a priori. Training over random orders and ensembling predictions (Uria et al. 2014;
Germain et al. 2015) is one mitigation.

**The sequential-evaluation bottleneck and masking.** A straightforward recurrent autoregressive
model updates a hidden state once per variable, so computing p(x) of a D-dimensional vector costs D
*sequential* steps — poorly suited to GPUs. The fix is to start from a fully-connected network with
D inputs and D outputs and *drop connections* so output i depends only on inputs 1, …, i−1; then
output i parameterizes the i-th conditional and the whole vector of conditionals is computed in one
parallel forward pass while still respecting the autoregressive property. The Masked Autoencoder for
Distribution Estimation (MADE; Germain et al. 2015) implements exactly this by multiplying each
weight matrix with a binary mask. Its mask-design rule: assign every input and hidden unit an
integer *degree* in {1, …, D} (an input's degree is its index in the chosen order; the D outputs get
degrees 0, …, D−1), and allow a connection only from a unit of lower-or-equal degree to a higher
one. For output i to connect to all inputs of degree < i with no spurious conditional independences,
it is necessary and sufficient that every hidden layer contains every degree. Concretely the mask
between two layers is M[j, k] = 1{ degree_next[j] ≥ degree_prev[k] }. Other connection-dropping
schemes include masked convolutions (PixelCNN) and causal convolutions (WaveNet).

**Flows whose Jacobian is tractable by design.** Early flow ideas include Gaussianization (Chen &
Gopinath 2000), built from successive ICA. Enforcing invertibility through nonsingular weight
matrices (Rippel & Adams 2013; Ballé et al. 2016) leaves a Jacobian determinant that is O(D³) in
general. Planar and radial flows (Rezende & Mohamed 2015) and Inverse Autoregressive Flow (IAF;
Kingma et al. 2016) have tractable Jacobians by construction but were built for *variational
inference*: they can efficiently evaluate the density of *their own samples* but not of an
externally provided datapoint, which makes them awkward for density estimation. NICE (Dinh et al.
2014) and its successor Real NVP (Dinh et al. 2017) have tractable Jacobians and are usable for
density estimation.

**The realization that links the two families.** Kingma et al. (2016) pointed out that an
autoregressive model, *viewed as a data generator*, is a differentiable transformation of an
external source of randomness u (the random numbers a sampler draws): with conditionals
parameterized by per-coordinate functions of the earlier coordinates, the sampler maps u to x
coordinate by coordinate.

**Batch normalization.** Batch normalization (Ioffe & Szegedy 2015) is an elementwise affine
rescaling. Real NVP used it between its coupling layers and found it sped up and stabilized training.

# Baselines

**RNADE / recurrent autoregressive models.** p(x) = ∏_i p(x_i | x_{1:i-1}) with mixture-of-Gaussian
conditionals (Uria et al. 2013) or LSTM hidden states. *Math/algorithm:* exact per-dimension
log-likelihood; flexibility comes from rich conditionals. *Gaps:* sequential D-step evaluation in
the recurrent form (GPU-unfriendly), and sensitivity to the variable order.

**MADE (Germain et al. 2015).** A masked feedforward autoencoder computing all conditional
parameters in one pass. *Math/algorithm:* binary masks enforce the autoregressive property; with a
single-Gaussian conditional per dimension it is a one-pass exact density estimator. *Gap:* a single
MADE with single-Gaussian conditionals has *unimodal* conditionals, so its expressiveness is
limited; one fixed order is baked in.

**Inverse Autoregressive Flow (IAF; Kingma et al. 2016).** A flow whose layer is the affine
autoregressive recursion x_i = u_i·exp(α_i) + μ_i with μ_i, α_i computed from the *random numbers*
u_{1:i-1} (via a MADE). *Math/algorithm:* because each conditioner reads earlier u's, the whole u→x
map runs in one parallel pass, so IAF can sample and score its own samples in one pass. *Gap for
density estimation:* to score an external x it must first invert to recover u, which is a D-step
sequential recursion — efficient only for the variational-inference use where it scores its own
samples.

**Real NVP (Dinh et al. 2017) / NICE (Dinh et al. 2014).** A flow of *coupling layers*: split the
vector at index d, copy x_{1:d} = u_{1:d}, and affinely transform the rest,
x_{d+1:D} = u_{d+1:D} ⊙ exp(α) + μ with α, μ functions of u_{1:d}; NICE is the α = 0 (additive,
volume-preserving) special case. *Math/algorithm:* the Jacobian is triangular with the copied block
giving an identity diagonal, so the log-det is Σα over the transformed block; permuting which
elements are copied across layers mixes all coordinates. *Strength:* both sampling and density run in
one forward pass. *Gap:* each coupling layer leaves a whole block untouched and only conditions the
transformed block on the copied block — strictly less flexible than scaling/shifting *every*
coordinate as a function of *all* previous ones.

# Evaluation settings

The yardstick is **average test log-likelihood in nats** for general-purpose density estimation
(and bits-per-pixel for images, lower better). Unconditional estimation is run on four UCI datasets
— POWER (D = 6), GAS (8), HEPMASS (21), MINIBOONE (43) — preprocessed by standardization, removal of
discrete-valued and near-perfectly-correlated attributes (Pearson > 0.98), and small added noise to
avoid trivial density spikes; and on BSDS300 (Martin et al. 2001), random 8×8 monochrome natural-image
patches dequantized and rescaled to [0, 1] with the patch mean subtracted and the bottom-right pixel
discarded (D = 63). Conditional estimation p(x | y) is run on MNIST (D = 784) and CIFAR-10 (D = 3072)
with a 10-dimensional one-hot class label y, evaluating p(x) = Σ_y p(x | y)p(y) under a uniform label
prior; images are dequantized, rescaled to [0, 1], and mapped into logit space by
x ↦ logit(λ + (1 − 2λ)x) with λ = 10⁻⁶ for MNIST and 0.05 for CIFAR-10, and CIFAR-10 is augmented with
horizontal flips. A Gaussian fitted to the train data is the natural simple baseline. Optimization is
Adam with minibatches of 100, a small ℓ₂ coefficient of 10⁻⁶, and early stopping after 30 epochs
without validation improvement. The number of hidden layers and units per dataset are chosen by
validation. Comparison points from the literature include Deep RNADE (Uria et al. 2014) on BSDS300.

# Code framework

The primitives needed already exist: a masked linear layer (a `Linear` whose weight is multiplied by a
fixed binary mask), an Adam optimizer, and tensor reshaping. The model itself is to be built.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedLinear(nn.Linear):
    """A Linear whose weight is masked by a fixed binary matrix: y = x @ (mask * W.T) + b."""
    def __init__(self, n_in, n_out, bias=True):
        super().__init__(n_in, n_out, bias)
        self.mask = None
    def set_mask(self, mask):
        self.mask = mask
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class DensityModel(nn.Module):
    def __init__(self, dim, hidden_dims):
        super().__init__()
        # TODO: build the model.
        pass
    def log_prob(self, x):
        # TODO: return the exact log-density of x.
        pass

def make_optimizer(params, deep=True):
    return torch.optim.Adam(params, lr=1e-4 if deep else 1e-3, weight_decay=1e-6)
```
