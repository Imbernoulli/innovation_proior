# Context: second-order optimization for neural networks

## Research question

Training deep neural networks is fundamentally an optimization problem, and the method of choice is
stochastic gradient descent, usually with momentum. SGD is cheap per step and tolerates the
stochasticity of minibatch gradients gracefully, but it ignores the geometry of the loss surface:
it treats every parameter direction as if it had the same curvature. For the loss surfaces that
arise in deep networks this is badly false. The curvature is strongly anisotropic and strongly
*correlated across parameters* — the Hessian (and its well-behaved relatives) is dense and far from
diagonal. An optimizer that accounted for this could, in principle, make far more progress per
update.

The precise problem is this. There is a curvature matrix `C` (the Hessian, or a positive
semi-definite surrogate for it) such that the locally optimal update is `−C⁻¹∇h`. Methods that
actually compute such updates make orders of magnitude fewer updates to reach a given loss than
methods that only scale the gradient by a diagonal. But forming and inverting `C` is hopeless for a
network with millions of parameters: `C` is (#params)², which is millions², so it can be neither
stored nor inverted directly. The question is whether there is an approximation of `C` that is

1. **rich** — not diagonal, not low-rank, capturing the dominant cross-parameter correlations, so
   that its updates retain the power of true second-order updates;
2. **directly invertible** — cheaply, without an inner iterative solver whose curvature estimate
   must be frozen while it runs;
3. **summarizable in a small, data-independent structure** — so the curvature can be estimated from
   a long running history of minibatches (the way a diagonal preconditioner can), rather than from a
   single frozen minibatch.

No existing method had all three. Diagonal/low-rank approximations have (2) and (3) but not (1).
The strongest non-diagonal method of the day had (1) but not (2) or (3). A method with all three
could be nearly as cheap as SGD per unit of data while taking the large, well-aimed steps of a true
second-order method.

## Background

**The right metric: natural gradient.** A network with a probabilistic output defines a conditional
distribution `P_{y|x}(θ)`, with density `p(y|x,θ) = r(y|f(x,θ))` for a predictive family `R_{y|z}`
(Gaussian for least squares, multinomial for cross-entropy), so the loss is `L(y,z) = −log r(y|z)`
and minimizing the average loss is maximum-likelihood learning. Amari's natural gradient (Amari,
1998; Amari & Nagaoka, 2000) is steepest descent measured by how much the *distribution* changes
(KL divergence) rather than how much the *parameters* change (Euclidean distance). Its direction is
`F⁻¹∇h`, where

    F = E[ ∇_θ log p · ∇_θ log pᵀ ] = E[ ∇θ ∇θᵀ ]

is the Fisher information matrix, and the expectation is over the input distribution and the
*model's* predictive distribution over `y`. The natural gradient is invariant to how the model is
parameterized, which is exactly the property a curvature-aware optimizer wants. Its problem is size:
`F` is (#params)².

**The Fisher is a second-order matrix.** When the loss is the negative log-likelihood of an
exponential-family `R_{y|z}` with `z` the natural parameters, the Fisher coincides with the
Generalized Gauss-Newton matrix (Schraudolph, 2002; Martens, 2014; Martens & Sutskever, 2012;
Pascanu & Bengio, 2014), a PSD approximation to the Hessian. So "natural-gradient method" and
"second-order method" are two views of the same object, and the natural gradient inherits the entire
optimization toolbox built around quadratic models — trust regions, Levenberg–Marquardt damping,
line searches.

**The structure of the layer gradient.** For a feed-forward net with `s_i = W_i ā_{i-1}`,
`a_i = φ_i(s_i)` (where `ā` appends a constant `1` so the bias is the last column of `W`),
backpropagation gives the per-layer gradient as a single outer product,

    ∇_{W_i} L = g_i ā_{i-1}ᵀ,    g_i = D s_i,

of the backpropagated pre-activation gradient `g_i` and the forward activation `ā_{i-1}`.

**Inverse covariances are sparse even when covariances are not.** There is a classical link
(Pourahmadi, 1999, 2011) between the inverse of a covariance matrix and linear regression: row `i`
of `Σ⁻¹` is, up to scale, the coefficients of the best linear predictor of variable `i` from the
others, so `Σ⁻¹ = D⁻¹(I − B)`. Entries of `Σ⁻¹` are small wherever a variable is not *useful* for
predicting another. This is why an inverse can have a much sparser effective structure than the
dense matrix it inverts — a fact about precision matrices and Gaussian graphical models
(Bishop, 2006).

**The Kronecker product.** For `A ∈ ℝ^{m×n}`, `A ⊗ B` is the block matrix `[A]_{ij} B`. The
identities that make it useful here are `(A⊗B)(C⊗D) = AC⊗BD`, `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`,
`(A⊗B)ᵀ = Aᵀ⊗Bᵀ`, and the vectorization rule `(A⊗B) vec(X) = vec(B X Aᵀ)` with `vec(uvᵀ) = v⊗u`
under column-stacking (Van Loan, 2000).

**Why the strongest existing second-order method is awkward.** Curvature-aware updates make ~10²
updates do what diagonal methods need ~10⁴–10⁵ updates for. But the leading way to get them used an
inner conjugate-gradient solve per update, with the curvature estimate held fixed across the (often
hundreds of) CG iterations. Each CG iteration costs a curvature-matrix-vector product, about as
expensive as a minibatch gradient, and the frozen curvature means the method consumes little data
per update — poor in the stochastic regime. CG is also a first-order method internally, so on
quadratics with spread-out spectra it has no edge over well-tuned momentum (Li, 2005; Sutskever et
al., 2013).

**Damping is not optional.** Any method that takes large steps from a local quadratic model must
guard against the model's breakdown. Tikhonov damping (add `λI` to the curvature) implements a
trust region; Levenberg–Marquardt (Moré, 1978; Nocedal & Wright, 2006) adapts `λ` from the observed
reduction ratio. Powerful second-order methods *fail* without good damping.

## Baselines

**SGD with momentum (Nesterov-style).** The default. Update `θ ← θ − α∇h` augmented with a velocity
term (Polyak, 1964; Plaut, Nowlan & Hinton, 1986; Sutskever et al., 2013). Cheap, stochastic-robust,
and on the standard deep-autoencoder benchmarks momentum is what makes SGD competitive at all. Gap:
no curvature; rate of progress saturates because anisotropic, correlated curvature is ignored.

**Hessian-free / truncated-Newton (Martens, 2010; Martens & Sutskever, 2012; Vinyals & Povey, 2012).**
Minimizes the quadratic model `M(δ) = ½δᵀCδ + ∇hᵀδ` by linear CG, using exact
curvature-matrix-vector products (the Gauss-Newton/Fisher), with adaptive Tikhonov damping.
High-quality non-diagonal updates. Gaps: (i) each update is many expensive mat-vec products;
(ii) the curvature must be frozen during CG, so each update sees only a small fixed minibatch — weak
in the stochastic regime; (iii) the exact curvature has no compact summary, so it cannot be
accumulated over a long data history.

**Diagonal / per-unit / low-rank curvature (Becker & LeCun, 1989; LeCun et al., 1998; Schaul et al.,
2013; Zeiler, 2013; TONGA — Le Roux et al., 2008; Ollivier, 2013).** Invert a diagonal, small-block
(per unit), or low-rank approximation directly. Cheap and online-friendly. Gap: they discard the
off-diagonal curvature that gives second-order methods their power, so they deliver only limited
speedups over SGD with momentum. TONGA's and Ollivier's blocks correspond to a single unit's
weights — small, and there are as many of them as there are units.

**Kronecker-factored block-diagonal Fisher (Heskes, 2000).** The closest prior art: approximate the
layer-`i` Fisher block as a Kronecker product of two small matrices and invert via
`(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`. Heskes computes the gradient-covariance factor exactly, damps with a single
hand-set constant added to each factor, and stops there. Gaps: the hand-set, non-adaptive damping is
fragile across the course of optimization; basic factored damping alone produces poor updates; the
exact factor computation does not scale to networks with many outputs; and only the crudest
(block-diagonal) inverse structure is used.

**Centering / reparameterization methods (Schraudolph, 1998; Raiko et al., 2012; Vatanen et al.,
2013; Wiesler et al., 2014).** Shift activities and local derivatives to be zero-mean so the
gradient better resembles the natural gradient; argued to make the Fisher more diagonal. Gaps:
relies on assuming within-layer correlations are zero (it only centers, never whitens), and usually
needs added skip connections to preserve expressivity.

**Concurrent Kronecker-factored natural gradient (Povey et al., 2015).** Similar block-diagonal
Kronecker factoring, but with the empirical Fisher, only basic factored Tikhonov, no rescaling, no
momentum.

## Evaluation settings

The natural yardstick is the set of deep-autoencoder optimization problems of Hinton &
Salakhutdinov (2006) on the **MNIST**, **CURVES**, and **FACES** datasets — deep narrow autoencoders
that are notoriously hard to optimize and had become the standard benchmark for neural-network
optimizers (Martens, 2010; Vinyals & Povey, 2012; Sutskever et al., 2013). Networks use `tanh`/
logistic units; some layers reach ~2000 units. Light `ℓ2` regularization (coefficient ~`10⁻⁵`) is
added. The figure of merit is *optimization speed*: training-set reconstruction error versus
wall-clock time and versus number of updates/iterations (training error, not test error, since the
interest is optimization, not generalization). The natural baseline is well-tuned SGD with
Nesterov momentum, with learning-rate and momentum schedules calibrated per problem (Sutskever et
al., 2013), under a matched "sparse initialization" (Martens, 2010) and exponentially-decayed
iterate averaging.

## Code framework

The primitives that already exist: an automatic-differentiation library with `Linear`/`Conv2d`
layer modules exposing per-layer parameters and forward/backward hooks, a base optimizer abstraction
with parameter groups and per-parameter state, and a standard training loop. A second-order method
fits as a custom optimizer that hooks each layer to gather curvature statistics, then preconditions
the gradient before stepping. Below is the pre-method scaffold; the one big empty slot is the
curvature object — how per-layer statistics are summarized, inverted, and applied to the gradient.

```python
import torch
import torch.optim as optim

KNOWN_LAYERS = {"Linear", "Conv2d"}  # layers whose gradient is a single outer product g·āᵀ

class CurvatureStats:
    # Per-layer running statistics that summarize curvature in a compact,
    # data-amount-independent structure. TODO: what exactly do we accumulate per layer?
    def update_from_forward(self, layer, layer_input):
        pass  # TODO

    def update_from_backward(self, layer, grad_output):
        pass  # TODO

class SecondOrderOptimizer(optim.Optimizer):
    def __init__(self, model, lr, momentum=0.9, stat_decay=0.95, damping=1e-3,
                 weight_decay=0, t_stats=10, t_inv=100):
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)
        self.model = model
        self.stat_decay = stat_decay      # exp-decay weight for the running curvature average
        self.t_stats = t_stats            # refresh-stats period
        self.t_inv = t_inv                # refresh-inverse period
        self.steps = 0
        self.stats = CurvatureStats()
        self._register_hooks()            # forward/backward hooks on KNOWN_LAYERS

    def _register_hooks(self):
        pass  # TODO: forward-pre-hook -> stats.update_from_forward;
              #       backward-hook    -> stats.update_from_backward

    def _refresh_inverse(self, layer):
        # TODO: turn the accumulated per-layer statistics into whatever form
        #       lets us cheaply apply the inverse curvature to a gradient.
        pass

    def _precondition(self, layer):
        # TODO: map the raw layer gradient to the curvature-corrected update direction.
        pass

    def _rescale(self, updates):
        # TODO: choose a step scale for the proposed update.
        pass

    def step(self, closure=None):
        updates = {}
        for layer in self._layers():            # the KNOWN_LAYERS we hooked
            if self.steps % self.t_inv == 0:
                self._refresh_inverse(layer)
            updates[layer] = self._precondition(layer)
        self._rescale(updates)
        self._apply_with_momentum(updates)      # standard momentum + weight decay + θ ← θ + δ
        self.steps += 1

    def _layers(self):
        pass  # TODO

    def _apply_with_momentum(self, updates):
        pass  # TODO
```
