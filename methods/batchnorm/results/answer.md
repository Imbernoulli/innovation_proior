# Batch Normalization, distilled

Batch Normalization (BN) is a layer inserted into a deep network that, at every training step, normalizes a chosen activation to zero mean and unit variance using the statistics of the current mini-batch, then restores representational freedom with a learned per-feature scale γ and shift β. Because the normalization is part of the model and differentiable, the optimizer accounts for it; this stabilizes the distribution of each layer's inputs throughout training (combating *internal covariate shift*), which lets the network use much higher learning rates, be far less sensitive to initialization, train with saturating nonlinearities (sigmoid/tanh), and often drop Dropout.

## The problem

In a deep net ℓ = F₂(F₁(u, Θ₁), Θ₂), the upper part F₂ sees x = F₁(u, Θ₁) as its input, and a gradient step on Θ₂ is identical to the step a stand-alone net fed x would take. Every update to Θ₁ shifts the distribution of x, so each layer continually re-adapts to its own drifting inputs — an effect that compounds with depth and pushes saturating nonlinearities into their flat, zero-gradient tails. The fix must keep each layer's input distribution stable during training, be cheap enough to run every step, and be folded into the gradient so the optimizer cannot fight it.

## The design decisions and why each

- **Normalization must live inside the model (in the gradient).** If normalization is a side step outside the gradient, it breaks: for x = u + b centered by x̂ = x − E[x], a step that ignores ∂E[x]/∂b gives (u+b+Δb) − E[u+b+Δb] = u+b − E[u+b] — output and loss unchanged while b grows without bound and the model blows up. So the transform must be differentiable and inside the model, carrying both ∂Norm/∂x and ∂Norm/∂X.
- **Per-dimension, not full whitening.** Joint whitening needs the d×d covariance, its inverse square root Cov[x]^{−1/2} (O(d³) via eigendecomposition, not everywhere differentiable, and its derivative for backprop), recomputed over the whole set every step — and the covariance is singular when the batch is smaller than d. Normalizing each scalar feature independently is cheap, closed-form differentiable, never forms a covariance, and still speeds convergence even without decorrelation (LeCun et al., 1998).
- **Learnable γ, β.** Pure normalization constrains each pre-activation to mean 0 / variance 1 (e.g. pinning a sigmoid to its near-linear regime), shrinking what the layer can represent. The affine y = γ x̂ + β restores any mean/variance, and at γ = √(Var[x]+ε), β = E[x] it is exactly the identity — so BN never reduces capacity.
- **Mini-batch statistics in training, fixed statistics at inference.** Using the batch's own mean/variance makes the statistics a differentiable function of the batch (the gradient owns them for free) and injects regularizing noise. At inference the output must be deterministic and independent of batch-mates. A separate population-estimation pass can use the unbiased variance Var[x] = (m/(m−1))·E_B[σ²_B]; the working code below follows the common cs231n-style implementation and stores a moving average of the biased batch variance σ²_B directly.
- **Convolutional BN.** Convolutions share one filter across all spatial locations, so all locations of a feature map must be normalized identically to keep translation equivariance. BN therefore pools each feature map jointly over the batch and all spatial locations (effective size m′ = m·p·q) with one (γ, β) per feature map; per-location normalization would break the weight-sharing symmetry.
- **Placement and dropped bias.** Apply BN to the pre-activation: z = g(BN(Wu)). The pre-activation Wu+b is "more Gaussian" (Hyvärinen & Oja, 2000) than the post-nonlinearity u, so matching its first two moments actually stabilizes it. The bias b is dropped — the mean subtraction cancels it and β subsumes its role.

## The algorithm

**BN transform** (per activation, over a mini-batch B = {x₁…x_m}):

    μ_B  = (1/m) Σ_i x_i
    σ²_B = (1/m) Σ_i (x_i − μ_B)²
    x̂_i  = (x_i − μ_B) / sqrt(σ²_B + ε)
    y_i  = γ x̂_i + β  ≡  BN_{γ,β}(x_i)

**Backward** (three paths into x_i — direct, via σ²_B, via μ_B):

    ∂ℓ/∂x̂_i  = ∂ℓ/∂y_i · γ
    ∂ℓ/∂σ²_B = Σ_i ∂ℓ/∂x̂_i · (x_i−μ_B) · (−½)(σ²_B+ε)^{−3/2}
    ∂ℓ/∂μ_B  = (Σ_i ∂ℓ/∂x̂_i · −(σ²_B+ε)^{−1/2}) + ∂ℓ/∂σ²_B · (Σ_i −2(x_i−μ_B))/m
    ∂ℓ/∂x_i  = ∂ℓ/∂x̂_i (σ²_B+ε)^{−1/2} + ∂ℓ/∂σ²_B · 2(x_i−μ_B)/m + ∂ℓ/∂μ_B · 1/m
    ∂ℓ/∂γ    = Σ_i ∂ℓ/∂y_i · x̂_i
    ∂ℓ/∂β    = Σ_i ∂ℓ/∂y_i

Collapsing the three paths gives the simplified single expression
dx = (γ (σ²+ε)^{−1/2} / m)·(m·dy − Σ dy − x̂·Σ(dy·x̂)).

**Inference.** Replace batch statistics with frozen *population* estimates: x̂ = (x − E[x]) / sqrt(Var[x] + ε). Per-batch test statistics are non-deterministic (and degenerate to a constant for a batch of one), so freeze the population mean/variance instead. The biased batch variance underestimates by exactly E[σ²_B] = ((m−1)/m)·σ² (the sample mean wobbles by σ²/m, telescoping σ²_B = (1/m)Σ(xᵢ−μ)² − (μ_B−μ)²), so the unbiased population estimate from fixed-size batches is Var[x] = (m/(m−1))·E_B[σ²_B]; the NumPy code below instead mirrors the common running-stat implementation and tracks `running_var` as a moving average of the biased σ²_B. BN then collapses to one affine map (zero added inference cost):
y = (γ/sqrt(Var[x]+ε))·x + (β − γE[x]/sqrt(Var[x]+ε)).

**Higher learning rates / scale invariance.** For a > 0, ignoring ε or when ε is negligible relative to a²σ²_B, scaling W by a scales the value and the batch std by a (and the mean by a), which cancel in the normalizer: BN((aW)u) = BN(Wu). So BN(Wu) is homogeneous of degree 0 in the positive weight scale. Hence ∂BN((aW)u)/∂u = ∂BN(Wu)/∂u (the backward signal to lower layers is independent of this layer's weight scale) and, by differentiating the homogeneity identity, ∂BN((aW)u)/∂(aW) = (1/a)·∂BN(Wu)/∂W (larger positive weight scales get smaller gradients); Euler's identity gives ⟨∇_W BN, W⟩ = 0, so the gradient is orthogonal to the radial direction and only rotates W rather than pumping its length. Parameter growth self-stabilizes, killing the runaway that makes high learning rates diverge in plain nets. Heuristically, between two normalized layers ẑ = F(x̂) ≈ Jx̂ with unit covariances forces I = JCov[x̂]Jᵀ = JJᵀ, so the layer Jacobians are driven toward singular values near 1 (gradient scales like sᴸ through L layers, fatal unless s ≈ 1).

**Regularization.** Each example's representation depends on its batch-mates through μ_B and σ²_B, injecting batch-dependent noise that regularizes (much as Dropout does), so Dropout can be reduced or removed.

## Working code (NumPy)

Each function below is the body of one network layer obeying the harness's `forward`/`backward` contract: insert `batchnorm_forward` after an affine pre-activation and before its nonlinearity, with any affine bias redundant once β is present; use `spatial_batchnorm_forward` after a convolution. γ and β are trained by the same optimizer as the weights.

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var  = bn_param.get('running_var',  np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mu      = 1.0 / N * np.sum(x, axis=0)
        xmu     = x - mu
        carre   = xmu ** 2
        var     = 1.0 / N * np.sum(carre, axis=0)   # biased batch variance
        sqrtvar = np.sqrt(var + eps)
        invvar  = 1.0 / sqrtvar
        va2     = xmu * invvar                 # x_hat
        va3     = gamma * va2
        out     = va3 + beta                   # y = gamma*x_hat + beta
        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var  = momentum * running_var  + (1.0 - momentum) * var
        cache = (mu, xmu, carre, var, sqrtvar, invvar, va2, va3,
                 gamma, beta, x, bn_param)
    elif mode == 'test':
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out  = gamma * xhat + beta
        cache = (running_mean, running_var, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var']  = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """Staged backward, mirroring the per-path chain rule."""
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    dbeta  = np.sum(dout, axis=0)
    dva2   = gamma * dout
    dgamma = np.sum(va2 * dout, axis=0)

    dxmu    = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)
    dsqrtvar = -1.0 / (sqrtvar ** 2) * dinvvar
    dvar    = 0.5 * (var + eps) ** (-0.5) * dsqrtvar
    dcarre  = 1.0 / N * np.ones(carre.shape) * dvar
    dxmu   += 2 * xmu * dcarre

    dx   = dxmu
    dmu  = -np.sum(dxmu, axis=0)
    dx  += 1.0 / N * np.ones(dout.shape) * dmu
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Algebraically collapsed backward (single-line dx)."""
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta  = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps) ** (-0.5) * dout, axis=0)
    dx = (1.0 / N) * gamma * (var + eps) ** (-0.5) * (
            N * dout
            - np.sum(dout, axis=0)
            - (x - mu) * (var + eps) ** (-1.0) * np.sum(dout * (x - mu), axis=0))
    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Convolutional BN: pool statistics per channel over batch + space."""
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var  = bn_param.get('running_var',  np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        mu  = (1.0 / (N*H*W) * np.sum(x, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        var = (1.0 / (N*H*W) * np.sum((x - mu) ** 2, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)
        out  = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        running_mean = momentum * running_mean + (1.0 - momentum) * np.squeeze(mu)
        running_var  = momentum * running_var  + (1.0 - momentum) * np.squeeze(var)
        bn_param['running_mean'] = running_mean
        bn_param['running_var']  = running_var
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    elif mode == 'test':
        mu  = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)
        out  = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    mu, var, x, xhat, gamma, beta, bn_param = cache
    N, C, H, W = x.shape
    eps = bn_param.get('eps', 1e-5)
    gamma = gamma.reshape(1, C, 1, 1)

    dbeta  = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3))

    Nt = N * H * W
    dx = (1.0 / Nt) * gamma * (var + eps) ** (-0.5) * (
            Nt * dout
            - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
            - (x - mu) * (var + eps) ** (-1.0)
              * np.sum(dout * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1))
    return dx, dgamma, dbeta
```

The forward staging mirrors the BN transform step for step; `batchnorm_backward` realizes the per-path chain rule (direct, via σ²_B, via μ_B); `batchnorm_backward_alt` is the collapsed single-line form; the test branch and `spatial_batchnorm_*` implement inference from frozen running statistics and the convolutional per-feature-map joint normalization. The code uses cs231n-style moving averages of biased batch variances; an exact fixed-batch population-estimation pass would apply the m/(m−1) correction.
