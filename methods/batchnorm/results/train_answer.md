Training deep networks by mini-batch SGD is fragile: high learning rates diverge, initialization must be babysat, and saturating nonlinearities like sigmoid or tanh quickly die. These three annoyances look separate, but they share one cause. If I write a network as a composition ℓ = F₂(F₁(u, Θ₁), Θ₂), the upper sub-network F₂ sees x = F₁(u, Θ₁) as its input, and every update to Θ₁ shifts the distribution of x. Each layer must keep re-adapting to its own drifting inputs, the effect compounds with depth, and saturating nonlinearities get pushed into their flat tails where gradients vanish. Static fixes such as Xavier initialization or input-only whitening only pin the distribution at the start; ReLU merely treats the symptom by avoiding saturation. Normalization performed as a side step outside the gradient is even worse: a bias added before a centering transform can drift to infinity while the loss stays flat, because the gradient ignores the dependence of the normalization statistics on the parameters. The real need is a cheap, per-step normalization that lives inside the model and is fully differentiable.

The method I propose is Batch Normalization. At every training step, Batch Normalization normalizes each scalar activation to zero mean and unit variance using the statistics of the current mini-batch, then restores representational freedom with a learned per-feature scale γ and shift β. Because the normalization is part of the forward computation, backprop accounts for both the direct dependence on the activations and the dependence of the batch statistics on those activations; the optimizer and the normalizer no longer fight. The transform is applied to the pre-activation x = Wu + b, just before the nonlinearity. The bias b can be dropped entirely, because subtracting the batch mean cancels any constant offset and β subsumes the bias's old role. For convolutional layers, statistics are pooled jointly over the batch and all spatial locations of each feature map, with one (γ, β) per channel, so translation equivariance and weight sharing are preserved.

During training, for a mini-batch B = {x₁,…,x_m}, Batch Normalization computes μ_B = (1/m) Σᵢ xᵢ, σ²_B = (1/m) Σᵢ (xᵢ − μ_B)², then x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε), and finally yᵢ = γ x̂ᵢ + β. The small constant ε prevents division by zero when a feature is nearly constant across the batch. The backward pass sums three chain-rule paths into each input: direct through x̂ᵢ, indirect through σ²_B, and indirect through μ_B. At inference, the output must be deterministic and independent of batch-mates, so the layer switches to frozen population estimates: x̂ = (x − E[x]) / √(Var[x] + ε). These are typically tracked as running averages during training and then fixed. The whole inference transform collapses to a single affine map in x, costing one multiply-add per activation.

Batch Normalization unlocks higher learning rates through a scale-invariance property. For positive a, BN((aW)u) ≈ BN(Wu) when ε is negligible; scaling the weights cancels because both the pre-activation and its batch standard deviation scale by a. Consequently, the gradient flowing down to lower layers is insensitive to this layer's weight scale, and larger weight scales receive smaller gradients, so parameter growth self-stabilizes instead of running away. The stable, near-origin pre-activation distribution also makes saturating nonlinearities trainable again, and the batch-dependent jitter in μ_B and σ²_B acts as a free source of regularization similar to Dropout. The implementation below follows the standard NumPy layer contract: `batchnorm_forward` and `batchnorm_backward` for fully-connected pre-activations, and `spatial_batchnorm_forward` and `spatial_batchnorm_backward` for convolutional feature maps.

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mu = 1.0 / N * np.sum(x, axis=0)
        xmu = x - mu
        carre = xmu ** 2
        var = 1.0 / N * np.sum(carre, axis=0)
        sqrtvar = np.sqrt(var + eps)
        invvar = 1.0 / sqrtvar
        va2 = xmu * invvar
        va3 = gamma * va2
        out = va3 + beta
        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var
        cache = (mu, xmu, carre, var, sqrtvar, invvar, va2, va3,
                 gamma, beta, x, bn_param)
    elif mode == 'test':
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * xhat + beta
        cache = (running_mean, running_var, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dva2 = gamma * dout
    dgamma = np.sum(va2 * dout, axis=0)

    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)
    dsqrtvar = -1.0 / (sqrtvar ** 2) * dinvvar
    dvar = 0.5 * (var + eps) ** (-0.5) * dsqrtvar
    dcarre = 1.0 / N * np.ones(carre.shape) * dvar
    dxmu += 2 * xmu * dcarre

    dx = dxmu
    dmu = -np.sum(dxmu, axis=0)
    dx += 1.0 / N * np.ones(dout.shape) * dmu
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps) ** (-0.5) * dout, axis=0)
    dx = (1.0 / N) * gamma * (var + eps) ** (-0.5) * (
            N * dout
            - np.sum(dout, axis=0)
            - (x - mu) * (var + eps) ** (-1.0) * np.sum(dout * (x - mu), axis=0))
    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        mu = (1.0 / (N * H * W) * np.sum(x, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        var = (1.0 / (N * H * W) * np.sum((x - mu) ** 2, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        running_mean = momentum * running_mean + (1.0 - momentum) * np.squeeze(mu)
        running_var = momentum * running_var + (1.0 - momentum) * np.squeeze(var)
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    elif mode == 'test':
        mu = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    mu, var, x, xhat, gamma, beta, bn_param = cache
    N, C, H, W = x.shape
    eps = bn_param.get('eps', 1e-5)
    gamma = gamma.reshape(1, C, 1, 1)

    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3))

    Nt = N * H * W
    dx = (1.0 / Nt) * gamma * (var + eps) ** (-0.5) * (
            Nt * dout
            - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
            - (x - mu) * (var + eps) ** (-1.0)
              * np.sum(dout * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1))
    return dx, dgamma, dbeta
```
