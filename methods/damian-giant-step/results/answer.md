# Gradient-based representation learning for low-dimensional polynomials

Train a two-layer ReLU network on a low-dimensional polynomial target
`f*(x)=g(<u_1,x>,...,<u_r,x>)`, `x~N(0,I_d)`, by using the first full-batch gradient step to
replace random first-layer rows with empirical gradient features that span the hidden subspace
`S*`, then train only the readout on those fixed features.

The result requires the average Hessian

```
H = E[grad^2 f*(x)]
```

to have rank exactly `r`, so `span(H)=S*`.

## Algorithm

1. Preprocess labels:

   ```
   alpha = (1/n) sum_i y_i
   beta  = (1/n) sum_i y_i x_i
   y_i^res = y_i - alpha - beta.x_i
   ```

2. Symmetric initialization for even width `m`:

   ```
   a_j = -a_{m-j},   w_j = w_{m-j},   b_j = b_{m-j}=0
   a_j in {+-1},    w_j ~ N(0, I_d/d)
   ```

   This makes `f_theta0(x)=0`.

3. First-layer full-batch step:

   ```
   grad_{w_j} L(theta_0)
     = -2 a_j (1/n) sum_i y_i^res x_i 1_{w_j.x_i >= 0}

   W^(1) = W^(0) - eta_1 [grad_W L(theta_0) + lambda_1 W^(0)]
   eta_1 = O~(sqrt(d)),   lambda_1 = eta_1^{-1}
   ```

   Therefore, with

   ```
   g_n(w) = (1/n) sum_i y_i^res x_i sigma'(w.x_i),
   ```

   each row satisfies

   ```
   w_j^(1) = 2 eta_1 a_j g_n(w_j).
   ```

4. Reinitialize hidden biases:

   ```
   b_j ~ N(0,1)
   ```

5. Freeze `W^(1), b`, train only the head `a` with weight-decayed gradient descent.

6. Return the predictor:

   ```
   x -> alpha + beta.x + a^T sigma(W^(1)x + b)
   ```

## Why It Works

For ReLU `sigma(z)=sum_k c_k He_k(z)/k!`, Stein's identity gives, after the preprocessing step,

```
E[y^res x sigma'(w.x)]
  = (C_1-beta)/2 + w(C_0-alpha)/sqrt(2*pi)
    + H w/sqrt(2*pi)
    + higher Hermite contractions.
```

The empirical `alpha,beta` make the first two terms small. Since `f*` depends only on `S*`,
`H w` lies in `S*`; since `rank(H)=r`, random probes supply directions spanning `S*`. The
signal has size `O(d^{-1/2})`, so the method needs `n >= O~(d^2 kappa^2 r)` samples to estimate
it and `eta_1=O~(sqrt(d))` to move the first layer by constant scale.

The theorem gives

```
E|f_thetaT(x)-y| - varsigma
  <= O~( sqrt(d r^p kappa^(2p) / n)
         + sqrt(r^p kappa^(2p) / m)
         + n^(-1/4) ).
```

For transfer to a new degree-`p` polynomial on the same subspace, retrain only the head; the
target sample bound is independent of `d`:

```
O~( sqrt(r^p kappa^(2p) / min(m,N)) + N^(-1/4) ).
```

No public canonical code repository was found, so the code below follows the algorithm and the
sign/constant derivation directly.

## Scaffold Implementation

```python
import math
import types

import torch
import torch.nn.functional as F


def _cfg(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class Strategy:
    def __init__(self, config):
        self.config = config

    def init_two_layer(self, net, config):
        d = net.fc1.in_features
        m = net.fc1.out_features
        if m % 2:
            raise ValueError("The paired-neuron init requires an even width.")

        h = m // 2
        device = net.fc1.weight.device
        dtype = net.fc1.weight.dtype

        with torch.no_grad():
            probes = torch.randn(h, d, device=device, dtype=dtype) / math.sqrt(d)
            signs = torch.empty(h, device=device, dtype=dtype).bernoulli_(0.5)
            signs = signs.mul_(2.0).sub_(1.0)

            net.fc1.weight[:h].copy_(probes)
            net.fc1.weight[h:].copy_(probes)
            net.fc1.bias.zero_()

            net.fc2.weight[0, :h].copy_(signs)
            net.fc2.weight[0, h:].copy_(-signs)
            net.fc2.bias.zero_()

    def make_optimizer(self, net, config):
        # The analyzed first-layer step is full-batch and is executed in finalize().
        return torch.optim.SGD(net.parameters(), lr=0.0)

    def training_step(self, net, optimizer, x, y, step, config):
        return StepMetrics(loss=float(torch.mean(y.view(-1) ** 2).item()), extra={})

    @staticmethod
    def _install_affine_forward(net, alpha, beta):
        alpha = alpha.detach().reshape(1)
        beta = beta.detach().reshape(-1)

        if "_dls_alpha" in net._buffers:
            net._buffers["_dls_alpha"].copy_(alpha)
            net._buffers["_dls_beta"].copy_(beta)
        else:
            net.register_buffer("_dls_alpha", alpha.clone())
            net.register_buffer("_dls_beta", beta.clone())

        def forward_with_affine(module, x):
            features = F.relu(module.fc1(x))
            nonlinear = module.fc2(features)
            affine = module._dls_alpha + x @ module._dls_beta
            return nonlinear + affine.view(-1, 1)

        net.forward = types.MethodType(forward_with_affine, net)

    def finalize(self, net, x_train, y_train, config):
        device = next(net.parameters()).device
        dtype = net.fc1.weight.dtype
        x = x_train.to(device=device, dtype=dtype)
        y = y_train.to(device=device, dtype=dtype).view(-1)
        n = x.shape[0]
        d = x.shape[1]

        eta1 = float(_cfg(config, "first_layer_lr", math.sqrt(d)))
        lambda1 = float(_cfg(config, "first_layer_weight_decay", 1.0 / eta1))

        with torch.no_grad():
            alpha = y.mean()
            beta = (y[:, None] * x).mean(dim=0)
            y_res = y - alpha - x @ beta

            W0 = net.fc1.weight.detach().clone()
            a0 = net.fc2.weight.detach().view(-1).clone()
            gates = (x @ W0.t() >= 0).to(dtype)

            g_n = gates.t().matmul(y_res[:, None] * x) / float(n)
            grad_W = -2.0 * a0[:, None] * g_n

            W1 = W0 - eta1 * (grad_W + lambda1 * W0)
            net.fc1.weight.copy_(W1)
            net.fc1.bias.normal_(0.0, 1.0)
            net.fc2.weight.copy_(a0.view(1, -1))
            net.fc2.bias.zero_()

        for param in net.fc1.parameters():
            param.requires_grad_(False)
        net.fc2.bias.requires_grad_(False)

        head_steps = int(_cfg(config, "head_steps", 2000))
        head_lr = float(_cfg(config, "head_lr", 1e-2))
        head_decay = float(_cfg(config, "head_weight_decay", _cfg(config, "weight_decay", 1e-4)))

        opt = torch.optim.SGD([net.fc2.weight], lr=head_lr, weight_decay=head_decay)
        for _ in range(head_steps):
            opt.zero_grad(set_to_none=True)
            features = F.relu(net.fc1(x))
            pred_res = F.linear(features, net.fc2.weight, None).view(-1)
            loss = torch.mean((pred_res - y_res) ** 2)
            loss.backward()
            opt.step()

        self._install_affine_forward(net, alpha, beta)


def build_strategy(config):
    return Strategy(config)
```
