We want to understand why a two-layer ReLU network trained by gradient descent can beat the kernel that arises from linearizing that same network at initialization, and the cleanest setting to isolate this is Gaussian regression with hidden low-dimensional structure: $x \sim N(0, I_d)$ and a normalized degree-$p$ target $f^*(x) = g(\langle u_1, x\rangle, \dots, \langle u_r, x\rangle)$ with $r \ll d$. The target depends only on the subspace $S^* = \mathrm{span}(u_1,\dots,u_r)$, but the input distribution is isotropic, so the relevant directions can only be found through label–feature correlations. If I freeze the first-layer features at random initialization I have exactly a kernel method, and a kernel can fit functions in the span of its fixed random features but has no mechanism to discover that $f^*$ lives on $S^*$. For a degree-$p$ polynomial this is the wrong scaling: random-feature and NTK regression pay for the ambient polynomial space of size $\Theta(d^p)$, and generic degree-$p$ regression in $d$ variables has feature dimension $\Theta(d^p)$ outright, ignoring the promise entirely. Specialized low-rank polynomial algorithms — filtered PCA plus geodesic optimization in the style of Chen and Meka — prove the statistical structure is exploitable, but they are tailored spectral procedures, not ordinary gradient descent on the very network whose representation we are studying. Online single-index SGD theory classifies link difficulty by an information exponent but is not a finite-sample analysis of a fixed multi-neuron ReLU net learning a reusable representation. The only useful question, then, is whether the gradient on the first layer already contains a recoverable signal for $S^*$, and at what price in samples and step size.

I propose a gradient-based representation-learning recipe — the Damian giant-step method — in which a single, full-batch, large-step first-layer update replaces the random first-layer rows with empirical gradient features that span $S^*$, after which only the readout is trained on those now-fixed features. The object that makes this possible is the average Hessian $$H = \mathbb{E}[\nabla^2 f^*(x)].$$ Because every derivative of a function of $\langle u_i, x\rangle$ stays inside $S^*$, the column span of $H$ lies inside $S^*$; the method requires the nondegeneracy condition that $\mathrm{rank}(H) = r$, so that $\mathrm{span}(H) = S^*$ exactly and second-order information sees the whole subspace. Before touching the network gradient I strip off the parts of the target that need no learned nonlinearity: I compute $\alpha = \frac1n\sum_i y_i$ and $\beta = \frac1n\sum_i y_i x_i$ and work with residual labels $y_i^{\mathrm{res}} = y_i - \alpha - \beta\cdot x_i$, restoring $\alpha + \beta\cdot x$ in the final predictor. This is not cosmetic: in the Hermite expansion the constant $C_0$ and linear $C_1$ pieces sit ahead of the Hessian term, and if I leave them in they contaminate the first-layer gradient.

The mechanism turns on a symmetric initialization. I pair neurons so that $a_j = -a_{m-j}$, $w_j = w_{m-j}$, and $b_j = b_{m-j} = 0$, with $a_j \in \{\pm 1\}$ and $w_j \sim N(0, I_d/d)$. The cancellation makes the network output identically zero at initialization, $f_{\theta_0}(x) = 0$ for every $x$, which is precisely what cleans up the gradient. Under the square loss $L = \frac1n\sum_i (f_\theta(x_i) - y_i)^2$, the empirical gradient of one first-layer row at initialization is $$\nabla_{w_j} L(\theta_0) = -2\,a_j\,\frac1n\sum_i y_i^{\mathrm{res}}\,x_i\,\mathbf{1}_{w_j\cdot x_i \ge 0},$$ where the residual label enters with a minus sign because $f_{\theta_0} = 0$ and the indicator is the ReLU derivative gate. Writing the gated empirical feature $g_n(w) = \frac1n\sum_i y_i^{\mathrm{res}}\,x_i\,\sigma'(w\cdot x_i)$, this is $\nabla_{w_j} L(\theta_0) = -2\,a_j\,g_n(w_j)$. The first update uses weight decay tuned to $\lambda_1 = \eta_1^{-1}$, so that $$w_j^{(1)} = w_j^{(0)} - \eta_1\big(\nabla_{w_j} L + \eta_1^{-1} w_j^{(0)}\big) = -\eta_1\,\nabla_{w_j} L = 2\,\eta_1\,a_j\,g_n(w_j).$$ The decay is doing real work here, not merely regularizing: it exactly cancels the original random row and leaves a clean scaled gradient feature. This is why an implementation that overwrites rows with a separately estimated subspace direction would be a different algorithm — the method must compute this gated empirical gradient, or an exactly equivalent quantity.

What licenses calling $g_n$ a feature pointing into $S^*$ is its population content. Expanding ReLU as $\sigma(z) = \sum_k c_k He_k(z)/k!$ so that $\sigma'(z) = \sum_k c_{k+1} He_k(z)/k!$, Stein's identity gives, after preprocessing, $$\mathbb{E}\!\left[y^{\mathrm{res}}\,x\,\sigma'(w\cdot x)\right] = \frac{C_1 - \beta}{2} + \frac{w\,(C_0 - \alpha)}{\sqrt{2\pi}} + \frac{H w}{\sqrt{2\pi}} + \text{higher Hermite contractions},$$ with the higher terms being even-Hermite contractions $\sum_{k\ge 2} c_{2k} C_{2k}(w^{\otimes(2k-1)})/(2k-1)! + w\sum_{k\ge 1} c_{2k+2} C_{2k}(w^{\otimes 2k})/(2k)!$. The constant $1/\sqrt{2\pi}$ on the Hessian term is the ReLU coefficient $c_2 = 1/\sqrt{2\pi}$. The empirical $\alpha, \beta$ concentrate and kill the first two terms, and the remaining contractions are smaller by powers of the random overlap of $w$ with $S^*$, so the leading informative feature is $H w / \sqrt{2\pi}$. Since $\mathrm{rank}(H) = r$, a population of random probes $w_j$ produces projected directions that together span $S^*$.

Two prices follow directly, and they fix the design. A random row has overlap $O(d^{-1/2})$ with any fixed relevant direction, so $\|H w\|$ is only $O(d^{-1/2})$ at constant target scale; making the Hessian signal dominate the empirical fluctuation costs $n \ge \tilde O(d^2 \kappa^2 r)$ samples, which is why the analyzed first-layer step must be full-batch — a mini-batch update is simply not the object under analysis. Because the per-row signal is $O(d^{-1/2})$, an $O(1)$ learning rate barely moves the row; to push the first layer into the feature-learning regime in one step I need the giant step $\eta_1 = \tilde O(\sqrt{d})$. There is no width normalization here because the readout uses $a_j \in \{\pm 1\}$ rather than $1/\sqrt{m}$, and there is no branch on an information exponent — the method assumes the Hessian signal is present and analyzes that signal; if the rank condition fails, a CSQ lower-bound argument shows the same construction cannot simply be rescued with a third-Hermite estimator. After this single step the rows are scaled copies of $g_n(w_j)$ whose population part lies in $S^*$, and to get a rich family of scalar nonlinear features on that learned subspace I reinitialize the hidden biases as independent $N(0,1)$ draws; that random threshold spread is part of the construction, not an interchangeable engineering choice. With $W^{(1)}$ and $b$ frozen, the remaining task is linear regression over the ReLU features $\sigma(W^{(1)} x + b)$ with weight decay on the head, justified by the equivalence between weight-decayed least squares and a norm-constrained readout, and the predictor returns $x \mapsto \alpha + \beta\cdot x + a^\top \sigma(W^{(1)} x + b)$. The resulting guarantee, requiring $n \ge \tilde O(d^2 \kappa^2 r)$ and $d \ge \tilde O(\kappa r^{3/2})$, is $\mathbb{E}|f_{\theta_T}(x) - y| - \varsigma \le \tilde O\big(\sqrt{d r^p \kappa^{2p}/n} + \sqrt{r^p \kappa^{2p}/m} + n^{-1/4}\big)$; and because the representation is reusable, transfer to a new degree-$p$ polynomial on the same subspace requires retraining only the head, with target sample complexity independent of $d$, $\tilde O\big(\sqrt{r^p \kappa^{2p}/\min(m,N)} + N^{-1/4}\big)$.

One implementation subtlety is worth stating plainly: the fixed `TwoLayerMLP` forward has no skip connection, so to add $\alpha + \beta\cdot x$ back into the prediction the strategy must patch the forward method after finalization. The mini-batch callback is a no-op because the analyzed first step is full-batch and lives in `finalize`, where I form $\alpha$, $\beta$, and residual labels, compute the gated empirical first-layer gradient at the symmetric initialization, apply $W_1 = W_0 - \eta_1(\nabla_W + \eta_1^{-1} W_0)$, resample Gaussian biases, train only the head with weight decay, and install the affine residual into the forward pass.

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
