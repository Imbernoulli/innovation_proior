The puzzle I want to resolve is the stubborn mismatch between two pictures of a trained neural network that refuse to be reconciled. Training is gradient descent on the parameters, and the loss as a function of those parameters is hopelessly non-convex — a high-dimensional surface strewn with saddle points, with no convexity to lean on — and yet gradient descent reliably drives the training loss to zero on networks far wider than the data demands. At the same time the same overparametrized network, which can be made to memorize random labels, generalizes when the labels are real. Capacity arguments explain neither half: capacity says it should overfit, and it does interpolate the training set, yet it still generalizes. The telling clue is that plain kernel methods do exactly this too — they interpolate and still generalize. The existing tools each catch only a fragment of the phenomenon. The infinite-width Gaussian-process picture (Neal, 1996; Lee et al., 2018; Matthews et al., 2018) shows that a wide network *at initialization* is a centered Gaussian process with a covariance given by a layer-wise recursion, but it describes only the untrained prior and says nothing about the trajectory gradient descent traces. Rahimi & Recht's (2007) random-feature model is the one clean case where training provably *is* kernel gradient descent against a fixed kernel — but only because that model is linear in its parameters, so its feature map never moves; a real network is nonlinear in its weights and its feature map wanders as training proceeds. Cho & Saul's (2009) arc-cosine kernels and Mei et al.'s (2018) mean-field analysis give deep closed-form kernels and two-layer training dynamics respectively, but neither connects a fixed kernel to the dynamics of a *deep* network trained by gradient descent. So I suspect I am watching the wrong object. I should stop tracking the weights and track the function.

I propose the Neural Tangent Kernel. The realization map $F$ sends parameters $\theta$ to the function $f_\theta$, and what actually generalizes — what the cost is really a functional of — is $f_\theta$ itself, a point moving through function space, where the squared-error cost is convex; the non-convexity is entirely an artifact of the nonlinear parametrization. So I push the parameter dynamics through $F$. Under continuous-time gradient flow $\dot\theta_p = -\partial_{\theta_p}(C\circ F)$, and writing the functional derivative $\partial_f C$ as pairing against a dual function $d$ in the data inner product $\langle g,h\rangle_{p_{in}}=\mathbb{E}_{x\sim p_{in}}[g(x)^\top h(x)]$ (for squared error $C(f)=\tfrac12\|f-f^*\|^2$ this dual is simply $d=f-f^*$), the chain rule gives $\dot\theta_p=-\langle d,\partial_{\theta_p}f_\theta\rangle_{p_{in}}$. The function then changes only because $\theta$ does, $\partial_t f_\theta=\sum_p(\partial_{\theta_p}f_\theta)\dot\theta_p=-\sum_p\langle d,\partial_{\theta_p}f_\theta\rangle_{p_{in}}\,\partial_{\theta_p}f_\theta$. The right-hand side is a sum, over parameters, of (how much $\partial_{\theta_p}f$ overlaps the error direction $d$) times (that same $\partial_{\theta_p}f$) — exactly an operator built from the family $\{\partial_{\theta_p}f\}$. That operator is the kernel I name the tangent kernel,
$$\Theta(x,x')=\sum_p\partial_{\theta_p}f_\theta(x)\otimes\partial_{\theta_p}f_\theta(x')=\sum_p\langle\nabla_\theta f(x),\nabla_\theta f(x')\rangle,$$
the Gram matrix of the tangent map of $F$. With it the function obeys $\partial_t f_\theta(x)=-\tfrac1N\sum_j\Theta(x,x_j)d(x_j)$, i.e. it follows *kernel gradient descent* against $\Theta$; for squared error evaluated on the data this collapses to $\dot f=-\Theta(f-f^*)$. The payoff is immediate: $\partial_t C=-\|d\|_\Theta^2\le 0$, strictly negative whenever $d\ne 0$ as seen by the kernel, so if $\Theta$ is positive definite the convex function-space cost decreases to its global minimum — the non-convex parameter landscape never enters. But this is hollow while $\Theta=\Theta(\theta)$ is random at initialization and drifts as $\theta$ moves. The whole method rests on two facts I must establish: that $\Theta$ at initialization concentrates on a deterministic kernel as width grows, and that $\Theta$ stays frozen during training.

The parametrization is the load-bearing design choice. I write each pre-activation as $\tilde a^{(\ell+1)}=\tfrac{1}{\sqrt{n_\ell}}W^{(\ell)}a^{(\ell)}+\beta\,b^{(\ell)}$ with all entries of $W^{(\ell)},b^{(\ell)}$ iid $\mathcal{N}(0,1)$ and $a^{(\ell)}=\sigma(\tilde a^{(\ell)})$. The $\tfrac{1}{\sqrt{n_\ell}}$ factor renormalizes a sum of $n_\ell$ order-one terms back to $O(1)$ so the wide limit exists; the textbook small-weight init represents the same functions, but its *derivatives* differ, and that is the point — $\partial_{W^{(\ell)}_{ij}}f$ picks up a $\tfrac{1}{\sqrt{n_\ell}}$, so each individual weight in a wide layer has a vanishing gradient and moves negligibly. The scalar $\beta$ is there because, with connection-weight gradients shrunk by $\tfrac{1}{\sqrt{n}}$, the bias gradients (which carry no such factor) would otherwise dominate; $\beta$ rebalances biases against connections so neither is starved, with something like $\beta=0.1$ and a correspondingly large learning rate behaving like a classical moderate-width net. Taking widths to infinity one layer at a time and inducting on depth, the covariance of the network function follows $\Sigma^{(1)}(x,x')=\tfrac1{n_0}x^\top x'+\beta^2$ and $\Sigma^{(L+1)}(x,x')=\mathbb{E}_{f\sim\mathcal{N}(0,\Sigma^{(L)})}[\sigma(f(x))\sigma(f(x'))]+\beta^2$ — the Gaussian-process fact, recovered as a byproduct, since each layer's empirical second moment $\tfrac1{n_L}\sum_i\sigma(\tilde a^{(L)}_i(x))\sigma(\tilde a^{(L)}_i(x'))$ concentrates on its Gaussian expectation by the law of large numbers. Carrying the same bookkeeping through the tangent kernel, the bottom layer gives $\Theta^{(1)}_\infty=\Sigma^{(1)}$, and splitting an $(L+1)$-network's parameters into the first $L$ layers and the last layer yields two contributions: the last layer's own weights contribute $\Sigma^{(L+1)}$, and the lower layers, after the induction hypothesis collapses the smaller network's kernel to $\Theta^{(L)}_\infty\delta_{ii'}$ and the LLN turns $W^{(L)}_{ik}W^{(L)}_{ik'}$ into $\delta_{kk'}$, contribute $\Theta^{(L)}_\infty\dot\Sigma^{(L+1)}$ with $\dot\Sigma^{(L+1)}(x,x')=\mathbb{E}_{f\sim\mathcal{N}(0,\Sigma^{(L)})}[\dot\sigma(f(x))\dot\sigma(f(x'))]$. Adding them,
$$\Theta^{(L+1)}_\infty(x,x')=\Theta^{(L)}_\infty(x,x')\,\dot\Sigma^{(L+1)}(x,x')+\Sigma^{(L+1)}(x,x'),$$
a deterministic kernel depending only on depth, nonlinearity, and $\beta$ — the second summand is the top layer learning, the first is the gradient back-propagated into all lower layers, modulated by how $\sigma$ passes gradients.

The harder claim is that $\Theta$ does not move during training. Writing the training as a general direction $d_t$ with the only assumption that $\int_0^T\|d_t\|_{p_{in}}\,dt$ stays stochastically bounded (automatic for squared error, since $\|f-f^*\|$ decreases), I again induct on depth. Splitting off the last layer, the subnetwork sees a back-propagated direction $d'_t=\dot\sigma(\tilde a^{(L)})(\tfrac{1}{\sqrt{n_L}}W^{(L)})^\top d_t$; because $\sigma$ is $c$-Lipschitz, controlling its forcing reduces to bounding $\|\tfrac{1}{\sqrt{n_L}}W^{(L)}(t)\|_{op}$ along the whole trajectory. The last layer's weights and the hidden pre-activations both evolve with an explicit $\tfrac{1}{\sqrt{n_L}}$ out front, so I couple their drifts in a single aggregate $A(t)$ whose derivative is bounded by $\tfrac{\max\{c^2\|\Theta^{(L)}_\infty\|_{op},1\}}{\sqrt{n_L}}\|d_t\|_{p_{in}}A(t)$; Grönwall then gives $A(t)\le A(0)\exp(\cdots)$ with an exponent that vanishes in probability, so every scaled weight displacement and every pre-activation drifts at rate $O(1/\sqrt{n_L})$. The supporting lemma — that $\|\tfrac{1}{\sqrt{n_\ell}}(W^{(\ell)}(t)-W^{(\ell)}(0))\|_{op}\to 0$ for *every* layer simultaneously — comes from back-propagating the same Grönwall bookkeeping through the whole stack: the $c$-Lipschitz bound gives $\|d^{(\ell)}_t\|\le c^{L+1-\ell}(\prod_k w^{(k)})\|d_t\|$, the kernel recursion bounds $\|\Theta^{(\ell)}\|_{op}$ by a positive-coefficient polynomial in the scaled norms, and a nonlinear Grönwall inequality with a $1/\sqrt{\min_k n_k}$ prefactor keeps the aggregate bounded and forces its derivative to zero. The crucial input here is only that the initial scaled operator norms and activations are *bounded* (tight Gaussian norms; activations converge by the GP limit), not that they vanish. Once parameters barely move, every time-varying factor of $\Theta^{(L+1)}$ — the top-layer activations, the connection weights $W^{(L)}_{ij}$, and $\dot\sigma(\tilde a^{(L)}_i)$ (whose drift is controlled because $\sigma$ has bounded second derivative) — moves at $1/\sqrt{n_L}$, so $\Theta^{(L+1)}(t)\to\Theta^{(L+1)}_\infty$ uniformly on $[0,T]$. This looks paradoxical, since the point of hidden layers is to learn representations, and the resolution is a counting argument: each of the $n_\ell$ neurons drifts at $1/\sqrt{n_\ell}$, but the function depends on their $\tfrac{1}{\sqrt{n_\ell}}$-weighted aggregate, and $n_\ell$ coherent $1/\sqrt{n}$ drifts make an $O(1)$ effect on the function — precisely the lower-layer term $\Theta^{(L)}_\infty\dot\Sigma^{(L+1)}$ in the recursion. The layers learn in aggregate while no single neuron moves appreciably, and the pre-activations stay Gaussian throughout.

With $\Theta$ frozen, convergence needs $\Theta^{(L)}_\infty$ positive definite. For the case that matters for high-dimensional data — inputs on the sphere, where points share a norm and the kernel is a function of $x^\top x'$ — I prove this for non-polynomial Lipschitz $\sigma$ and $L\ge 2$. Since $\Theta^{(L+1)}_\infty=\dot\Sigma^{(L+1)}\Theta^{(L)}_\infty+\Sigma^{(L+1)}$ and the Schur product of PSD kernels is PSD, it suffices that the covariances $\Sigma^{(L)}$ are PD; positive-definiteness propagates up the recursion because $\sum_{ij}c_ic_j\Sigma^{(L+1)}(x_i,x_j)=\mathbb{E}[(\sum_i c_i\sigma(f(x_i)))^2]+(\beta\sum_i c_i)^2$ vanishes only when all $c_i=0$ (the Gaussian vector is non-degenerate and $\sigma$ non-constant), reducing everything to $\Sigma^{(2)}$. There, rescaling to unit variance gives $\Sigma^{(2)}(x,x')=\hat\mu(\rho)+\beta^2$ with $\rho=(n_0\beta^2+x^\top x')/(n_0\beta^2+1)$ and $\hat\mu(\rho)=\sum_i a_i^2\rho^i$ the Daniely et al. (2016) Hermite dual; non-polynomial $\sigma$ gives infinitely many nonzero $a_i$, and the positive shift $n_0\beta^2$ from $\beta>0$ spreads each high term across both even and odd powers of $x^\top x'$, so by the Schoenberg/Gneiting (2013) sphere criterion $\Sigma^{(2)}$ — hence every $\Sigma^{(L)}$ and $\Theta^{(L)}_\infty$ for $L\ge 2$ — is positive definite. Solving the least-squares dynamics, $f_t=f^*+e^{-t\Pi}(f_0-f^*)$ with $\Pi$ the empirical kernel operator; diagonalizing $\Pi$ by its kernel principal components with eigenvalues $\lambda_i$, each component relaxes as $e^{-\lambda_i t}$, so large-eigenvalue (smooth) directions are fit first and small-eigenvalue (noisy) directions last — which is exactly why early stopping helps. At convergence the mean predictor $f_{\infty,k}(x)=\kappa_{x,k}^\top\tilde K^{-1}y^*+(f_0(x)-\kappa_{x,k}^\top\tilde K^{-1}y_0)$ is ridgeless kernel regression with $\Theta^{(L)}_\infty$ plus a mean-zero Gaussian fluctuation pinned to zero on the training points: a wide network trained to convergence *is* a kernel machine, and the generalization mystery dissolves into the spectrum of one fixed, explicit kernel.

```python
import numpy as np
import torch


def relu_dual(cov_xx, cov_xpxp, cov_xxp):
    """Return E[ReLU(X)ReLU(X')] and E[ReLU'(X)ReLU'(X')]."""
    denom = np.sqrt(cov_xx * cov_xpxp)
    rho = np.clip(cov_xxp / np.maximum(denom, 1e-12), -1.0, 1.0)
    angle = np.arccos(rho)
    nngp = denom / (2.0 * np.pi) * (
        np.sin(angle) + (np.pi - angle) * np.cos(angle)
    )
    nngp_dot = (np.pi - angle) / (2.0 * np.pi)
    return nngp, nngp_dot


def infinite_ntk(X, Xp, depth, beta=0.1):
    """Compute (Theta_inf^(L), Sigma^(L)) for a depth-L ReLU MLP."""
    if depth < 1:
        raise ValueError("depth counts affine layers and must be at least 1")
    n0 = X.shape[1]
    beta2 = beta ** 2
    sig = X @ Xp.T / n0 + beta2
    sig_xx = (X * X).sum(axis=1) / n0 + beta2
    sig_pp = (Xp * Xp).sum(axis=1) / n0 + beta2
    theta = sig.copy()

    for _ in range(depth - 1):
        nngp, nngp_dot = relu_dual(sig_xx[:, None], sig_pp[None, :], sig)
        sig_xx = relu_dual(sig_xx, sig_xx, sig_xx)[0] + beta2
        sig_pp = relu_dual(sig_pp, sig_pp, sig_pp)[0] + beta2
        sig = nngp + beta2
        theta = theta * nngp_dot + sig
    return theta, sig


class WideMLP(torch.nn.Module):
    """Finite-width MLP in the NTK parameterization."""
    def __init__(self, n0, width, depth, beta=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth counts affine layers and must be at least 1")
        self.beta = beta
        sizes = [n0] + [width] * (depth - 1) + [1]
        self.Ws = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim, in_dim))
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
        )
        self.bs = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim)) for out_dim in sizes[1:]
        )
        self.scales = [in_dim ** -0.5 for in_dim in sizes[:-1]]

    def forward(self, x):
        a = x
        last = len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = self.scales[i] * (a @ W.T) + self.beta * b
            if i != last:
                a = torch.relu(a)
        return a.squeeze(-1)


def _jacobian_rows(net, X):
    params = list(net.parameters())
    rows = []
    for xi in X:
        out = net(xi.unsqueeze(0)).squeeze()
        grads = torch.autograd.grad(out, params)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)


def empirical_ntk(net, X, Xp=None):
    """Finite NTK: J_theta f(X) J_theta f(Xp)^T."""
    J = _jacobian_rows(net, X)
    Jp = J if Xp is None else _jacobian_rows(net, Xp)
    return (J @ Jp.T).detach().cpu().numpy()


def kernel_regression(K_train, K_test, y, ridge=0.0):
    """Ridgeless kernel-regression mean; use ridge > 0 for conditioning."""
    A = K_train + ridge * np.eye(K_train.shape[0])
    return K_test @ np.linalg.solve(A, y)
```
