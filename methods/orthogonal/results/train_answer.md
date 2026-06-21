Every time I make a feedforward network deeper, the same problem returns. A forward signal is multiplied by a weight matrix at each layer and a backward error is multiplied by transposed weight matrices on the way back, so if the typical per-layer scale is even slightly below one, depth drives it to nearly zero, and if it is slightly above one, depth blows it up. The standard variance-propagation analysis already names this: a layer with fan-in $n$ and i.i.d. weights picks up a factor about $n\,\mathrm{Var}[W]$ per layer, and across depth $d$ a gradient variance carries a product like $(n\,\mathrm{Var}[W])^{d-i}$. Glorot-style scaling, $\mathrm{Var}[W]=2/(n_{\rm in}+n_{\rm out})$, was useful because it tries to hold that per-layer average near one. But average-scale preservation is the wrong target. A matrix can preserve the expected squared norm of a typical random vector while stretching some directions and crushing others: for an $N\times N$ Gaussian with entries of variance $1/N$, $\mathbb{E}[v^TW^TWv]=v^Tv$ holds exactly, yet the squared singular values have a Marchenko–Pastur spread. In a product of many such matrices, it is the full singular spectrum, not the first moment, that decides which error directions still reach early layers. Greedy unsupervised pretraining was the strongest practical answer at the time, but it is not a pure initialization rule: it needs the data and an extra training phase. What I want is a data-independent rule, run once at construction from parameter shapes alone, that keeps both forward activations and backward gradients alive through arbitrarily many layers, so the iteration count to learn is not governed by an exponential depth penalty.

To find the right target I strip the nonlinearity and keep the depth, because a stack of linear layers is still a single linear map yet its learning dynamics in the weights remain nonlinear: the loss depends on a product of matrices and each matrix's gradient contains the product of all the others. For the three-layer map $y=W^{32}W^{21}x$, continuous-time batch gradient descent on squared error, with $\tau=1/\lambda$, the input correlation $\Sigma^{11}=\sum_\mu x^\mu x^{\mu T}$, and the input–output correlation $\Sigma^{31}=\sum_\mu y^\mu x^{\mu T}$, gives
$$\tau\,\frac{dW^{21}}{dt}=W^{32T}(\Sigma^{31}-W^{32}W^{21}\Sigma^{11}),\qquad \tau\,\frac{dW^{32}}{dt}=(\Sigma^{31}-W^{32}W^{21}\Sigma^{11})W^{21T}.$$
Whitening the inputs, $\Sigma^{11}=I$, leaves only $\Sigma^{31}=USV^T$, and the change of variables $W^{21}=\overline W^{21}V^T$, $W^{32}=U\overline W^{32}$ cancels the $U$ and $V$ factors so the modes decouple. Putting each active mode along its own orthonormal hidden direction kills the competitive cross-terms, and a single mode of strength $s$ reduces to two scalars,
$$\tau\,\frac{da}{dt}=b(s-ab),\qquad \tau\,\frac{db}{dt}=a(s-ab).$$
The quantity $a^2-b^2$ is conserved, so each trajectory rides a hyperbola toward the fixed manifold $ab=s$; the origin is also a fixed point but unstable, which is exactly the plateau small random weights sit on. In the symmetric case $u=ab=a^2$ this becomes a logistic equation $\tau\,\dot u=2u(s-u)$ with solution $u(t)=s\,e^{2st/\tau}/(e^{2st/\tau}-1+s/u_0)$, whose sigmoid shape reproduces the plateau-then-jump without any extra mechanism, and whose learning time is $t\approx(\tau/s)\log(s/\epsilon)=O(\tau/s)$: stronger modes learn faster, and the cutoff enters only logarithmically. Pushing to $N_l$ layers, the composite strength is the product $u=\prod_i a_i$, the symmetric dynamics become $\tau\,\dot u=(N_l-1)\,u^{2-2/(N_l-1)}(s-u)$, and computing the top Hessian eigenvalue on the symmetric path at the optimum $a_{\rm opt}=s^{1/(N_l-1)}$ gives $\lambda_1(a_{\rm opt})=(N_l-1)s^{(2N_l-4)/(N_l-1)}/\tau$, so a stable first-order step must shrink as $\alpha_{\rm opt}\sim O(1/(N_l s^2))$. Folding that depth-dependent rate back in, the infinite-depth delay over the three-layer case stays finite — but only under one condition, which the analysis makes precise: the starting composite strength $u_0$ must be $O(1)$. If each layer starts with scalar strength $a_0<1$, then $u_0=a_0^{N_l-1}$ vanishes exponentially with depth and the finite-delay story collapses; if $a_0>1$ it explodes. So the real requirement is a data-free way to make the end-to-end product preserve mode strength through any number of layers.

The method I propose is orthogonal initialization for dynamical isometry: initialize every convolutional and linear weight tensor as a random semi-orthogonal matrix, scaled by a gain set by the nonlinearity. Scaled Gaussian weights are exactly the data-free candidate that fails, and seeing why fixes the target. They satisfy the average condition $\mathbb{E}[v^TW^TWv]=v^Tv$, but multiplying many of them yields a non-normal product with highly non-orthogonal eigenvectors, eigenvalues concentrating near the origin while singular values spread, so most singular values shrink toward zero and a small tail grows large. In backpropagation the transpose of that product shares those singular values, and error components in the small-singular-value subspace are crushed before they reach early layers — vanishing gradients even though the first-moment calculation looked fine. The condition I actually need is therefore a full-spectrum one: the product of layer Jacobians should act like a near-isometry, with as many singular values as possible in a narrow band around an $O(1)$ constant. The matrix that achieves this exactly is an orthogonal one, $Q^TQ=I$, with every singular value equal to one, and crucially the property survives composition, since $(Q_2Q_1)^T(Q_2Q_1)=Q_1^TQ_2^TQ_2Q_1=I$. So initializing each layer isometric makes the end-to-end map isometric before training begins, holding $u_0$ at $O(1)$ no matter the depth. Rectangular tensors are not an obstacle: a matrix with fewer rows than columns can have orthonormal rows, one with more rows than columns can have orthonormal columns, and in either case all nonzero singular values are one, the best available condition; for a convolution kernel I flatten the trailing dimensions and view the tensor as $\texttt{out\_channels}\times(\texttt{in\_channels}\cdot k_H\cdot k_W)$.

Two design choices make the construction correct. First, the random draw has to be Haar-uniform, not biased by the factorization. Drawing a Gaussian $Z$ and taking $Z=QR$ gives an orthonormal $Q$, but the decomposition is ambiguous: $(Q\Lambda)(\Lambda^{-1}R)$ is equally valid for any diagonal sign matrix $\Lambda$, and a numerical QR routine resolves the signs by a convention that skews the distribution. I remove the bias by forcing the diagonal of $R$ positive, setting $d=\operatorname{sign}(\operatorname{diag}(R))$ and $Q\leftarrow Q\,d$ — in code, `d = torch.diag(r, 0); ph = d.sign(); q *= ph` — and when $\texttt{rows}<\texttt{cols}$ I transpose before QR and back after, so the returned matrix is semi-orthogonal in the requested orientation. Second, the gain is what places the nonlinear network in the propagating regime. With the linear part isometric, the population variance $q^l=\frac1N\sum_i(x_i^l)^2$ obeys, under a Gaussian-activity approximation, the one-dimensional recursion $q^{l+1}=g^2\int Dz\,\phi(\sqrt{q^l}z)^2$. For $\tanh$ the slope at the origin is $g^2$, so the zero fixed point loses stability at the critical gain $g_c=1$, the edge of chaos where linear gain balances damping. For ReLU, $\int Dz\,\max(0,\sqrt q z)^2=q/2$, so the variance-preserving condition is $g^2/2=1$, hence $g=\sqrt2$; a purely linear layer uses gain $1$. One distinction I keep clean: the mathematical $\tanh$ edge here is $g_c=1$, whereas PyTorch's `calculate_gain("tanh")` returns $5/3$ as its own variance heuristic — a different quantity. Since the hook targets ReLU-style convolutional networks, the relevant library gain is `calculate_gain("relu") = sqrt(2)`. Rather than reimplement QR inside the model hook I call the tested primitive, which already performs the flatten, the $\texttt{rows}<\texttt{cols}$ transpose, the QR, the Haar sign correction, the copy, and the gain multiplication; the hook then leaves the bias-free convolutions alone, zeroes the linear bias, and keeps batch-normalization affine parameters at the identity ($\gamma=1$, $\beta=0$).

The core of `nn.init.orthogonal_` implements the Haar-corrected QR recipe the method relies on:

```python
def orthogonal_core(tensor, gain=1, generator=None):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if tensor.numel() == 0:
        return tensor

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1, generator=generator)
    if rows < cols:
        flattened.t_()

    q, r = torch.linalg.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor
```

and the initialization hook that applies it across the model is:

```python
def initialize_weights(model, config):
    """Orthogonal initialization.

    Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
    BatchNorm2d: weight=1, bias=0.
    """
    gain = nn.init.calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```
