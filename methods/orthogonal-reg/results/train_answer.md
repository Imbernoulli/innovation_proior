When I train a deep convolutional net, a signal entering at the bottom is multiplied by one layer's weights, then a nonlinearity, then the next layer's weights, all the way up to the logits, and a backpropagated error does the same thing in reverse through the transposed weight matrices. So both the forward activation and the backward gradient are, at heart, a vector being hit by a long product of matrices, and a product of matrices does to a vector what a product of numbers does to a scalar: if the typical multiplicative factor is below one, the vector shrinks geometrically with depth and the early layers see vanishing gradients; if it is above one, it grows geometrically and the gradient explodes. Pascanu, Mikolov and Bengio made this precise for recurrent nets, where unrolling a tied weight matrix in time gives the gradient of a late state with respect to an early one as a product of Jacobians, $\frac{\partial x_t}{\partial x_k} = \prod_{t \ge i > k} W_{\mathrm{rec}}^\top \operatorname{diag}(\sigma'(x_{i-1}))$. Taking the 2-norm and using that the norm of a product is at most the product of norms gives $\left\lVert \frac{\partial x_t}{\partial x_k} \right\rVert \le \prod \lVert W_{\mathrm{rec}}^\top \rVert \, \lVert \operatorname{diag}(\sigma') \rVert$, with $\lVert \operatorname{diag}(\sigma') \rVert \le \gamma = \sup |\sigma'|$ (one for $\tanh$, $1/4$ for the logistic sigmoid), so $\lVert W_{\mathrm{rec}} \rVert_2 < 1/\gamma$ is a sufficient condition for long-range gradients to vanish geometrically. The mirror statement about explosion is only necessary, not sufficient — the dynamics also have to align along an expansive direction — but the diagnosis is the same either way: the singular values and operator norms of the weight matrices are the levers that decide whether deep signal propagation is stable.

What makes a square matrix's singular values all exactly one is that it is exactly norm-preserving, $\lVert W x \rVert = \lVert x \rVert$ for every $x$, which is precisely the orthogonal matrices, characterizable either as $W^\top W = I$ (orthonormal columns) or $W W^\top = I$ (orthonormal rows). Saxe, McClelland and Ganguli showed why this is the configuration to want at depth: scaling random Gaussian weights to preserve norm on average — drawing entries i.i.d. with standard deviation $1/\sqrt{N}$ so that $\langle v^\top W^\top W v\rangle = v^\top v$ — does not give a flat spectrum, because the squared singular values follow the Marchenko-Pastur law with a real spread that survives $N \to \infty$, and a product of such matrices develops a wildly kurtotic spectrum that crushes most singular directions to zero and blows up a thin tail. Such a product preserves the norm of a typical vector but anisotropically, annihilating any backpropagated error component lying in the crushed subspace. A product of orthogonal matrices is itself orthogonal, so every singular value stays exactly one no matter the depth — dynamical isometry — and initializing each layer to a random orthogonal matrix (the $Q$ of a QR decomposition) gives depth-independent learning time where scaled Gaussian init keeps slowing down. The trouble is that this is a condition at step zero and nothing holds it there: the instant the data gradient takes its first step it pulls $W$ off the orthogonal manifold in whatever direction reduces the task loss, with no regard for the spectrum, and by mid-training — exactly the long stretch where most of the weight movement happens — the flat-spectrum property is gone. A good initial condition does not pin the weights; I need a force that keeps acting throughout training. The heavy-handed answer, a hard Stiefel constraint that confines $W$ to $\{W : W W^\top = I\}$ and re-orthogonalizes after every step with a fresh QR or SVD, holds the spectrum exactly but costs a matrix factorization of every weight matrix on every iteration, which grows badly with size and is a non-starter in the inner loop. And it sits awkwardly on convolutional weights, which are not square; ordinary L2 weight decay, the other adjacent tool, pulls every singular value down toward zero, which is the opposite of holding the nonzero spectrum at one.

I propose Orthogonal Regularization: instead of constraining $W$ to the manifold, add a soft, differentiable penalty to the loss that grows as $W$ drifts away from orthogonal, hand it to the optimizer already running, and let the data loss and the penalty negotiate. The first thing to settle is which orthogonality condition to enforce, because a conv weight is rectangular. A conv weight tensor of shape $[\,\text{out\_channels}, \text{in\_channels}, k, k\,]$ flattens naturally to a matrix $W \in \mathbb{R}^{m \times n}$ with $m = \text{out\_channels}$ rows and $n = \text{in\_channels}\cdot k\cdot k$ columns, almost never square. The two Gram conditions stop being equivalent here: $W^\top W = I_n$ asks for $n$ orthonormal columns living in $\mathbb{R}^m$, impossible when $n > m$, while $W W^\top = I_m$ asks for the $m$ rows — which are exactly the $m$ filters, one per output channel — to be orthonormal in $\mathbb{R}^n$, feasible when $n \ge m$, the usual fat-conv-weight case. So the rule is to penalize the Gram that lives on the smaller dimension, the one whose identity is attainable: penalizing $W^\top W - I$ on a fat $W$ would minimize toward a floor it can never reach, since $W^\top W$ has rank at most $m < n$ and can never equal the full-rank $n \times n$ identity, a biased objective. For the standard wide conv weight that means $W W^\top$ over the output channels, which doubles as a clean goal in its own right — no two filters point in the same direction, the filters are decorrelated, and each has unit norm.

The deviation from orthogonality is just the residual of the attainable Gram from the identity,
$$R = W W^\top - I.$$
This is the right quantity to measure because it encodes both things I care about at once in the natural inner-product geometry: the off-diagonal entry $R_{ij} = \langle \text{filter}_i, \text{filter}_j\rangle$ is exactly the pairwise filter correlation I want driven to zero, and the diagonal entry $R_{ii} = \lVert \text{filter}_i\rVert^2 - 1$ is exactly the magnitude collapse I want prevented. $R$ is zero precisely when the selected filters are orthonormal, which is the feasible version of the flat-spectrum property the propagation analysis pointed at; the obvious alternatives all miss — penalizing $\lVert W \rVert$ is plain weight decay pulling the whole spectrum to zero, and a determinant-based penalty is non-convex and degenerate for rectangular $W$. The leanest way to measure the size of $R$ is the entrywise absolute residual,
$$L_{\mathrm{ortho}} = \sum_{i,j} |R_{ij}| = \sum_{i,j}\bigl|(W W^\top - I)_{ij}\bigr|,$$
which says exactly what I mean — every dot-product error and every unit-norm error contributes by its magnitude — and is non-differentiable only at zero, where a perfectly usable subgradient exists, with no factorization needed. For a smooth variant with a clean closed-form gradient I square the Frobenius norm instead,
$$L_{\mathrm{smooth}} = \lVert W W^\top - I\rVert_F^2 = \sum_{i,j} R_{ij}^2 = \operatorname{tr}(R^\top R),$$
where squaring is deliberate: it removes the root that makes the bare Frobenius norm non-differentiable at zero, makes the penalty a smooth quadratic in the residual, and grows fast on large residual entries so it pushes hardest on the worst-off filters.

The reason the squared form is cheap is worth doing carefully, since I am handing the gradient to the optimizer and want no hidden factorization. Working entrywise with $L = \sum_{i,j}(\sum_a W_{ia}W_{ja} - \delta_{ij})^2$, the derivative is $\frac{\partial L}{\partial W_{pq}} = 2\sum_{i,j} R_{ij}(\delta_{ip}W_{jq} + \delta_{jp}W_{iq}) = 2\sum_j R_{pj}W_{jq} + 2\sum_i R_{ip}W_{iq}$, and since $R = W W^\top - I$ is symmetric both sums are $2(RW)_{pq}$, giving
$$\nabla_W L = 4\,R\,W = 4\,(W W^\top - I)\,W.$$
No SVD, no eigendecomposition — two matrix multiplies, $R = W W^\top - I$ and then $RW$, both cheap relative to the convolution itself, and autodiff computes exactly this when I write the forward penalty. The column-Gram mirror is $\nabla_W \lVert W^\top W - I\rVert_F^2 = 4\,W\,(W^\top W - I)$, the same shape with the residual on the other side. The absolute form replaces $R$ in the same chain with the subgradient $S = \operatorname{sign}(R)$, giving $2\,\operatorname{sign}(R)\,W$ away from zero. The wall that killed Stiefel optimization — a factorization every step — is gone, traded for a cheap gradient nudge toward the manifold, which is all I need to keep the spectrum from drifting.

This is a soft encouragement, not a law, so it carries a small coefficient $\lambda$ added to the task loss, $L_{\mathrm{total}} = L_{\mathrm{CE}} + \lambda \sum_W \sum_{i,j}|(W W^\top - I)_{ij}|$ or, for the smooth variant, $L_{\mathrm{total}} = L_{\mathrm{CE}} + \lambda \sum_W \lVert W W^\top - I\rVert_F^2$, where the sum runs over every conv filter bank, each layer penalized independently since each layer's own propagation matters. $\lambda$ must be small — the data loss is what I care about, and a perfectly orthogonal but task-blind filter bank is useless, so something on the order of a fraction of a percent of the loss scale, swept over a few orders of magnitude. It coexists with the optimizer's built-in L2 weight decay rather than replacing it: weight decay shrinks overall magnitude, this flattens the feasible nonzero singular values toward one, and the two pull on different things. The diagonal of $R$ already anchors $\lVert \text{filter}\rVert^2$ near one, so the penalty has a mild scale-anchoring effect of its own, complementary to weight decay. And there is a generalization story beyond gradient flow: the off-diagonal part decorrelates filters so the layer is not wasting capacity on near-duplicates, and the diagonal part keeps filter magnitudes from collapsing toward an uninformative origin, so more of the model's nominal capacity stays usable and the representation stays well-conditioned — all at essentially no per-step cost and with no change to the architecture, base loss, optimizer, or schedule.

To fill the training-loop slot, I iterate the conv filter banks, reshape each weight tensor to $[\text{out\_channels}, \text{fan\_in}]$ so rows are filters, build the chosen Gram, subtract an identity with matching device and dtype, and accumulate either the absolute residual or its square, skipping biases, BatchNorm scales, and other one-dimensional parameters because the orthogonality claim is about linear filter banks.

```python
import torch
import torch.nn as nn


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _orthogonal_residual(W, side="rows"):
    if side == "auto":
        side = "rows" if W.size(0) <= W.size(1) else "cols"

    if side == "rows":
        gram = W @ W.t()          # W W^T: row/filter Gram
    elif side == "cols":
        gram = W.t() @ W          # W^T W: column Gram
    else:
        raise ValueError("side must be 'rows', 'cols', or 'auto'")

    eye = torch.eye(gram.size(0), device=W.device, dtype=W.dtype)
    return gram - eye


def compute_regularization(model, inputs, outputs, targets, config=None):
    """Soft orthogonal regularization for convolutional filter banks.

    Defaults to the abs row-Gram form: lambda * sum |W W^T - I|.
    Set config["orthogonal_reg_norm"] to "fro2" for the squared-Frobenius variant.
    Set config["orthogonal_reg_side"] to "auto" to choose the feasible smaller Gram.
    """
    config = config or {}
    lam = config.get("orthogonal_reg_lambda", 1e-6)
    norm = config.get("orthogonal_reg_norm", "abs")
    side = config.get("orthogonal_reg_side", "rows")

    reg = outputs.new_zeros(())
    for module in model.modules():
        if isinstance(module, _CONV_TYPES):
            W = module.weight.reshape(module.weight.size(0), -1)
            R = _orthogonal_residual(W, side=side)
            if norm == "abs":
                reg = reg + R.abs().sum()
            elif norm in {"fro2", "squared", "l2"}:
                reg = reg + (R * R).sum()
            else:
                raise ValueError("orthogonal_reg_norm must be 'abs', 'fro2', 'squared', or 'l2'")

    return lam * reg
```

Use `orthogonal_reg_lambda` as a small coefficient, commonly around $10^{-6}$ for the abs form or $10^{-4}$ for the squared variant, and tune it with the task loss scale.
