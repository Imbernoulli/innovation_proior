The bilevel problem couples an upper objective f(x, y) to a lower one through an argmin constraint y in S(x) := argmin_u g(x, u). The difficulty is that the feasible set S(x) drifts as the upper variable x moves, so admissibility of y is redefined by a fresh optimization at every step. Existing routes each fail on part of the requirement. The implicit-gradient method assumes g(x, .) is strongly convex, so S(x) is a single point and the lower Hessian is invertible; once the lower problem is merely Polyak-Lojasiewicz or non-convex, the inverse Hessian does not exist and there is no implicit gradient to compute. Reverse-mode hypergradient methods avoid the Hessian inverse by unrolling T steps of inner gradient descent and backpropagating through the trajectory, but that forces memory or recomputation to grow with the unroll length T, and constrained inner loops are awkward to differentiate. A tempting shortcut is to replace the argmin constraint with a penalty on a lower-level optimality metric and run joint gradient descent on (x, y), but a naive penalty can converge to points that are not bilevel solutions. The diagnostic instance min sin^2(y - 2pi/3) subject to y in argmin (y^2 + 2 sin^2 y) has only y = 0 as the true solution, yet for every penalty constant gamma the penalized objective f + gamma (y + sin 2y)^2 has a spurious stationary point at y = 2pi/3 where the lower Hessian degenerates. So the question is not merely how to add a penalty, but which penalty, with what geometry, and under what finite gamma, yields a true bilevel solution.

The method I propose is G-PBGD, gradient-norm penalty-based bilevel gradient descent. It folds the lower-level optimality requirement into a single joint objective over (x, y), then runs one projected gradient step on that objective. The canonical formulation is F_gamma(x, y) = f(x, y) + (gamma / 2) ||grad_y g(x, y)||^2. The 1/2 is a normalization so that differentiating the penalty gives exactly gamma times the Hessian-vector product: the y-gradient is gamma * grad_yy g . grad_y g and the x-gradient is gamma * grad_xy g . grad_y g. No Hessian inverse is needed, no inner optimizer is unrolled, and no value-function estimate is required. The gradient-norm penalty is computable with one extra backward pass: compute grad_y g with a retained autodiff graph, form the scalar (gamma / 2) ||grad_y g||^2, and call backward once. The framework hands back the exact penalty gradient as Hessian-vector products.

Why is this principled? The lower-level constraint y in S(x) is the same as requiring the squared distance d^2_{S(x)}(y) to be zero. A penalty p is called a rho-squared-distance bound if p is nonnegative, dominates d^2 up to a constant rho, and vanishes exactly on S(x). Under such a bound, minimizing f + gamma p approximates the bilevel problem: a residual p <= epsilon yields an objective gap at most O(sqrt(epsilon)). For global solutions the residual is O(1/gamma); for local solutions it improves to O(1/gamma^2) provided p is well behaved toward S(x). For the gradient-norm penalty, the Polyak-Lojasiewicz inequality provides the squared-distance bound without strong convexity; specifically, under a (1/sqrt(mu))-PL condition on g(x, .), p = ||grad_y g||^2 is a mu-SDB. The needed penalty constant is finite and tight: gamma = Theta(delta^{-1/2}) suffices to reach lower-level accuracy delta, as witnessed by the simple instance min y subject to y in argmin y^2, whose penalized solution is y = -1/(2 gamma). Thus G-PBGD is a first-order, single-loop, memory-cheap method for non-strongly-convex bilevel problems, with the caveat that the local guarantee requires the lower Hessian to stay nonsingular away from solutions. At points where grad_yy g degenerates while grad_y g is nonzero, the gradient-norm penalty can stall; the value-gap sibling is the more robust fallback, but G-PBGD is the simplest computational variant.

```python
import torch
import torch.nn.functional as F


def loss_F(tensors):
    """Squared L2 norm summed over a list of tensors."""
    return sum(torch.linalg.norm(w) ** 2 for w in tensors)


def g_pbgd_step(x, net, x_opt, y_opt, tr, val, gam, reg):
    """One joint G-PBGD update on (x = cleaner logits, y = network parameters)."""
    x_opt.zero_grad()
    y_opt.zero_grad()

    # Upper objective: clean validation loss.
    fy = F.cross_entropy(net(val.data), val.clean_target)

    # Lower objective: importance-weighted corrupted training loss, plus optional ridge.
    ce_tr = F.cross_entropy(net(tr.data), tr.dirty_target, reduction="none")
    gxy = (torch.sigmoid(x) * ce_tr).mean() + reg * loss_F(net.parameters())

    # Lower-level gradient with retained graph so a second backward gives the HVP.
    dgdy = torch.autograd.grad(gxy, net.parameters(), create_graph=True)

    # Once gamma exceeds 1, rescale the whole step by 1/gamma so the penalty term
    # contributes an O(1) update rather than an O(gamma) one.
    lr_decay = min(1.0 / (gam + 1e-8), 1.0)
    loss = lr_decay * (fy + gam / 2.0 * loss_F(dgdy))

    loss.backward()
    x_opt.step()
    y_opt.step()


def train(x, net, hparams, tr, val):
    y_opt = torch.optim.SGD(net.parameters(), lr=hparams["lry"])
    x_opt = torch.optim.SGD([x], lr=hparams["lrx"])
    gam = hparams["gamma_init"]
    step_gam = (hparams["gamma_max"] - hparams["gamma_init"]) / hparams["gamma_argmax_step"]
    for _ in range(hparams["outer_itr"]):
        g_pbgd_step(x, net, x_opt, y_opt, tr, val, gam, hparams["reg"])
        gam = min(hparams["gamma_max"], gam + step_gam)
    return x, net
```
