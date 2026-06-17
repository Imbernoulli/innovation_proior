# Adversarial Weight Perturbation (AWP)

## Problem

Adversarial training flattens the *input* loss landscape but leaves a large **robust
generalization gap**: training robustness keeps rising while test robustness peaks early and
then degrades (robust overfitting). Early stopping narrows the gap only by halting at low
training robustness. AWP is designed to narrow the gap directly while keeping training
robustness high, and it plugs on top of any adversarial-training loss.

## Key idea

The flatness of the **weight loss landscape** (how the adversarial loss `rho(w)` changes as
the weights move) is strongly correlated with the robust generalization gap: flatter landscapes
track smaller gaps. Successful methods flatten it only implicitly. AWP flattens it *explicitly* by training
the network at an adversarially perturbed set of weights, forming a **double perturbation** —
inputs and weights are both perturbed toward the worst case:

```
min_w  max_{v in V}  rho(w + v),
        rho(w + v) = (1/n) sum_i max_{||x'_i - x_i|| <= eps}  ell( f_{w+v}(x'_i), y_i ).
```

Minimizing `rho(w) + (rho(w+v) - rho(w))` collapses to `min_w rho(w+v)`: optimizing at the
perturbed point both lowers the loss and flattens the surface. A *worst-case* `v` is used
(not random), which is far more efficient and gives a conservative upper bound on the
expected sharpness term in the PAC-Bayes flatness bound.

## Weight perturbation: direction and size

- **Direction:** projected gradient *ascent* on `v` of the (batch-averaged) loss — the
  worst-case batch direction, by analogy with the PGD input attack.
- **Size:** layer-wise *relative* constraint `||v_l|| <= gamma * ||w_l||` (not a fixed value),
  because per-layer weight magnitudes differ and ReLU networks are scale-invariant. `gamma`
  is one small dimensionless knob (default `5e-3`, useful range `[1e-3, 5e-3]`).
- **Projection** `Pi_gamma`: per layer, if `||v_l|| > gamma*||w_l||` then
  `v_l <- gamma*(||w_l||/||v_l||)*v_l`, else `v_l`.

## Algorithm (AWP on the KL-regularized base loss; defaults A=1, K1=10, K2=1)

```
for each minibatch (x, y):
    # 1) inner input attack: maximize KL(f_w(x) || f_w(x')) for K1 PGD steps
    x' = PGD_KL(w, x; eps, eta1, K1)

    # 2) worst-case weight perturbation v (K2 ascent steps, on a proxy copy, after warmup):
    proxy <- w
    L_v = CE(f_proxy(x), y) + beta * KL(f_proxy(x) || f_proxy(x'))   # base loss
        # in PyTorch: kl_div(log_softmax(proxy(x')), softmax(proxy(x))) computes KL(clean || adv)
    one SGD step on proxy minimizing (-L_v)        # == ascent on L_v
    for each weight tensor l (dim > 1, 'weight'):  # skip BN scale/bias
        dw_l    = w_proxy_l - w_l                   # ascent direction (∝ +grad L_v)
        diff_l  = (||w_l|| / (||dw_l|| + 1e-20)) * dw_l     # so ||diff_l|| = ||w_l||
    w <- w + gamma * diff                           # apply v;  ||v_l|| = gamma*||w_l||

    # 3) base SGD step AT THE PERTURBED WEIGHTS w+v
    L = CE(f_w(x), y) + beta * KL(f_w(x) || f_w(x'))
    SGD step: minimize L

    # 4) come back to center: undo the temporary perturbation after the optimizer step
    w <- w - gamma * diff
```

A proxy copy is used for step 2 so the throwaway forward passes do not pollute the real
model's batch-norm running statistics. The proxy learning rate does not set the perturbation
radius: only the *direction* of its step is used, and the magnitude is set by `gamma` and the
renormalization.
For vanilla PGD-AT the base loss is just `CE(f_w(x'), y)` and `x'` is crafted by maximizing
that CE; AWP is otherwise unchanged.

## Theoretical justification

The PAC-Bayes flatness bound (Neyshabur et al. 2017) rewrites the randomized-predictor error as
empirical loss plus **expected sharpness**
`E_u[Lhat(f_{w+u})] - Lhat(f_w)` plus the KL complexity of the posterior perturbation
distribution `Q` relative to a data-independent prior `P`. With `P = N(0, sigma^2 I)` and the
same spherical variance around `w`, the KL part is `||w||^2/(2*sigma^2)`. If
`sigma^2 = a*||w||^2`, that contribution becomes `1/(2a)`, so the scale-fixed bound depends on
expected sharpness. For perturbations supported on the same feasible set,
`E_u[rho(w+u)] - rho(w) <= max_u rho(w+u) - rho(w)`, so minimizing worst-case sharpness reduces
a sufficient upper bound on the expected sharpness term that controls the robust generalization
gap.

## Implementation

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict

EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (_, new_w) in zip(model_state_dict.items(),
                                          proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        # PyTorch computes KL(target || input_distribution), so this is KL(clean || adv).
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = -1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        return diff_in_weights(self.model, self.proxy)

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


def perturb_input_trades(model, images, eps, step_size, perturb_steps):
    model.eval()
    adv_images = torch.clamp(images.detach() + 0.001 * torch.randn_like(images), 0.0, 1.0)
    for _ in range(perturb_steps):
        adv_images.requires_grad_()
        loss_kl = F.kl_div(F.log_softmax(model(adv_images), dim=1),
                           F.softmax(model(images), dim=1),
                           reduction='sum')  # KL(clean || adv), maximized over adv
        grad = torch.autograd.grad(loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + step_size * torch.sign(grad.detach())
        adv_images = torch.min(torch.max(adv_images, images - eps), images + eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()


def train_step(model, images, labels, optimizer, awp_adversary,
               eps, step_size, perturb_steps, beta=6.0,
               epoch=0, awp_warmup=0):
    x_adv = perturb_input_trades(model, images, eps, step_size, perturb_steps)

    model.train()
    awp = None
    if epoch >= awp_warmup:
        awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                     inputs_clean=images,
                                     targets=labels,
                                     beta=beta)
        awp_adversary.perturb(awp)

    optimizer.zero_grad()
    logits_adv = model(x_adv)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                           F.softmax(model(images), dim=1),
                           reduction='batchmean')  # KL(clean || adv)
    logits = model(images)
    loss_natural = F.cross_entropy(logits, labels)
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    if awp is not None:
        awp_adversary.restore(awp)
    return {'loss': loss.item()}
```
