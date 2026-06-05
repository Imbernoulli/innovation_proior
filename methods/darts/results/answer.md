# DARTS: Differentiable Architecture Search

## Problem

Make neural architecture search orders of magnitude cheaper than RL/evolution over a discrete space. The blocker is that a discrete architecture choice has no gradient w.r.t. validation performance, forcing the expensive sample-train-reward loop. DARTS makes the architecture **continuous and differentiable**, so it can be learned by gradient descent jointly with the weights.

## Key idea

A cell is a DAG of N nodes; node j sums transformed predecessors: x^(j) = Σ_{i<j} o^(i,j)(x^(i)). Relax the categorical choice of operation on each edge into a **softmax mixture** over candidate ops, parameterized by continuous logits α^(i,j):

```
ō^(i,j)(x) = Σ_{o∈O}  exp(α_o^(i,j)) / Σ_{o'∈O} exp(α_{o'}^(i,j))  ·  o(x).
```

The architecture is α = {α^(i,j)}. Optimize it for generalization via a **bilevel** problem:

```
min_α   L_val( w*(α), α ),    s.t.  w*(α) = argmin_w L_train(w, α).
```

## Approximate architecture gradient

Approximate w*(α) by a single inner step w' = w − ξ ∇_w L_train(w, α) and differentiate through it. By the chain rule (using ∂w'/∂α = −ξ ∇²_{α,w} L_train):

```
∇_α L_val(w', α) = ∇_α L_val(w', α) − ξ ∇²_{α,w} L_train(w, α) · ∇_{w'} L_val(w', α).
```

The Hessian-vector product is computed by central finite difference, with v = ∇_{w'}L_val(w',α), w± = w ± ε v (ε = 0.01/‖v‖₂):

```
∇²_{α,w} L_train(w, α) · v ≈ [ ∇_α L_train(w⁺, α) − ∇_α L_train(w⁻, α) ] / (2ε),
```

reducing cost from O(|α||w|) to O(|α|+|w|). Setting **ξ = 0** drops the second term, giving the cheaper **first-order** approximation ∇_α L_val(w, α) (worse empirically); **ξ > 0** is the **second-order** version.

## Algorithm

```
Create a mixed operation ō^(i,j) parameterized by α^(i,j) for each edge (i,j)
while not converged:
    1. Update α by descending ∇_α L_val( w − ξ ∇_w L_train(w, α), α )   (ξ=0 ⇒ first-order)
    2. Update w by descending ∇_w L_train(w, α)
Derive the discrete cell from the learned α.
```

**Discretization.** For each node keep the top-k strongest incoming ops from distinct predecessors, strength = softmax weight exp(α_o^(i,j))/Σ_{o'}exp(α_{o'}^(i,j)); k=2 for conv cells, k=1 for recurrent. The *zero* op is excluded (need exactly k non-zero edges; its strength is underdetermined because batch-norm makes scaling the zero logit irrelevant to the output).

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5',
       'max_pool_3x3','avg_pool_3x3','skip_connect','none']

class MixedEdge(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([make_op(n, C) for n in OPS])
    def forward(self, x, alpha_edge):
        w = F.softmax(alpha_edge, dim=-1)
        return sum(wi * op(x) for wi, op in zip(w, self.ops))

def hessian_vector(model, alpha, batch, v, eps):
    R = eps / (v.norm() + 1e-12)
    for p, vi in zip(model.weights(), v): p.data.add_(R, vi)
    g_plus  = torch.autograd.grad(L_train(model, batch, alpha), alpha)
    for p, vi in zip(model.weights(), v): p.data.sub_(2*R, vi)
    g_minus = torch.autograd.grad(L_train(model, batch, alpha), alpha)
    for p, vi in zip(model.weights(), v): p.data.add_(R, vi)
    return [(gp - gm) / (2*R) for gp, gm in zip(g_plus, g_minus)]

def arch_step(model, alpha, tb, vb, xi, eps, opt_a):
    opt_a.zero_grad()
    if xi == 0:
        L_val(model, vb, alpha).backward()
    else:
        w0 = [p.detach().clone() for p in model.weights()]
        gw = torch.autograd.grad(L_train(model, tb, alpha), model.weights())
        for p, g in zip(model.weights(), gw): p.data.sub_(xi, g)         # w'
        dalpha = torch.autograd.grad(L_val(model, vb, alpha), alpha, retain_graph=True)
        v      = torch.autograd.grad(L_val(model, vb, alpha), model.weights())
        hv = hessian_vector(model, alpha, tb, v, eps)
        for p, w in zip(model.weights(), w0): p.data.copy_(w)            # restore w
        for a, da, h in zip(alpha, dalpha, hv): a.grad = da - xi * h
    opt_a.step()

def search(model, alpha, train_loader, val_loader, xi, eps):
    opt_w = torch.optim.SGD(model.weights(), lr=xi, momentum=0.9, weight_decay=3e-4)
    opt_a = torch.optim.Adam(alpha, lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    for tb, vb in zip(train_loader, val_loader):
        arch_step(model, alpha, tb, vb, xi, eps, opt_a)
        opt_w.zero_grad(); L_train(model, tb, alpha).backward(); opt_w.step()
```
