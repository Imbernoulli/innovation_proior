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

Approximate w*(α) by a single inner step w' = w − ξ ∇_w L_train(w, α) and differentiate through it. The Jacobian is ∂w'/∂α = −ξ ∇²_{w,α} L_train, so the transpose-Jacobian product contributes −ξ ∇²_{α,w} L_train · ∇_{w'}L_val:

```
d/dα L_val(w'(α), α) = ∂_α L_val(w', α) − ξ ∇²_{α,w} L_train(w, α) · ∇_{w'} L_val(w', α).
```

The Hessian-vector product is computed by central finite difference, with v = ∇_{w'}L_val(w',α), w± = w ± ε v (ε = 0.01/‖v‖₂):

```
∇²_{α,w} L_train(w, α) · v ≈ [ ∇_α L_train(w⁺, α) − ∇_α L_train(w⁻, α) ] / (2ε),
```

reducing cost from O(|α||w|) to O(|α|+|w|). Setting **ξ = 0** drops the second term, giving the cheaper **first-order** approximation ∇_α L_val(w, α) (faster but empirically worse); the **second-order** version **ξ > 0** keeps the one-step coupling between architecture and weights.

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

PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
NONE = PRIMITIVES.index('none')

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in PRIMITIVES:
            op = make_op(name, C, stride, affine=False)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class SearchNetwork(nn.Module):
    def _initialize_alphas(self, steps=4):
        k = sum(1 for i in range(steps) for _ in range(2+i))
        self.alphas_normal = nn.Parameter(torch.zeros(k, len(PRIMITIVES)))  # uniform softmax at init
        self.alphas_reduce = nn.Parameter(torch.zeros(k, len(PRIMITIVES)))
    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]
    def named_weight_parameters(self):
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) not in arch_ids]
    def weight_parameters(self):
        return [p for _, p in self.named_weight_parameters()]
    def genotype(self):
        return {
            'normal': parse_cell(F.softmax(self.alphas_normal, dim=-1).detach().cpu()),
            'reduce': parse_cell(F.softmax(self.alphas_reduce, dim=-1).detach().cpu()),
        }

def _concat(xs):
    return torch.cat([x.contiguous().view(-1) for x in xs])

def construct_model_from_theta(model, theta):
    model_new = model.new()
    model_dict = model.state_dict()
    offset = 0
    for name, p in model.named_weight_parameters():
        n = p.numel()
        model_dict[name] = theta[offset:offset+n].view_as(p)
        offset += n
    assert offset == theta.numel()
    model_new.load_state_dict(model_dict)
    return model_new

def compute_unrolled_model(model, train_batch, eta, weight_opt, momentum, weight_decay):
    weights = list(model.weight_parameters())
    theta = _concat([p.detach() for p in weights])
    loss = L_train(model, train_batch)
    moment = _concat([weight_opt.state[p].get('momentum_buffer', torch.zeros_like(p))
                      for p in weights]) * momentum
    grads = torch.autograd.grad(loss, weights)
    dtheta = _concat([g.detach() for g in grads]) + weight_decay * theta
    return construct_model_from_theta(model, theta - eta * (moment + dtheta))

def hessian_vector_product(model, train_batch, vector, r=1e-2):
    weights = list(model.weight_parameters())            # perturb current w, not unrolled w'
    R = r / _concat(vector).norm()
    for p, v in zip(weights, vector): p.data.add_(R, v)
    g_plus = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector): p.data.sub_(2*R, v)
    g_minus = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector): p.data.add_(R, v)
    return [(gp - gm) / (2*R) for gp, gm in zip(g_plus, g_minus)]

def arch_step(model, train_batch, val_batch, eta, weight_opt, arch_opt, unrolled=True):
    arch_opt.zero_grad()
    if not unrolled or eta == 0:
        L_val(model, val_batch).backward()
    else:
        unrolled = compute_unrolled_model(
            model, train_batch, eta, weight_opt, momentum=0.9, weight_decay=3e-4)
        L_val(unrolled, val_batch).backward()
        dalpha = [a.grad.detach().clone() for a in unrolled.arch_parameters()]
        vector = [p.grad.detach().clone() for p in unrolled.weight_parameters()]
        implicit = hessian_vector_product(model, train_batch, vector)
        for alpha, da, h in zip(model.arch_parameters(), dalpha, implicit):
            alpha.grad = (da - eta * h).detach().clone()
    arch_opt.step()

def search(model, train_loader, val_loader):
    opt_w = torch.optim.SGD(model.weight_parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    opt_a = torch.optim.Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    for tb in train_loader:
        vb = next(iter(val_loader))
        eta = opt_w.param_groups[0]['lr']
        arch_step(model, tb, vb, eta, opt_w, opt_a, unrolled=True)
        opt_w.zero_grad()
        L_train(model, tb).backward()
        nn.utils.clip_grad_norm_(model.weight_parameters(), 5)
        opt_w.step()

def parse_cell(weights, steps=4, k=2):
    gene, start, n = [], 0, 2
    for _ in range(steps):
        node_edges = weights[start:start+n]
        keep = sorted(range(n),
                      key=lambda j: -max(node_edges[j][op].item()
                                         for op in range(len(PRIMITIVES)) if op != NONE))[:k]
        for j in keep:
            op = max((op for op in range(len(PRIMITIVES)) if op != NONE),
                     key=lambda op: node_edges[j][op].item())
            gene.append((PRIMITIVES[op], j))
        start += n; n += 1
    return gene
```
