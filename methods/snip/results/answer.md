# SNIP: Single-shot Network Pruning based on Connection Sensitivity

## Problem

Prune a neural network to a hard sparsity budget ‖w‖_0 ≤ κ without losing accuracy, and do it **once, at initialization, before training** — eliminating the expensive prune–retrain cycles and the weight-scale-dependent, architecture-sensitive criteria of prior pruning methods.

## Key idea

Decouple *whether a connection exists* from *how strong it is* by introducing a per-connection gate c_j ∈ {0,1}, writing the effective weights as c ⊙ w. The importance of connection j is how much the loss reacts to its gate. Relaxing c to be continuous, this is the derivative

```
g_j = ∂L(c ⊙ w; D)/∂c_j |_{c=1} = (∂L/∂u_j) · w_j,   u_j = c_j w_j.
```

Because the gate perturbs the effective weight *multiplicatively* (a small gate change gives an effective perturbation δ·w_j), the score measures the loss response to removing that connection's own contribution, rather than to a uniform additive weight nudge. It still depends on the initialized weights, which is why initialization scale matters, but it does not require those weights to be pretrained. **Connection sensitivity** is the normalized magnitude of the gate-gradient product:

```
a_j = |g_j c_j|,   c = 1,
s_j = a_j / Σ_k a_k = |g_j| / Σ_k |g_k|.
```

All g_j are obtained in a single forward–backward pass over one mini-batch at initialization. Normalization does not change the ranking, so the retained mask is the top-κ set by |∂L/∂c|, equivalently |g ⊙ c| at c=1: c_j = 1[s_j − s̃_κ ≥ 0], with s̃_κ the κ-th largest score. Then train the resulting sparse network normally.

Two requirements make the score robust: the initial weights must be in a sensible range (else activations saturate and gradients are uninformative), and a **variance-scaling initialization** (Glorot- or He-style) must be used so the signal variance is preserved across layers — otherwise the saliency can track architectural depth/width rather than task importance.

## Algorithm

```
Input: loss L, dataset D, sparsity level κ
1. w ← VarianceScalingInitialization;  gates c ← 1
2. sample a mini-batch D^b ~ D
3. a_j ← |g_j(w; D^b)c_j|, with c = 1, for all j   (one forward-backward pass)
4. s_j ← a_j / Σ_k a_k
5. s̃ ← SortDescending(s)
6. c_j ← 1[ s_j − s̃_κ ≥ 0 ]                  (retain top-κ connections)
7. w* ← argmin_w L(c ⊙ w; D)                  (standard training of the sparse net)
8. return w* ← c ⊙ w*
```

No pretraining, no pruning schedule, no hyperparameters beyond κ.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GatedConv2d(nn.Conv2d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))  # gate c = 1
        self.weight.requires_grad = False                              # freeze w to get dL/dc
    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class GatedLinear(nn.Linear):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))
        self.weight.requires_grad = False
    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def variance_scaling_init(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

def compute_keep_mask(net, batch, keep_fraction):
    x, y = batch
    net.zero_grad(set_to_none=True)
    F.cross_entropy(net(x), y).backward()                 # gates at c = 1
    gated = [m for m in net.modules() if isinstance(m, (GatedConv2d, GatedLinear))]
    scores = [torch.abs(m.weight_mask.grad * m.weight_mask).detach() for m in gated]  # |g * c|
    flat_scores = torch.cat([s.flatten() for s in scores])
    flat_scores = flat_scores / flat_scores.sum().clamp_min(torch.finfo(flat_scores.dtype).tiny)

    k = int(keep_fraction * flat_scores.numel())
    k = max(1, min(k, flat_scores.numel()))
    keep = torch.zeros_like(flat_scores)
    keep[torch.topk(flat_scores, k, sorted=False).indices] = 1.0

    masks, offset = [], 0
    for score in scores:
        n = score.numel()
        masks.append(keep[offset:offset + n].view_as(score))
        offset += n
    return gated, masks

def prune_and_train(net, gated, masks, loader, epochs, lr):
    for m, msk in zip(gated, masks):
        with torch.no_grad():
            m.weight_mask.copy_(msk)        # fix binary mask c
        m.weight_mask.requires_grad = False
        m.weight.requires_grad = True       # train surviving weights
    net.zero_grad(set_to_none=True)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()                      # pruned weights stay zero (mask fixed at 0)
```
