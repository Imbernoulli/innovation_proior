# SNIP: Single-shot Network Pruning based on Connection Sensitivity

## Problem

Prune a neural network to a hard sparsity budget ‖w‖_0 ≤ κ without losing accuracy, and do it **once, at initialization, before training** — eliminating the expensive prune–retrain cycles and the weight-scale-dependent, architecture-sensitive criteria of prior pruning methods.

## Key idea

Decouple *whether a connection exists* from *how strong it is* by introducing a per-connection gate c_j ∈ {0,1}, writing the effective weights as c ⊙ w. The importance of connection j is how much the loss reacts to its gate. Relaxing c to be continuous, this is the derivative

```
g_j = ∂L(c ⊙ w; D)/∂c_j |_{c=1} = (∂L/∂u_j) · w_j,   u_j = c_j w_j.
```

Because the gate perturbs the weight *multiplicatively* (effective perturbation δ·w_j), the score self-normalizes by weight scale and is therefore meaningful at random initialization — unlike magnitude (|w_j|) or curvature (w_j² H_jj/2), which require a trained network. **Connection sensitivity** is the normalized magnitude:

```
s_j = |g_j| / Σ_k |g_k|.
```

All g_j are obtained in a single forward–backward pass over one mini-batch. Keep the top-κ connections (c_j = 1[s_j − s̃_κ ≥ 0], s̃_κ the κ-th largest score), then train the resulting sparse network normally.

Two requirements make the score robust: the initial weights must be in a sensible range (else activations saturate and gradients are uninformative), and a **variance-scaling initialization** (Glorot-style) must be used so the signal variance is preserved across layers — otherwise the saliency would track architectural depth/width rather than task importance.

## Algorithm

```
Input: loss L, dataset D, sparsity level κ
1. w ← VarianceScalingInitialization;  gates c ← 1
2. sample a mini-batch D^b ~ D
3. s_j ← |g_j(w; D^b)| / Σ_k |g_k(w; D^b)|   for all j   (one forward-backward pass)
4. s̃ ← SortDescending(s)
5. c_j ← 1[ s_j − s̃_κ ≥ 0 ]                  (retain top-κ connections)
6. w* ← argmin_w L(c ⊙ w; D)                  (standard training of the sparse net)
7. return w* ← c ⊙ w*
```

No pretraining, no pruning schedule, no hyperparameters beyond κ.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))  # gate c = 1
        self.weight.requires_grad = False                              # freeze w to get dL/dc
    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class MaskedLinear(nn.Linear):
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

def snip_mask(net, batch, keep_fraction):
    x, y = batch
    net.zero_grad()
    F.cross_entropy(net(x), y).backward()                 # gates at c = 1
    masked = [m for m in net.modules() if isinstance(m, (MaskedConv2d, MaskedLinear))]
    scores = [torch.abs(m.weight_mask.grad) for m in masked]
    flat = torch.cat([s.flatten() for s in scores])
    norm = flat.sum()
    flat = flat / norm                                    # s_j = |g_j| / Σ|g|
    k = int(keep_fraction * flat.numel())
    thr = torch.topk(flat, k, sorted=True).values[-1]     # s̃_κ
    masks = [((s / norm) >= thr).float() for s in scores]
    return masked, masks

def prune_and_train(net, masked, masks, loader, epochs, lr):
    for m, msk in zip(masked, masks):
        m.weight_mask.data = msk            # fix binary mask c
        m.weight_mask.requires_grad = False
        m.weight.requires_grad = True       # train surviving weights
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()                      # pruned weights stay zero (mask fixed at 0)
```
