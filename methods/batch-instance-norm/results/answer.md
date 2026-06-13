# Batch-Instance Normalization (BIN), distilled

Batch-Instance Normalization is a drop-in replacement for the per-channel normalization layer
in a CNN that learns, **per channel**, how much image *style* to keep versus remove. It computes
both the batch-normalized response (which preserves instance-level style) and the
instance-normalized response (which removes it), and blends them with a learnable per-channel
gate `ρ ∈ [0,1]^C`: `ρ = 1` recovers pure BatchNorm (style kept), `ρ = 0` recovers pure
InstanceNorm (style scrubbed). The network opens the gate on channels where style is a useful
discriminative feature and closes it on channels where style is nuisance.

## Problem it solves

Image *style* — texture, contrast, brightness, lighting, camera/filter color cast — varies in
ways usually irrelevant to the object class and complicates recognition. Removing it wholesale
(InstanceNorm) hurts classification, because in some channels style *is* the signal (brightness
for weather prediction, texture for "spotted vs. striped"). Keeping it wholesale (BatchNorm)
cannot scrub the nuisance. The right choice differs per channel, and BIN learns it.

## Key idea

Treat the per-channel **mean and variance** of a feature map as its *style* (following the
style-transfer reading that aligning feature mean/variance controls style). With input
`x ∈ R^{N×C×H×W}`:

- **Batch-normalized branch** pools statistics per channel over the batch and spatial axes
  `(N,H,W)` — keeps per-instance style (subtracts one shared mean per channel):

  `x̂^(B)_{nchw} = (x_{nchw} − μ^(B)_c) / sqrt(σ²^(B)_c + ε)`,
  `μ^(B)_c = mean over (N,H,W)`, `σ²^(B)_c = var over (N,H,W)`.

- **Instance-normalized branch** pools statistics per `(n,c)` over the spatial axes `(H,W)` —
  removes per-instance style (subtracts each image's own mean):

  `x̂^(I)_{nchw} = (x_{nchw} − μ^(I)_{nc}) / sqrt(σ²^(I)_{nc} + ε)`,
  `μ^(I)_{nc} = mean over (H,W)`, `σ²^(I)_{nc} = var over (H,W)`.

- **Per-channel convex mix, then shared affine** with `γ, β ∈ R^C` and gate `ρ ∈ [0,1]^C`:

  `y = (ρ · x̂^(B) + (1 − ρ) · x̂^(I)) · γ + β`.

The endpoints are exactly BN (`ρ=1`) and IN (`ρ=0`); in between it interpolates. The gate is the
whole contribution — `C` extra parameters per layer, negligible compute, drop-in for BN.

## Why each design choice

- **Convex scalar mix (not concat / not a learned net):** adds only `C` params so any gain is
  not a capacity effect; both branches share shape and the same `γ,β`; endpoints are exactly the
  two trusted methods.
- **`ρ ∈ [0,1]` by clipping after each update (not sigmoid):** keeps `ρ` a literal mixing weight
  with an honest gradient (a sigmoid's `ρ(1−ρ)` factor vanishes at the 0/1 endpoints where `ρ`
  should settle). Constraint: `ρ ← clip_{[0,1]}(ρ − η Δρ)`.
- **Init `ρ = 1` (start as pure BN):** BN is the proven recognition default; open IN only where
  it helps. Starting at 0.5 or 0 hands IN's degradation to every channel up front.
- **Amplified learning rate for `ρ` (×10):** the gate gradient is proportional to the *small*
  difference `x̂^(B) − x̂^(I)` (see below), so `ρ` is sluggish; a matched LR multiplier lets it
  travel to its preference.
- **No weight decay on `ρ`:** `ρ` is a mixing coefficient, not a magnitude; decaying it would
  bias every channel toward 0 (pure IN) and fight the init and the data.
- **Per-channel (not per-layer) gate:** channels in one layer carry different styles; a single
  scalar can't keep one channel's style while removing another's.

## Gate gradient (why `ρ` needs an amplified learning rate)

With `y_{nchw} = γ_c·(ρ_c x̂^(B)_{nchw} + (1−ρ_c) x̂^(I)_{nchw}) + β_c`,
`∂y_{nchw}/∂ρ_c = γ_c (x̂^(B)_{nchw} − x̂^(I)_{nchw})`, so

```
∂ℓ/∂ρ_c = γ_c · Σ_n Σ_h Σ_w ( x̂^(B)_{nchw} − x̂^(I)_{nchw} ) · ∂ℓ/∂y_{nchw}.
```

The factor `x̂^(B) − x̂^(I)` is small whenever the minibatch's per-channel style variation is
marginal (then `μ^(B)_c ≈ μ^(I)_{nc}`, `σ^(B)_c ≈ σ^(I)_{nc}`, so the two normalizations nearly
agree). A small gradient means `ρ` barely moves, so its learning rate is scaled up to compensate.

## Working code

Drop-in replacement for `nn.BatchNorm2d`, matching the implementation pattern of the existing
batch-normalization layer: inherit the ordinary affine and running-stat fields, fold `ρ` into the
branch weights, compute the IN branch by reshaping to `[1, N*C, H, W]`, and tag the gate so the
training loop can give it special optimizer treatment.

```python
import torch
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)                 # init pure BN
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)
        # BN branch carries affine weight (gamma * gate) and bias beta
        bn_w = self.weight * self.gate if self.affine else self.gate
        out_bn = F.batch_norm(input, self.running_mean, self.running_var, bn_w, self.bias,
                              self.training, self.momentum, self.eps)
        # IN branch via [1, b*c, H, W] reshape; affine weight (gamma * (1 - gate)), no bias
        b, c = input.size(0), input.size(1)
        in_w = self.weight * (1 - self.gate) if self.affine else (1 - self.gate)
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(input, None, None, None, None, True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])
        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
```

Optimizer + training loop: a separate parameter group for the gate (lr x10, no weight decay),
and a clip to `[0,1]` after each step:

```python
import torch.optim as optim


def set_optimizer(model, args):
    params = [{'params': [p for p in model.parameters()
                          if not getattr(p, 'bin_gate', False)]},
              {'params': [p for p in model.parameters()
                          if getattr(p, 'bin_gate', False)],
               'lr': args.lr * args.bin_lr, 'weight_decay': 0}]
    return optim.SGD(params,
                     lr=args.lr,
                     momentum=args.momentum,
                     weight_decay=args.weight_decay)


def train(trainloader, model, criterion, optimizer):
    model.train()
    bin_gates = [p for p in model.parameters() if getattr(p, 'bin_gate', False)]
    for inputs, targets in trainloader:
        loss = criterion(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for p in bin_gates:
            p.data.clamp_(min=0, max=1)
```

This is mathematically `y = (ρ·x̂^(B) + (1−ρ)·x̂^(I))·γ + β` with `ρ` initialized to 1, clipped to
`[0,1]`, trained at an amplified learning rate, and exempted from weight decay.
