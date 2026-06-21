The central difficulty in training image classifiers is that the same object can appear under wildly different visual styles. Brightness, contrast, texture, lighting conditions, camera color cast, and weather effects all change the pixel and feature statistics, yet most of that variation is irrelevant to the object category. The traditional fix is to add more data and depth so the network implicitly learns to ignore these nuisances, but that is expensive and indirect. What I want is a normalization layer that can explicitly scrub away disturbing style variation while preserving style wherever it happens to carry useful discriminative signal, because the same statistic can be noise for one task and signal for another.

The existing normalization layers each sit at a fixed extreme of this trade-off, and none of them can adapt. Batch Normalization pools mean and variance over the whole minibatch and spatial dimensions for each channel. It subtracts one shared per-channel mean, so if one image is bright and another is dim, their difference survives; it preserves instance-level style. That is helpful when style is the signal, but it does nothing to remove style when it is nuisance. Instance Normalization does the opposite: it normalizes each image independently per channel over spatial dimensions, subtracting each image's own mean and dividing by its own standard deviation. This strips instance-specific style and is excellent for stylization, but dropped into a classifier it degrades accuracy because it also erases style channels where that style is exactly what the label depends on. A fixed blend of the two, or switching to Layer Normalization or Group Normalization, still commits to a single global rule rather than deciding per channel whether style should stay or go.

Batch-Instance Normalization resolves this by learning the decision per channel. The premise, borrowed from the style-transfer literature, is that the style of a feature map is captured by its per-channel mean and variance, while the shape is captured by the spatial configuration of activations. Given an input tensor of shape [N, C, H, W], the layer computes two normalized responses. The batch-normalized branch pools statistics per channel over the batch and spatial axes (N, H, W), producing the usual BatchNorm output that keeps per-instance style differences. The instance-normalized branch pools statistics per (n, c) over only the spatial axes (H, W), producing an InstanceNorm output that removes per-instance style. These two responses are then combined with a learnable per-channel convex gate ρ ∈ [0, 1]^C, followed by the standard learnable affine γ, β per channel: y = (ρ · x̂^(B) + (1 − ρ) · x̂^(I)) · γ + β. When ρ_c is 1 the channel is pure BatchNorm; when ρ_c is 0 it is pure InstanceNorm; values in between interpolate. The network thus opens the gate toward BatchNorm on channels where style is useful and closes it toward InstanceNorm on channels where style is nuisance.

The design choices are deliberately minimal. A scalar convex mix per channel adds only C extra parameters per layer, so any improvement cannot be attributed to adding capacity. Both branches share the same output shape and the same affine parameters, making the layer a true drop-in replacement for BatchNorm. The gate is constrained to [0, 1] by clipping after each optimizer step rather than by a sigmoid, because a sigmoid would bend the gradient and vanish exactly at the endpoints where the gate most needs to settle. The gate is initialized to 1 so training starts from the safe, proven BatchNorm behavior and only opens the InstanceNorm branch where the gradient says it helps. The gradient of the loss with respect to ρ_c is proportional to γ_c times the sum over the batch and spatial locations of (x̂^(B) − x̂^(I)) · ∂ℓ/∂y. The difference between the two normalized responses is small whenever the minibatch's style variation is marginal, which makes the gate sluggish under a normal learning rate, so the gate is trained with a ten-times larger learning rate. Weight decay is removed from ρ because it is a mixing coefficient, not a magnitude; decaying it would bias every channel toward pure InstanceNorm and fight both the initialization and the data.

In code the layer inherits from PyTorch's batch-norm base to reuse the affine parameters, running statistics, eps, momentum, and dimension checks. The batch branch uses the ordinary F.batch_norm primitive. The instance branch is obtained by reshaping the input from [N, C, H, W] to [1, N·C, H, W], so the same primitive pools over the spatial dimensions for each original (image, channel) pair. The gate is folded into the per-branch affine weights, with the BatchNorm branch carrying (γ · ρ) and β, and the InstanceNorm branch carrying (γ · (1 − ρ)) with no bias. The training loop then clips the gate to [0, 1] after each step and optimizes it with a higher learning rate and zero weight decay.

```python
import torch
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)
        bn_w = self.weight * self.gate if self.affine else self.gate
        out_bn = F.batch_norm(input, self.running_mean, self.running_var,
                              bn_w, self.bias, self.training,
                              self.momentum, self.eps)

        b, c = input.size(0), input.size(1)
        in_w = self.weight * (1 - self.gate) if self.affine else (1 - self.gate)
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(input, None, None, None, None,
                              True, self.momentum, self.eps)
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


import torch.optim as optim


def set_optimizer(model, base_lr, momentum=0.9, weight_decay=1e-4, bin_lr_mult=10.0):
    params = [{'params': [p for p in model.parameters()
                          if not getattr(p, 'bin_gate', False)]},
              {'params': [p for p in model.parameters()
                          if getattr(p, 'bin_gate', False)],
               'lr': base_lr * bin_lr_mult, 'weight_decay': 0}]
    return optim.SGD(params, lr=base_lr, momentum=momentum,
                     weight_decay=weight_decay)


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

This is mathematically y = (ρ · x̂^(B) + (1 − ρ) · x̂^(I)) · γ + β, with ρ initialized to 1, clipped to [0, 1], trained at an amplified learning rate, and exempted from weight decay.
