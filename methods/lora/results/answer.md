# LoRA: Low-Rank Adaptation

## Problem

Adapt one very large frozen pre-trained Transformer to many downstream tasks while (1) storing
only a tiny per-task delta, (2) matching full-fine-tuning quality, (3) adding **no** extra
inference latency, and (4) not consuming any input sequence length. Full fine-tuning meets the
quality bar but fails the storage and optimizer-memory constraints: its delta is the size of the
whole model. Adapter layers fail (3): they add sequential depth that cannot be merged away.
Prefix/prompt tuning fails (4) and is hard to optimize.

## Key idea

Empirically, the *solution* found by fine-tuning lives on a very low-dimensional manifold (the
intrinsic-dimension result). LoRA conjectures the *weight update* itself is low rank. For a
pre-trained matrix W₀ ∈ R^{d×k}, constrain its update to a rank-r product and freeze W₀:

  h = W₀x + ΔW x = W₀x + (α/r) · B A x,   B ∈ R^{d×r}, A ∈ R^{r×k},  r ≪ min(d,k).

Only A and B are trained. This replaces d·k parameters per matrix with r·(d+k). For a square
12288×12288 projection and r = 4, that is 98,304 trainable parameters instead of 150,994,944,
a 1,536× reduction for that matrix.

Three design choices make it work:

- **Zero-start initialization.** Initialize one factor with a random (Gaussian / Kaiming)
  distribution and the *other factor to zero*, so BA = 0 at step 0 and the adapted model is
  *identical* to the pre-trained model initially. With h = W₀x + sBAx and g = ∂L/∂h,
  ∂L/∂B = s g(Ax)ᵀ and ∂L/∂A = s Bᵀg xᵀ. If both factors are zero, both gradients vanish. If
  B = 0 and A is random, B gets the first nonzero update; once B moves, A receives gradients too.
- **α/r scaling.** Scale the update by α/r. The 1/r term damps the branch as the number of rank
  channels changes, so rank is less entangled with branch gain and learning-rate tuning. α is
  fixed from the first r tried (with Adam, changing α behaves much like changing effective step
  size, up to initialization and optimizer details).
- **Merge for deployment.** Because BAx is a *parallel linear* branch on the same input as W₀x
  (no nonlinearity between them), it folds into a single matrix W = W₀ + (α/r)BA of the original
  shape. Inference is then one ordinary matmul → **zero added latency by construction**. Task
  switching = subtract (α/r)BA for the current task, add (α′/r′)B′A′ for the next task.

Freezing W₀ also removes its gradient and (with Adam) its optimizer state, cutting training memory
substantially. In a Transformer, LoRA can be applied to selected attention projections or other
linear layers; fused qkv projections need slice-level control so the update can target only the
chosen logical matrices.

## Algorithm

1. Replace each target linear layer (e.g. attention W_q, W_v) with a LoRA layer: keep the
   frozen pre-trained weight, add trainable A (r×in) and B (out×r).
2. Initialize A random, B = 0 (so BA = 0); set scaling = α/r.
3. Freeze all non-LoRA parameters; train only the A, B (and optionally biases) with AdamW.
4. Save only the LoRA parameters per task (megabytes).
5. At deployment, merge W ← W₀ + (α/r)BA for latency-free inference; unmerge to switch tasks.

## Code

```python
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else (lambda x: x)
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged and self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged and self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and any(self.enable_lora):
            result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
```

Usage:

```python
# Separate projections:
q_proj = Linear(d_model, d_model, r=8, lora_alpha=16)
v_proj = Linear(d_model, d_model, r=8, lora_alpha=16)

# Fused qkv projection with only q and v adapted:
qkv_proj = MergedLinear(d_model, 3 * d_model, r=8, lora_alpha=16,
                        enable_lora=[True, False, True])

mark_only_lora_as_trainable(model)
# ... train with AdamW ...
torch.save(lora_state_dict(model), 'task.pt')
model.eval()
```
