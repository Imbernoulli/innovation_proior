## Research question

The setting is supervised regression of a structured-sequence input to a mixed target, where the input
coordinates lie on a *short, ordered, physically meaningful* axis. Concretely: emulate the sub-grid
atmospheric physics of a climate column. Each example is one atmospheric column — nine state variables, each a
profile sampled along 60 ordered vertical levels (540 numbers), plus ~16 whole-column scalars — and the target
is the column's sub-grid tendencies: six tendency profiles over the same 60 levels (360 numbers) and eight
single-level diagnostics. Learn the map by gradient descent on a fixed mean-squared-error loss, scored by
Normalized MSE (lower better) on a held-out split.

The map has two genuinely different kinds of dependency that an architecture must capture *at once*. One is
**local along height**: a tendency at a level depends most on that level and its immediate neighbors —
local vertical gradients, curvature, adjacent-level adjustments — and the same local interaction recurs at
different heights. The other is **long-range vertical coupling**: surface heat and moisture fluxes drive
convection that deposits heating and moistening hundreds of hectopascals higher; radiative cooling at a cloud
top depends on the layers below it; the column is one coupled system. The precise goal is an architecture that
models both the sharp local per-level structure and the content-dependent long-range level-to-level coupling,
on a fixed-length 60-level axis, within a fixed regression harness.

## Background

The field state is "the architecture is the inductive bias." Two building blocks each nail exactly one of the
two dependency types, and faking the missing half is weak.

**Self-attention (Vaswani et al. 2017).** Relates every position to every other in a single layer, with
weights computed from content. That is exactly right for global, content-based interactions — distance is no
obstacle, one hop reaches anywhere. But at a fine scale each output is a softmax-weighted average over all
positions: a smoothing operation, not built to extract a crisp, position-specific local pattern. Globally
strong, locally blunt.

**Convolution.** A kernel slides over a local window, picking up local patterns cheaply with translation
equivariance — ideal for local detail. But its receptive field grows only one window per layer; reaching
across the whole sequence needs a deep stack or fat kernels, i.e. global reach is paid for in depth or
parameters. Locally strong, globally expensive.

**Patching one side is weak.** A purely convolutional stack can bolt on a single averaged global summary
(squeeze-and-excitation style) to inject context, but a static averaged vector cannot model dynamic,
position-dependent global interactions — it cannot say "the surface level should couple to the upper-level
position but not the mid-level one." A purely attentional stack lacks a sharp local operator. So neither block
alone is right.

**The macaron / ODE view of the Transformer block.** A standard Transformer block is one position-wise
feed-forward layer after attention, each in a residual unit. The macaron view treats the block like a step of
an ODE solver and argues the lone feed-forward should be split into two *half-step* feed-forward layers — one
before and one after the mixing operation, each with a half-weighted residual — which approximates the
underlying dynamics better than one full step on one side.

**Multi-resolution context (the U-Net line) as the alternative the ladder already tried.** Coarsening the
axis by pooling lets a local window reach distant positions, and skip connections restore the localization
pooling destroys; a single self-attention block at the coarse bottleneck couples distant *coarse* positions.
This is a strong way to get long-range reach, but the global coupling only acts at coarse resolution — distant
*fine* levels never relate directly, only through their pooled summaries — and the bottleneck attention sees
only a handful of coarse positions.

## Baselines

**Flat MLP / encoder-decoder on the flat column.** Treats the 556 inputs as an unordered vector; blind to the
vertical axis, so it relearns the same local vertical interaction at every height (parameter waste) or crushes
the column through a scalar bottleneck (starves the high-dimensional multi-level target). No mechanism for
either local-along-height structure or long-range coupling as *structure*.

**1D convolution over the levels (residual conv stack).** Puts the 60 levels on the conv axis and the
variables as channels; excellent at local vertical structure with full representational width. Limitation: a
small kernel over a few blocks builds only a limited vertical receptive field, so long-range vertical coupling
is out of reach except by stacking many windows (slow, and the part that plateaus).

**1D U-Net over the levels (contracting/expanding with bottleneck attention).** Adds multi-resolution context
and one self-attention block at the coarse bottleneck, spanning long-range coupling *at coarse resolution*.
Limitation: full-resolution level-to-level content-based coupling is never modeled directly — only local
convolution acts at full resolution, and global mixing acts only on ~16 coarse positions.

## Evaluation settings

A candidate model fills a single editable architecture slot in a fixed regression harness: normalized
mini-batches of flat column inputs and flat targets, a fixed `nn.MSELoss`, AdamW with cosine-annealed learning
rate, gradient-norm clipping, validation-based early stopping, three training budgets (30 / 100 / 200 epochs).
The interface is settled: the first `9 * 60 = 540` inputs are profiles on the ordered vertical axis, the rest
are whole-column scalars; the first `6 * 60 = 360` outputs are tendency profiles, the rest are scalar
diagnostics. Primary metric Normalized MSE (MSE / Var(target), per variable, lower better); secondary R²,
RMSE, and separate multi-level / single-level NMSE breakdowns. What is open is the internal architecture: how
to wire the layers so a single model captures *both* local-along-height and long-range level-to-level
dependencies on the 60-level axis.

## Code framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


N_LEVELS = 60
N_PROFILE_IN = 9
N_PROFILE_OUT = 6


class Custom(nn.Module):
    """Architecture slot for a column emulator.

    Input `x` has shape (batch_size, input_dim); the first 9*60 entries are
    profile variables on the ordered vertical axis, the rest are whole-column
    scalars. Output must be (batch_size, output_dim): first 6*60 entries are
    profile tendencies, the rest scalar diagnostics. Loss is fixed MSE.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # TODO: the architecture we will design.

    def forward(self, x):
        raise NotImplementedError
```

The data pipeline, loss, optimizer, metrics, and training loop are fixed. The single open piece is `Custom`:
the architecture that fills the slot while preserving this exact input/output contract.
