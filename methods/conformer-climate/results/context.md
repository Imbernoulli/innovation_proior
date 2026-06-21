## Research question

The setting is supervised regression of a structured-sequence input to a mixed target, where the input
coordinates lie on a *short, ordered, physically meaningful* axis. Concretely: emulate the sub-grid
atmospheric physics of a climate column. Each example is one atmospheric column — nine state variables, each a
profile sampled along 60 ordered vertical levels (540 numbers), plus ~16 whole-column scalars — and the target
is the column's sub-grid tendencies: six tendency profiles over the same 60 levels (360 numbers) and eight
single-level diagnostics. Learn the map by gradient descent on a fixed mean-squared-error loss, scored by
Normalized MSE (lower better) on a held-out split.

The map has two different kinds of dependency along height. One is **local along height**: a tendency at a
level depends most on that level and its immediate neighbors — local vertical gradients, curvature,
adjacent-level adjustments — and the same local interaction recurs at different heights. The other is
**long-range vertical coupling**: surface heat and moisture fluxes drive convection that deposits heating and
moistening hundreds of hectopascals higher; radiative cooling at a cloud top depends on the layers below it;
the column is one coupled system. The question is how to wire an architecture on a fixed-length 60-level axis,
within a fixed regression harness, to model the dependencies of this map.

## Background

The working stance in the field is "the architecture is the inductive bias." Two building blocks each
emphasize one of the two dependency types.

**Self-attention (Vaswani et al. 2017).** Relates every position to every other in a single layer, with
weights computed from content. Distance is no obstacle, one hop reaches anywhere. Each output is a
softmax-weighted average over all positions.

**Convolution.** A kernel slides over a local window, picking up local patterns cheaply with translation
equivariance. Its receptive field grows one window per layer; reaching across the whole sequence is done with a
deep stack or fat kernels.

**Adding a global summary to a conv stack.** A purely convolutional stack can bolt on a single averaged global
summary (squeeze-and-excitation style) to inject context — a static averaged vector shared across positions.

**The macaron / ODE view of the Transformer block.** A standard Transformer block is one position-wise
feed-forward layer after attention, each in a residual unit. The macaron view treats the block like a step of
an ODE solver and splits the lone feed-forward into two *half-step* feed-forward layers — one before and one
after the mixing operation, each with a half-weighted residual — which approximates the underlying dynamics
better than one full step on one side.

**Multi-resolution context (the U-Net line).** Coarsening the axis by pooling lets a local window reach distant
positions, and skip connections restore the localization pooling destroys; a single self-attention block at the
coarse bottleneck couples distant *coarse* positions. The global coupling acts at coarse resolution: distant
*fine* levels relate through their pooled summaries, and the bottleneck attention sees a handful of coarse
positions.

## Baselines

**Flat MLP / encoder-decoder on the flat column.** Treats the 556 inputs as an unordered vector, with no
mechanism that uses the vertical axis as structure: it relearns the same local vertical interaction at every
height, or passes the column through a scalar bottleneck.

**1D convolution over the levels (residual conv stack).** Puts the 60 levels on the conv axis and the variables
as channels; captures local vertical structure with full representational width. A small kernel over a few
blocks builds a vertical receptive field set by depth and kernel size.

**1D U-Net over the levels (contracting/expanding with bottleneck attention).** Adds multi-resolution context
and one self-attention block at the coarse bottleneck, spanning long-range coupling *at coarse resolution*.
Local convolution acts at full resolution; global mixing acts on the ~16 coarse positions.

## Evaluation settings

A candidate model fills a single editable architecture slot in a fixed regression harness: normalized
mini-batches of flat column inputs and flat targets, a fixed `nn.MSELoss`, AdamW with cosine-annealed learning
rate, gradient-norm clipping, validation-based early stopping, three training budgets (30 / 100 / 200 epochs).
The interface is settled: the first `9 * 60 = 540` inputs are profiles on the ordered vertical axis, the rest
are whole-column scalars; the first `6 * 60 = 360` outputs are tendency profiles, the rest are scalar
diagnostics. Primary metric Normalized MSE (MSE / Var(target), per variable, lower better); secondary R²,
RMSE, and separate multi-level / single-level NMSE breakdowns. What is open is the internal architecture: how
to wire the layers on the 60-level axis.

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
