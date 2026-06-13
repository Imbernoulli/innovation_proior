# Convolutional neural network (CNN)

A convolutional neural network replaces the dense first layers of a fully-connected net with
**local receptive fields**, **weight sharing**, and **sub-sampling/pooling**, and trains the
whole stack end-to-end by back-propagation. Each layer slides a small shared kernel over an
ordered axis of the input, so it detects a local feature *everywhere* with one set of weights —
giving few parameters, translation equivariance, and an architecture built on the input's
topology instead of blind to it. For climate emulation, the ordered axis is the 60 vertical levels
of an atmospheric column and the physical variables are channels: a **ResNet-style 1D CNN** over
the vertical profile.

## The problem

Learn, by gradient descent, a map from a structured, high-dimensional input — variables laid out
along an ordered axis, with local features that recur at different positions — to a target. A
fully-connected net on the raw input fails three ways: (1) parameter explosion → large capacity
`h` → by `E_test − E_train ≈ k(h/P)^α` a large generalization gap for fixed data `P`, plus high
memory; (2) no built-in tolerance to *where* a feature appears, so the same detector must be
relearned at every position from extra data; (3) it ignores input topology (the coordinate
ordering), the one strong prior the problem hands you.

## The key idea

Constrain the net using the structure it was ignoring:

- **Local receptive fields.** Connect each unit only to a small window of neighboring positions.
  Cuts connections; matches "features are local."
- **Weight sharing = convolution.** Replicate one small kernel across all positions (an equality
  constraint among connections). Forward: `x_i = Σ_m w_m · o_{i+m} + b`, then a nonlinearity.
  Several kernels per layer → several **feature maps**. Consequences: parameter count decouples
  from axis length (so `h` drops and generalization improves); **translation equivariance** is
  structural (shift the input → the feature map shifts, otherwise unchanged); the layer is built
  on the input ordering.
- **Trainable by ordinary back-prop.** A shared weight's gradient is the sum of its per-position
  gradients, `∂E/∂w_m = Σ_i δ_i · o_{i+m}`. For an input position `j`,
  `∂E/∂o_j = Σ_m w_m · δ_{j-m}`; this is the full-convolution input gradient, equivalently a
  slide-and-dot backward pass with the finite kernel flipped. So the kernels are tuned
  end-to-end to the loss — unlike a hand-designed front end (frozen) or the self-organized
  neocognitron (no supervised objective).
- **Sub-sampling / pooling.** After detection, exact position is irrelevant and *harmful* (it
  varies across examples). Pool over small neighborhoods and reduce resolution → shift
  *invariance* (not just equivariance) and a further parameter/compute cut. Keep position only
  *approximately* so higher layers can still relate features.
- **Bi-pyramid.** Alternate convolution (increase #feature maps) and sub-sampling (decrease
  resolution) → a local→global, simple→complex feature hierarchy (the Hubel-Wiesel /
  neocognitron motif), now one differentiable system trained by gradient descent.

## Design decisions and why

| Decision | Why this, not the alternative |
|---|---|
| Local receptive fields | Features are local; full connection wastes parameters on far coords and needs more data. Window = smallest that captures the local feature (kernel 3 over levels = a level + its two neighbors → vertical gradient/curvature). |
| Weight sharing (convolution) | A detector useful at one position is useful at all; sharing decouples params from axis length, gives free translation equivariance, and shrinks the `(h/P)^α` gap. |
| Several feature maps per layer | One kernel = one feature; many local features are needed, so many kernels, each its own shared weights + bias. |
| Pooling after detection | Exact position is a nuisance variable; pooling converts equivariance to invariance and cuts resolution/params. |
| ↓ resolution, ↑ #maps with depth | Trade spatial precision for representational richness; build large effective receptive fields for high-order features. |
| End-to-end back-prop, not unsupervised self-organization | Gradient descent tunes the kernels to the actual task (the neocognitron could not); weight-sharing gradient = summed tied-connection gradients, cheap. |
| Scaled-tanh `A·tanh(Sa)`, with `A=1.7159` and `S=2/3`, ±1 targets / fan-in init | Symmetric, zero-mean outputs and operating-range targets avoid saturation and condition learning. **Modern stand-in (used in code): ReLU + BatchNorm**, same goals, what makes a deep stack trainable. |
| 1D conv over vertical levels (variables = channels) | The atmospheric column is genuinely 1D-ordered (height) with locally-coupled physics; conv exploits this and shares one vertical detector across all 60 levels. (Time-Delay Neural Networks are the 1D-conv precedent.) |
| Residual block `h <- h + F(h)` for the deep 1D stack | A deep plain conv stack *degrades* (training error rises) because identity-like added layers are hard to optimize; the additive identity skip makes "leave the input alone" the easy default and keeps the deep stack trainable. A 1x1-conv projection is used on the skip when channel count changes. |
| Two output heads | Targets are mixed: 6 multi-level tendencies (per-level → 1×1 conv readout) and 8 whole-column scalars (no level index → pool then MLP). |

## The architecture (climate-emulation 1D CNN)

- Split the flat input into **9 multi-level channels × 60 levels = 540 profile features** and
  the remaining whole-column scalars; project the scalar vector to a length-60 vector and append
  it as one learned channel.
- **Input conv** (kernel 3, "same" padding) lifts to a hidden width.
- A stack of **residual blocks**: `BatchNorm -> Conv1d(k3) -> ReLU -> Dropout -> Conv1d(k3)`,
  then `h <- h + F(h)`. The ClimSim Conv1D baseline uses tensors shaped `(batch, lev, vars)`,
  two same-padded `Conv1D` layers per block, and a 1x1 projected skip. The MLS-Bench edit below
  keeps the flat `Custom(input_dim, output_dim)` interface, switches to PyTorch's channel-first
  `Conv1d` layout, and uses a constant-width identity skip after the input projection.
- **Per-level head:** `Conv1d(hidden, 6, kernel_size=1)` → 6 × 60 = 360 multi-level tendencies.
- **Whole-column head:** adaptive average pool over levels → small MLP → 8 scalar diagnostics.
- Concatenate → 368-dim output.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom(nn.Module):
    """1D CNN with residual blocks over the vertical atmospheric profile.

    The 60 ordered vertical levels are the convolution axis; the multi-level
    variables are channels over that axis. Single-level scalars are projected onto
    the axis as one learned 60-level channel. Residual conv blocks detect local vertical features
    (each kernel shared across all heights); a 1x1 conv head reads off the per-level
    tendencies and a pooled MLP head reads off the whole-column scalars.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input layout: 9 multi-level vars x 60 levels = 540, then the
        # remaining scalar column variables.
        self.n_ml_in = 9
        self.n_levels = 60
        self.n_sl_in = input_dim - self.n_ml_in * self.n_levels

        # Learned map of the whole-column scalars onto the vertical axis.
        self.scalar_proj = nn.Linear(self.n_sl_in, self.n_levels)

        in_channels = self.n_ml_in + 1          # multi-level vars + projected scalars
        hidden_channels = 128
        n_blocks = 8

        # Input convolution: kernel 3 = a level and its two neighbors; pad keeps 60.
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Residual blocks: each learns F(h); h <- h + F(h) keeps the deep stack
        # optimizable. BatchNorm + ReLU keep activations in range and gradients
        # healthy; dropout regularizes.
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ))

        # Per-level head: 1x1 conv -> 6 tendency channels at each of the 60 levels.
        self.n_ml_out = 6
        self.ml_head = nn.Conv1d(hidden_channels, self.n_ml_out, kernel_size=1)

        # Whole-column head: pool the vertical axis, then an MLP -> 8 scalars.
        self.sl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        B = x.shape[0]
        ml_in = x[:, :self.n_ml_in * self.n_levels].view(B, self.n_ml_in, self.n_levels)
        sl_in = x[:, self.n_ml_in * self.n_levels:]
        sl_expanded = self.scalar_proj(sl_in).unsqueeze(1)   # (B, 1, 60) scalar-derived channel
        h = torch.cat([ml_in, sl_expanded], dim=1)           # (B, n_ml_in+1, 60)

        h = F.relu(self.input_conv(h))
        for block in self.blocks:
            h = h + block(h)                                 # residual: h <- h + F(h)

        ml_out = self.ml_head(h).reshape(B, -1)              # (B, 360)
        sl_out = self.sl_head(h)                             # (B, 8)
        return torch.cat([ml_out, sl_out], dim=-1)           # (B, 368)
```

The ClimSim Conv1D form keeps the vertical axis second, `(batch, lev=60, vars)`, and repeats or
stacks global variables along that level axis before applying two same-padded `Conv1D` layers and
a projected skip in each block. The MLS-Bench edit keeps the flat `Custom(input_dim, output_dim)`
surface: it reshapes the first `9*60` values to `(batch, 9, 60)`, learns one scalar-derived
60-level channel, uses `Conv1d` over the level axis, and returns exactly `6*60 + 8 = 368`
outputs.
