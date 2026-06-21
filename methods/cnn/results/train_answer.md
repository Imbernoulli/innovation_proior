The problem is to learn a mapping from a structured, high-dimensional input to a target by gradient descent, where the input coordinates are laid out along an ordered axis rather than being an unordered bag. In the climate-emulation instantiation, the first 540 coordinates are nine physical profiles sampled at sixty ordered vertical levels, and the remaining coordinates are whole-column scalars. A fully-connected network on the flattened vector fails in three ways: it uses an enormous number of independent weights, which by the capacity relation E_test − E_train ≈ k(h/P)^α inflates the generalization gap for fixed data; it has no built-in tolerance to where a local feature appears, so the same detector must be relearned at every position; and it discards the input topology, because permuting the coordinates would leave training unchanged. Earlier baselines in this ladder either throttled the multi-level path to a handful of scalars and starved the profile targets, or flattened the column and ignored its vertical structure. What is needed is an architecture that exploits the ordered vertical axis, shares parameters across positions, and is still trainable end to end by ordinary back-propagation.

The method I propose is the Convolutional Neural Network, specifically a ResNet-style 1D CNN applied to the vertical atmospheric profile. The core idea is to put the sixty vertical levels on the convolution axis and treat the physical variables as channels. A width-3 kernel then mixes a level with its two immediate neighbors, so it acts as a local vertical-gradient or curvature detector, and the same kernel weights are shared across all sixty levels. This weight sharing turns the operation into a convolution: it decouples the parameter count from the axis length, gives translation equivariance for free, and bakes the input ordering into the architecture rather than ignoring it. Whole-column scalars, which do not live on the vertical axis, are first projected by a learned linear map onto a length-60 vector and appended as an extra channel, so every input becomes a set of channels over the same ordered levels.

The network stacks several residual convolutional blocks. Each block has the form h ← h + F(h), where F consists of batch normalization, a width-3 convolution, a ReLU, dropout, and another width-3 convolution. The identity skip makes "do almost nothing" the easy default, which is what allows a deep stack of convolutions to remain trainable instead of degrading as more layers are added. Batch normalization and ReLU together play the modern role of keeping activations in range and gradients healthy. After the residual stack, the output is read off by two structure-matched heads. A 1×1 convolution maps the hidden channels at each level to six output channels, producing the 6×60=360 per-level tendency targets. The eight whole-column scalar targets have no level index, so the vertical axis is collapsed by adaptive average pooling and a small fully-connected head maps the pooled vector to eight scalars. The two heads are concatenated to give the final 368-dimensional output.

Back-propagation through a shared-weight layer is straightforward: the gradient of each tied kernel weight is the sum of its per-position contributions, and the input gradient is a convolution with the reversed kernel. The kernels are therefore tuned end to end to the actual supervised loss, unlike a frozen hand-designed feature extractor or a self-organized hierarchy with no discriminative objective.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom(nn.Module):
    """1D CNN with residual blocks for atmospheric column emulation.

    The 60 ordered vertical levels are the convolution axis and the multi-level
    physical variables are channels. Whole-column scalars are projected onto the
    axis as one learned channel. Residual conv blocks detect local vertical
    features with shared kernels; a 1x1 conv head reads per-level tendencies and
    a pooled MLP head reads whole-column scalars.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_ml_in = 9
        self.n_levels = 60
        self.n_sl_in = input_dim - self.n_ml_in * self.n_levels

        self.scalar_proj = nn.Linear(self.n_sl_in, self.n_levels)

        in_channels = self.n_ml_in + 1
        hidden_channels = 128
        n_blocks = 8

        self.input_conv = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ))

        self.n_ml_out = 6
        self.ml_head = nn.Conv1d(
            hidden_channels, self.n_ml_out, kernel_size=1
        )

        self.sl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        B = x.shape[0]
        ml_in = x[:, :self.n_ml_in * self.n_levels].view(
            B, self.n_ml_in, self.n_levels
        )
        sl_in = x[:, self.n_ml_in * self.n_levels:]
        sl_expanded = self.scalar_proj(sl_in).unsqueeze(1)
        h = torch.cat([ml_in, sl_expanded], dim=1)

        h = F.relu(self.input_conv(h))
        for block in self.blocks:
            h = h + block(h)

        ml_out = self.ml_head(h).reshape(B, -1)
        sl_out = self.sl_head(h)
        return torch.cat([ml_out, sl_out], dim=-1)
```
