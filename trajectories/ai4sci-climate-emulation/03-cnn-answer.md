**Problem (from step 2).** ED's 5-node bottleneck starved the high-dimensional multi-level tendencies
(long NMSE 0.583, ml_nmse 0.595) — five scalars cannot carry the 360 profile targets. HSR over-flat,
ED over-throttled; both flatten the column into an unordered vector and ignore its structure. Give the
network back representational room, spent on a prior that matches the data's shape.

**Key idea.** The first 540 inputs are nine *profiles* on an ordered 60-level vertical axis, and the physics
is *local along height* and recurs at different heights. So put the 60 levels on the convolution axis and the
variables as channels: a width-3 kernel is a local vertical-gradient detector shared across all heights
(locality + weight-sharing kill ED/HSR's parameter waste; translation-equivariance for free). Project the
whole-column scalars onto the axis as one extra channel. Stack residual conv blocks (h ← h + F(h)) so the
deep vertical stack stays trainable. Read off the two target halves with structure-matched heads: a 1×1 conv
for the 360 per-level tendencies, a pooled MLP for the 8 whole-column scalars.

**Why it works.** Full-resolution local feature extraction over levels (no scalar throat) targets exactly the
metric ED throttled — ml_nmse. The risk: a width-3 kernel over a few blocks builds only a limited vertical
receptive field, so long-range level coupling may still be out of reach (a ceiling for the next rung).

**Scaffold edit / hyperparameters.** in_channels = 9 profiles + 1 scalar-projected; hidden 128; 8 residual
blocks (BatchNorm→Conv1d(k3)→ReLU→Dropout(0.1)→Conv1d(k3), identity skip); ml_head = Conv1d(128→6, k1)
→ 360; sl_head = AdaptiveAvgPool1d→Linear(128,64)→ReLU→Linear(64,8). BatchNorm+ReLU as the modern
stand-in for scaled-tanh+fan-in init. AdamW + cosine LR + MSE, all fixed.

```python
class Custom(nn.Module):
    """1D CNN with residual blocks for climate emulation.

    Reshapes input into (n_vars, n_levels) for convolution over vertical profiles,
    then projects back to output space.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input structure: 9 multi-level vars x 60 levels = 540, then 16-17 scalars
        self.n_ml_in = 9
        self.n_levels = 60
        self.n_sl_in = input_dim - self.n_ml_in * self.n_levels

        # Project scalar inputs to per-level features
        self.scalar_proj = nn.Linear(self.n_sl_in, self.n_levels)

        # Conv channels: n_ml_in + 1 (from scalar projection)
        in_channels = self.n_ml_in + 1
        hidden_channels = 128
        n_blocks = 8

        # Initial projection
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ))

        # Output: multi-level tendencies
        self.n_ml_out = 6
        self.ml_head = nn.Conv1d(hidden_channels, self.n_ml_out, kernel_size=1)

        # Output: single-level scalars from pooled features
        self.sl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        B = x.shape[0]
        # Split multi-level and single-level inputs
        ml_in = x[:, :self.n_ml_in * self.n_levels].view(B, self.n_ml_in, self.n_levels)
        sl_in = x[:, self.n_ml_in * self.n_levels:]
        sl_expanded = self.scalar_proj(sl_in).unsqueeze(1)  # (B, 1, 60)
        h = torch.cat([ml_in, sl_expanded], dim=1)  # (B, n_ml_in+1, 60)

        h = F.relu(self.input_conv(h))
        for block in self.blocks:
            h = h + block(h)

        ml_out = self.ml_head(h).reshape(B, -1)  # (B, 360)
        sl_out = self.sl_head(h)  # (B, 8)
        return torch.cat([ml_out, sl_out], dim=-1)
```
