**Problem.** The linear ridge head inverted on the stability assay (ESTA_BACSU Spearman −0.107): a
hyperplane cannot bend around a stability cliff. The head must become nonlinear to carve delta-space,
and this rung also tests whether *structured weight-sharing* over the embedding features beats a flat
dense head.

**Key idea.** A convolutional head over the pooled embedding — but honestly so. The harness exposes
only the **mean-pooled** ESM-2 vector, no per-residue tokens, so a true Conv1D-over-residues is
impossible. Instead, concatenate `[embedding, delta_embedding]` to `[B, 2560]`, project and reshape
into a fake `(channels=64, length=40)` grid, and run 1D convolutions over the embedding-channel axis.
There is no real adjacency along that axis; the convolutions just enforce weight-sharing across blocks
of embedding dimensions — a different inductive bias than a dense layer, under test here.

**Why these choices.**
- **Both inputs concatenated:** unlike the linear head (delta only), a normalized nonlinear head can
  use the raw embedding to condition on which fitness regime the protein lives in.
- **Residual blocks (`out += residual`):** stacking conv-norm-nonlinearity layers degrades because
  learning the identity is hard and weight decay pulls toward the zero (annihilating) map. Learn the
  correction `F(x)` and add `x` back, so "leave alone" is `F=0` — the easiest target, where weight
  decay already pulls. The shortcut is parameter-free.
- **Activation after the add, branch ends conv→BN:** a ReLU/GELU at the end of the branch forces
  `F ≥ 0` (one-sided corrections only); rectify the sum so `F` can be signed.
- **Kernels 3/5/7, global-average-pool:** progressively wider receptive fields over the pseudo-axis;
  GAP collapses the meaningless length axis position-agnostically and parameter-free.

**Hyperparameters.** channels 64, length 40 (`2*1280/64`); three ConvBlocks (k=3,5,7), dropout 0.1,
BatchNorm + GELU; head `Linear(64,128)→GELU→Dropout(0.1)→Linear(128,1)`. `lr`/`weight_decay` left at
the loop defaults (no `CONFIG_OVERRIDES`); MSE loss, cosine schedule, early stopping fixed by the loop.

```python
# EDITABLE region of custom_mutation_pred.py — step 2: reshape-CNN over pooled ESM-2 features
class ConvBlock(nn.Module):
    """1D convolution block with BatchNorm and residual connection."""

    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x + residual


class MutationPredictor(nn.Module):
    """Reshape-CNN over mean-pooled ESM-2 features (NOT per-residue).

    Concatenates [embedding, delta_embedding] -> [B, 2*EMBED_DIM=2560],
    reshapes to (B, channels=64, length=40), applies a stack of 1D
    convolutions with residual connections over the embedding-channel
    axis, then global-average-pools and predicts. The reshape axis has
    NO real sequence structure.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.channels = 64
        self.length = (embed_dim * 2) // self.channels  # 40

        self.input_proj = nn.Linear(embed_dim * 2, self.channels * self.length)

        self.conv_blocks = nn.Sequential(
            ConvBlock(self.channels, kernel_size=3, dropout=0.1),
            ConvBlock(self.channels, kernel_size=5, dropout=0.1),
            ConvBlock(self.channels, kernel_size=7, dropout=0.1),
        )

        self.head = nn.Sequential(
            nn.Linear(self.channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, embedding, delta_embedding):
        x = torch.cat([embedding, delta_embedding], dim=-1)  # [B, 2*EMBED_DIM]
        x = F.gelu(self.input_proj(x))                       # [B, C*L]
        x = x.view(x.size(0), self.channels, self.length)    # [B, C, L]
        x = self.conv_blocks(x)                              # [B, C, L]
        x = x.mean(dim=-1)                                   # [B, C]
        return self.head(x).squeeze(-1)                      # [B]
```
