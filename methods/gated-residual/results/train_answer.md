Training very deep networks means wrestling with depth itself. A stack of L layers multiplies the per-layer scaling factor r by itself L times, so any r even slightly away from 1 sends forward signals and backward gradients vanishing or exploding exponentially. Mean-field analysis sharpens this to the condition chi = 1, the edge of chaos, which rescues the average perturbation. But dynamical isometry, the stronger requirement that every singular value of the input-output Jacobian stays near 1, is what actually makes deep learning fast. For ReLU networks and self-attention stacks it is provably impossible to achieve by any clever weight initialization: ReLU zeros out derivatives, LayerNorm is invariant to mean and variance shifts, and a near-uniform attention softmax collapses the embedding directions. Every standard fix therefore starts from a structurally damaged Jacobian.

Residual connections help, yet even they begin far from the identity. A vanilla residual block at standard He initialization adds a branch whose variance matches the skip path, so the hidden activation variance grows roughly as 2^l and the block output is half random branch, half skip. Batch normalization rescues depth only because it implicitly downscales the residual branch by about sqrt(depth), nudging blocks toward identity, but it adds compute and hyperparameters and its placement in Transformers remains contentious. Pre-activation ResNet moves activations off the merged path for cleaner gradient flow, yet the branch still contributes fully at init. Highway Networks use a learned input-dependent gate, but that gate is an entire weight tensor, never reaches exactly 0 or 1, and only biases the layer toward identity. Gated ResNet collapses the gate to a scalar, but initializes it so the block starts as a standard residual block and must unlearn its own random branch. Zero-gamma starts blocks at identity but only when a normalization layer sits at the end of the branch and it zeroes many parameters. FixUp works without normalization but needs a depth-dependent rescaling exponent and several coupled pieces. SkipInit uses a single scalar, yet frames it as a replacement for normalization and recommends a depth-dependent initial scale.

The method I propose is ReZero, short for residual with zero initialization. It places a single learnable scalar alpha directly on the residual branch and leaves the skip path untouched: x_{i+1} = x_i + alpha_i F(x_i), with alpha_i initialized to exactly 0. For dimension-preserving blocks this makes every block the identity map at initialization, so the entire network is the identity, the input-output Jacobian is exactly the identity matrix, and every singular value is exactly 1. This gives exact dynamical isometry for any branch F, including ReLU convolutions and self-attention, because the branch is multiplied by zero and its pathologies never enter the Jacobian. It is one scalar per block, architecture-agnostic, adds negligible compute, and needs no knowledge of depth or presence of normalization.

A natural worry is that multiplying the branch by zero freezes learning. At alpha = 0 the branch weights receive no gradient on the first step, because the gradient to W is proportional to alpha. But the scalar alpha itself receives gradient F(x_i), which is nonzero since the random branch evaluated on the data produces a nonzero vector and the upstream gradient is well-behaved precisely because the rest of the network is a clean identity. So alpha moves off zero immediately, and from the next step the branch weights begin receiving gradients proportional to alpha. The network therefore dynamically grows its own depth: it starts as the identity and turns on each block only as fast as optimization finds useful.

Why zero rather than 1 or 1/sqrt(d)? Starting at alpha = 1 places the block on the same ill-conditioned ridge as a vanilla residual block, where gradients explode or vanish exponentially with depth and admissible learning rates shrink as 2^{-L}. Starting at 1/sqrt(d) tames forward variance but is only approximately identity, requires knowing the total depth d, and still leaves a small branch contribution. Zero gives exactly identity, no depth dependence, and exact dynamical isometry, while alpha still leaves zero on the first step. The skip path remains a pure identity, following the pre-activation lesson that the cleanest gradient flow comes from an unmodulated shortcut; the scalar modulates only the branch.

The same idea drops into Transformers without special handling. Replacing LayerNorm and gating each layer with one shared zero-initialized scalar across the attention and feed-forward sublayers gives x_{i+1} = x_i + alpha_i sublayer(x_i). At init the sublayers are off, so J = I despite the LayerNorm and softmax pathologies. Under aggressive one-cycle schedules the linearity of alpha near init can destabilize training, so in that regime the residual weights should use a small constant learning rate while the rest of the network follows the full schedule; ordinary schedules need no special treatment.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """ReZero residual block: out = shortcut(x) + alpha * F(x), alpha init 0."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        self.resweight = nn.Parameter(torch.zeros(1))  # alpha, init 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # F(x)
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + self.resweight * out  # x + alpha * F(x)
        return out


class RZTXEncoderLayer(nn.Module):
    """ReZero Transformer layer: x_{i+1} = x_i + alpha * sublayer(x_i)."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.resweight = nn.Parameter(torch.zeros(1))  # shared alpha, init 0

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2 * self.resweight)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2 * self.resweight)
        return src
```
