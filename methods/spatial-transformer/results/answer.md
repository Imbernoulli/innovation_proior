# Spatial Transformer Networks

## Problem

A CNN's only built-in spatial-invariance mechanism is local max-pooling — a fixed, hand-wired window with tiny (e.g. 2×2) support. Real invariance to large translations, scale, rotation, and warps only accumulates slowly across a deep hierarchy, and intermediate feature maps are not invariant to large input transformations. Data augmentation only teaches *tolerance* to nuisance pose, never *normalisation*. We want a learnable module that actively warps a feature map into a canonical pose, conditioned on the input, trained end-to-end from the task loss alone — no transformation supervision — and droppable into any architecture at any depth.

## Key idea

A **Spatial Transformer** is a self-contained, differentiable module with three parts:

1. **Localisation network** — `θ = f_loc(U)`, any net (FC or conv) ending in a regression layer, mapping the input feature map `U ∈ R^{H×W×C}` to transformation parameters `θ` (6-D for affine).
2. **Grid generator** — defines, for each pixel of the *output* grid, the *source* coordinate to read from the input. The transform is applied **output→input** so every output pixel maps to exactly one source point (no holes, no collisions — the resampling is well-defined, as in graphics texture mapping).
3. **Differentiable sampler** — reads the input at those (fractional) source coordinates with a **bilinear** kernel, whose sub-gradients let the loss flow back to the input map *and* to `θ`, hence into the localisation network.

Because the whole chain is differentiable, the module learns *what* transformation to apply purely from the downstream loss.

## The method

**Affine grid (output→input, normalised coordinates `−1 ≤ · ≤ 1`).** For output grid point `(x_t^i, y_t^i)`,

```
( x_s^i )         ( x_t^i )
(       ) = A_θ · ( y_t^i ) ,    A_θ = [ θ11 θ12 θ13 ]
( y_s^i )         (   1   )           [ θ21 θ22 θ23 ]
```

i.e. `x_s = θ11·x_t + θ12·y_t + θ13`, `y_s = θ21·x_t + θ22·y_t + θ23`. Normalised coordinates make the identity `A_θ=[[1,0,0],[0,1,0]]` resolution-independent; a contraction of the left 2×2 block (|det|<1) produces a crop/zoom. Constrained variants: attention `A_θ=[[s,0,t_x],[0,s,t_y]]` (3 params); general `T_θ = M_θ B` with learnable target grid `B`, covering projective, piecewise-affine, and thin-plate-spline transforms. Any parameterisation works provided `(x_s,y_s)` is differentiable in `θ`.

**Bilinear sampler.** Same warp applied to every channel:

```
V_i^c = Σ_n Σ_m  U_{nm}^c · max(0, 1 − |x_s^i − m|) · max(0, 1 − |y_s^i − n|)
```

**Sub-gradients** (only the ≤4 pixels in the kernel support contribute; sub-gradients at the `|·|` kinks):

```
∂V_i^c/∂U_{nm}^c = max(0, 1 − |x_s^i − m|) · max(0, 1 − |y_s^i − n|)

∂V_i^c/∂x_s^i = Σ_n Σ_m U_{nm}^c · max(0, 1 − |y_s^i − n|) · {  0  if |m − x_s^i| ≥ 1
                                                              +1  if m ≥ x_s^i
                                                              −1  if m < x_s^i }
```

(`∂V/∂y_s` symmetric; the integer/nearest-neighbour kernel is rejected because its gradient w.r.t. `x_s` is zero almost everywhere.) Finally `∂x_s/∂θ` is immediate from the affine (`∂x_s/∂θ11 = x_t`, etc.), so `∂L/∂θ` chains through and the localisation network trains by ordinary backprop. The 3-D extension simply adds a `max(0,1−|z_s−l|)` factor and a 3×4 affine.

**Practicalities.** Initialise the regression layer to the identity transform (weights 0, bias `[1,0,0,0,1,0]`) so the module starts as a no-op and the host net trains like a normal CNN, deviating only as it helps. Use a lower learning rate for the localisation network (≈1/10, or far less on a large pretrained backbone) since `θ` is high-leverage. The module is cheap (~few % overhead); place one at the input to pose-normalise, several at depth to warp abstract features, or several in parallel to attend to multiple objects/parts; a smaller output grid crops-and-downsamples in one step (saving compute, with mild aliasing for large downsampling).

## Code

Grounded in the standard PyTorch implementation: `F.affine_grid` builds the normalised output grid and applies `A_θ` output→input; `F.grid_sample` does the bilinear read with exactly the sub-gradients above.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # recognition network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # --- Spatial Transformer ---
        # localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # regress the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )
        # initialise to the identity transform (start as a no-op)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)     # A_theta per sample
        grid = F.affine_grid(theta, x.size())      # grid generator (output->input)
        x = F.grid_sample(x, grid)                 # bilinear sampler
        return x

    def forward(self, x):
        x = self.stn(x)                            # warp into canonical pose
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

Trained with ordinary SGD on the task loss only:

```python
def train(model, loader, optimizer):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()        # gradient reaches f_loc via the sampler sub-gradients
        optimizer.step()
```
