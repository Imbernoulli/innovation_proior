**Problem (from step 5).** Progressive resizing bought speed but cost some accuracy; need to refill the
headroom, and there's now throughput slack to pay for it. A plain ResNet conv passes every output channel
forward with a fixed, input-independent role — it has no direct, per-image way to emphasize the channels
that matter for *this* image and damp the rest.

**Key idea — Squeeze-and-Excitation.** Add a channel-wise attention module after convs: (1) *squeeze* —
global-average-pool the C-channel feature map to a per-channel descriptor (one mean per channel); (2)
*excite* — a single-hidden-layer MLP (Linear C→C/r, ReLU, Linear C/r→C, Sigmoid) maps that descriptor to a
gate in (0,1) per channel, the bottleneck forcing a compact model of channel interactions; (3) *scale* —
multiply each channel of the feature map by its gate. Insert architecture-agnostically after convs whose
output has ≥ `min_channels` channels (512 for ImageNet-scale), putting SE in the deep, high-channel,
small-spatial stages where it's cheapest and most useful.

**Why it works.** The conv says *what* features exist; SE adds *which* features matter for this input — a
data-dependent channel reweighting that sharpens capacity use and consistently raises CNN accuracy. The cost
is throughput (training and inference, ~20% inference latency depending on hardware), but it's the right move
here: the banked throughput slack absorbs it, and because SE *slows the model*, it gives the data loader more
time during the progressive-resizing small-image phase — reducing the relative CPU/data-pipeline load that
phase created. Suggested ResNet-50 config: `min_channels=512`, `latent_channels=64`.

**Change / code.** The `SqueezeExcite2d` module — pool → MLP → sigmoid → channel-wise scale.

```python
class SqueezeExcite2d(torch.nn.Module):
    def __init__(self, num_features: int, latent_channels: float = .125):
        super().__init__()
        self.latent_channels = int(latent_channels if latent_channels >= 1
                                   else latent_channels * num_features)
        flattened_dims = num_features
        self.pool_and_mlp = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),                                  # squeeze
            torch.nn.Flatten(),
            torch.nn.Linear(flattened_dims, self.latent_channels, bias=False),
            torch.nn.ReLU(),                                                # excite
            torch.nn.Linear(self.latent_channels, num_features, bias=False),
            torch.nn.Sigmoid(),                                            # gates in (0,1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = input.shape
        attention_coeffs = self.pool_and_mlp(input)
        return input * attention_coeffs.reshape(n, c, 1, 1)                # scale
```
