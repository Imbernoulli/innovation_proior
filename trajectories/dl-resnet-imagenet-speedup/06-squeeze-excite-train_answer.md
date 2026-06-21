Progressive resizing made each step much cheaper but cost a little accuracy and ate into the headroom I had banked — and it left me with throughput slack, since the early steps now run on small images. So I need to refill the accuracy tank, and this time I am willing to pay a bit of throughput for it. The question is where in the network there is accuracy to be had. Look at how a ResNet convolution treats its output channels: after a conv produces a tensor with $C$ channels, every channel is passed forward with equal weight. The network has no explicit, input-dependent way to say "for *this* image, the channels detecting fur texture matter and the channels detecting sky do not." Each channel is a feature detector whose relevance depends on the image, yet a plain conv stack gives every channel a fixed role determined only by its learned weights, not modulated by what is actually in front of it. The only way the network can emphasize a channel for a particular input is indirectly, through the spatial convolution mixing — it cannot directly turn a channel up or down based on a global read of the image. That is a missing capability: per-image, per-channel reweighting. The conv tells you *what* features exist at each location; a channel-reweighting mechanism would tell you *which of those features matter* for this image.

The method I propose is **Squeeze-and-Excitation**, a channel-wise attention module inserted after a convolution's output, built from three steps. The first is the *squeeze*. The reweighting decision should depend on each channel's activity across the entire spatial extent, not any one location — "is fur-texture present in this image" is a whole-image question — so I summarize each channel by its mean over all spatial positions. Global average pooling collapses the $C$-channel feature map to a single $C$-vector, one number per channel: the channel's mean activation. The second is the *excite*: from that $C$-vector of channel summaries, compute a set of per-channel gates in $(0,1)$. This is a function from the channel descriptor to channel weights, and it should model interactions between channels (channel A being active might mean channel B is redundant), so a small fully-connected network does it: project the $C$-vector down to a smaller latent dimension, apply a ReLU, project back up to $C$, and squash through a sigmoid so each output is a gate in $(0,1)$. The down-projection is deliberate — forcing the gating through a bottleneck makes it learn a compact, low-dimensional description of channel interactions rather than a free $C\times C$ reweighting, keeping added parameters and overfitting risk down. The third is the *scale*: multiply each channel of the original feature map by its gate, so channels the gate deems irrelevant for this image are scaled toward zero and relevant ones pass through. The whole module is

$$\text{global-avg-pool} \;\to\; \text{Linear}(C \to C/r) \;\to\; \text{ReLU} \;\to\; \text{Linear}(C/r \to C) \;\to\; \text{Sigmoid} \;\to\; \text{channel-wise multiply},$$

a data-dependent channel-wise scaling — channel attention.

Where these go in ResNet-50 is itself a design choice. Rather than hand-place them at named points, I want a rule, and the rule that matters is *which* convs get an SE module, because SE adds compute and I want it where it is cheapest and most useful. The overhead is dominated by the channel MLP, and it is cheapest to attach where the spatial map is *small* — late in the network, where the channel count is high (so attention has many channels to choose among) and the spatial size is tiny (so the global pool and the multiply are cheap). So I gate on output-channel count: add SE only after convs whose output has at least `min_channels` channels — $512$ for ImageNet-scale models. That naturally places SE in the deep, high-channel, small-spatial stages and skips the early high-resolution layers where it would be expensive and the channel count is low anyway. The latent dimension is set so the bottleneck holds at least $\sim 64$ channels (`latent_channels=64`), enough to model channel interactions without blowing up parameters.

The cost, honestly: SE adds a pool, two small matmuls, and a multiply to many blocks, slowing both training and inference — the inference latency hit can be on the order of $\sim 20\%$ depending on architecture and hardware, since those ops run every forward pass forever. But two things make it the right move here. One, I have throughput slack from progressive resizing to absorb the training slowdown. Two — the nice composition — SE *slows the model down*, and during the progressive-resizing small-image phase, when the GPU was getting starved and the dataloader was the bottleneck, a slightly heavier model gives the data loader more time to keep up. A method that decreases GPU throughput reduces the *relative* load on the CPU/data pipeline, letting the GPU-bound and CPU-bound parts balance better. So SE does not just buy accuracy; it partially masks its own cost by un-starving the GPU in exactly the regime the previous rung created. I expect a consistent accuracy improvement — SE reliably raises CNN accuracy both in absolute terms and when controlling for training/inference time — refilling the headroom progressive resizing spent, via a mechanism orthogonal to everything in the stack so far. The core is the module's squeeze-excite-scale forward.

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
