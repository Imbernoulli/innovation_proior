The task is to make an unconditional image denoiser aware of a class label, using only the two sockets the scaffold exposes: a function that combines the timestep and class embeddings before they reach the residual blocks, and an optional hook that can touch hidden states between blocks. The challenge is not deciding whether to condition, but choosing the cheapest operator that is still strong enough. A class label is a single global scalar per example, so the natural first guess is a per-channel affine such as FiLM or adaptive group norm. Those operators are small and well matched to a global signal. The simpler version, however, is the additive-bias corner of that family. If I concatenate a conditioning vector to a feature map and follow it with a convolution, the linear map splits into a feature term and a conditioning term; the conditioning term is just a feature-dependent bias. That gives shift but not scale. Feature-wise gating gives scale but only attenuation, and with no additive offset. The full affine contains both, but for a single scalar label the question is whether the extra multiplicative knob is necessary. Earlier results showed that routing an adaptive norm only through a post-block conditioner left the timestep's own block-internal AdaGN path blind to the class, which produced a visible bottleneck. The hypothesis is that the affine operator was correct but the road was wrong.

The method is Concat-FiLM. It treats the class label as an additive side signal on the same path the timestep already travels. The class label is embedded to the same width as the timestep embedding, added elementwise to it, and the sum is fed into the residual blocks. Because the block's adaptive group norm projection is linear, the combined embedding produces scale and shift parameters that are the timestep contribution plus a class-dependent offset. In other words, Concat-FiLM realizes the gamma-equals-one, additive-bias corner of FiLM at the block's native modulation point. The class never needs its own separate conditioner; the existing time-conditional machinery absorbs it. This keeps the parameter count tiny, the initialization identical to the tuned unconditional backbone, and the computation exactly as fast as the original model.

Why this works cleanly in the scaffold is worth spelling out. The residual blocks already contain an AdaGN layer that regresses per-channel scale and shift from the side embedding. By making the side embedding the sum of time and class information, the class enters every block that the timestep enters, at the same depth and with the same channel-wise resolution. There is no new sublayer to initialize, no gate to learn from zero, and no risk that a badly initialized conditioner will suppress the backbone. If the class embedding starts near zero, the model is essentially the unconditional denoiser at initialization, and the class contribution grows during training. The optional hidden-state hook is left as a no-op, because all conditioning is handled through the time path.

The broader picture is that Concat-FiLM sits between the cheaper but weaker post-block adaptive norm and the richer but heavier cross-attention operator. It is not content-dependent or spatially varying; like any FiLM variant, it applies the same per-channel modulation across all spatial positions. But because it rides the timestep's existing AdaGN, it reaches deeper into the network than a post-block conditioner and does so with no new layers. For a single scalar class label, that is usually the right trade-off.

```python
import torch.nn as nn


def prepare_conditioning(time_emb, class_emb):
    return time_emb + class_emb


class ClassConditioner(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()

    def forward(self, h, class_emb):
        return h
```
