# Xception

**Problem.** A standard convolution's kernel does two jobs at once with one set of weights: spatial filtering (within-channel) and cross-channel recombination. Inception modules improved on plain convolution stacks by *partially* decoupling these — a 1×1 convolution mixes channels, then ordinary spatial convolutions act on a few non-overlapping segments of those channels. Xception asks whether decoupling can be taken to the limit, and whether the resulting architecture beats Inception at equal parameter count.

**Key idea.** A simplified Inception module is strictly equivalent to one large 1×1 convolution followed by spatial convolutions on non-overlapping segments of its output channels. This exposes a continuum indexed by the number of segments: one segment is an ordinary convolution; a few segments is Inception. The extreme — *one segment per channel* — is a 1×1 channel mix followed by an independent spatial filter per channel, i.e. a **depthwise separable convolution**. Xception ("Extreme Inception") makes the strong hypothesis that cross-channel and spatial correlations can be mapped *completely* separately, and builds the entire network as a linear stack of depthwise separable convolutions with residual connections.

**Architecture (final form).**
- **Depthwise separable conv** = depthwise (one `k×k` spatial filter per channel) then pointwise (1×1 channel mix). **No nonlinearity between the two stages**: the per-channel spatial space is one-channel-deep, so an intermediate ReLU destroys information rather than adding useful expressiveness (the opposite of what holds inside deep Inception modules). Cost `k²·M·H·W + M·N·H·W` vs. a full conv's `k²·M·N·H·W`.
- **36 convolutional layers**, grouped into 14 modules, all but the first and last wrapped in linear residual connections.
- **Entry flow**: stem of two *full* 3×3 convs (3→32 stride 2, then 32→64) — full convs because the 3-channel input has nothing to separate yet — followed by three residual modules stepping width 128 → 256 → 728, each ending in a 3×3/stride-2 max-pool with a strided 1×1 projection shortcut.
- **Middle flow**: the same 728-channel residual module repeated **8 times**, each three (ReLU → SeparableConv 728 → BN), with identity shortcuts.
- **Exit flow**: one residual module 728 → 1024 (max-pool, projection shortcut), then two final separable convs widening to 1536 then 2048 (each SeparableConv → BN → ReLU), global average pooling, and a logistic-regression head.
- BatchNorm after every conv and separable conv; ReLU placed *before* each separable conv (pre-activation). Residuals are empirically essential for this stack's convergence speed and final accuracy.
- Sized to match Inception V3 (~22.9M vs ~23.6M parameters), so accuracy differences reflect parameter efficiency, not capacity.

**Training.** SGD momentum 0.9 on ImageNet (LR 0.045, decay 0.94 every 2 epochs); RMSprop on very large data. Weight decay `1e-5` (lighter than Inception V3's `4e-5`), dropout 0.5 before the head on ImageNet, no auxiliary tower, Polyak-averaged weights at inference. Input 299×299.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_bn(x, filters, k, strides=1):
    # Full conv, used only in the stem (3-channel input: nothing to separate yet).
    x = layers.Conv2D(filters, k, strides=strides, padding="same", use_bias=False)(x)
    return layers.BatchNormalization()(x)


def sep_bn(x, filters):
    # Depthwise separable conv = per-channel 3x3 spatial filter, then 1x1 channel mix.
    # No nonlinearity inside: the intermediate space is one-channel-deep.
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    return layers.BatchNormalization()(x)


def residual_module(x, filters, do_first_relu=True, pool=True):
    shortcut = x
    y = x
    if do_first_relu:
        y = layers.ReLU()(y)
    y = sep_bn(y, filters)
    y = layers.ReLU()(y)
    y = sep_bn(y, filters)
    if pool:
        y = layers.MaxPooling2D(3, strides=2, padding="same")(y)
        shortcut = layers.Conv2D(filters, 1, strides=2, padding="same",
                                 use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.add([y, shortcut])


def Xception(input_shape=(299, 299, 3), num_classes=1000):
    inp = layers.Input(shape=input_shape)

    # Entry flow
    x = layers.ReLU()(conv_bn(inp, 32, 3, strides=2))
    x = layers.ReLU()(conv_bn(x, 64, 3))
    x = residual_module(x, 128, do_first_relu=False)
    x = residual_module(x, 256)
    x = residual_module(x, 728)

    # Middle flow: same module x8, identity shortcuts
    for _ in range(8):
        shortcut = x
        y = x
        for _ in range(3):
            y = layers.ReLU()(y)
            y = sep_bn(y, 728)
        x = layers.add([y, shortcut])

    # Exit flow
    x = residual_module(x, 1024)
    x = layers.ReLU()(sep_bn(x, 1536))
    x = layers.ReLU()(sep_bn(x, 2048))

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes)(x)
    return Model(inp, out)
```
