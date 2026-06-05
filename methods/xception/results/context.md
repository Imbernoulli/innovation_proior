# Research question

Convolutional networks have become the master algorithm for image recognition, and the field has converged on a recipe: stack convolutional layers, go deeper, and accuracy climbs. But a single convolution layer is doing something conceptually overloaded. Its kernel is a 3-D object — two spatial dimensions plus a channel dimension — and it is asked, with one set of weights, to *simultaneously* discover where things are in space (spatial correlations: edges, textures, shapes) and how the input channels should be recombined into new features (cross-channel correlations). These are arguably two different jobs that happen to be fused into one operator.

The Inception family has already shown empirically that *deliberately decoupling* these two jobs — first mixing channels with cheap 1×1 convolutions, then doing spatial convolutions inside the resulting lower-dimensional spaces — yields richer representations at lower parameter cost than a plain stack of full convolutions. The question this raises, and the one a solution must answer, is: how far can that decoupling be pushed? Inception splits the channels into a handful of segments (3 or 4) and convolves each segment spatially. Is there a principled extreme of this idea, and if cross-channel and spatial correlations can be mapped *completely* separately, does a network built entirely on that hypothesis match or beat Inception at equal parameter count? A solution would have to be at least as accurate as the best Inception model of the day, use no more parameters, and be dramatically simpler to define and modify than the hand-tuned, multi-branch Inception modules.

# Background

**Convolution as a joint spatial-and-channel filter.** A standard convolutional layer with a `k×k×M×N` kernel maps an `M`-channel input to an `N`-channel output. Each output channel is formed by convolving every input channel with its own `k×k` spatial filter and summing across input channels. So one layer's weights encode, jointly and inseparably, both the spatial filtering pattern and the cross-channel recombination. The cost is `k²·M·N·H·W` multiply-adds.

**The Network-in-Network insight (Lin et al., 2013) and 1×1 convolutions.** A 1×1 convolution has no spatial extent; at each pixel it is just a matrix-vector product mapping the `M`-vector of channels to an `N`-vector. It performs *pure cross-channel mixing* with zero spatial mixing. This is the cheap primitive that later architectures use to reshape channel dimensions.

**The Inception module and its hypothesis (Szegedy et al., 2014–2015; GoogLeNet, Inception V2/V3).** The canonical Inception module first applies a set of 1×1 convolutions to map the input into several separate, smaller channel spaces, then runs ordinary 3×3 (and 5×5) convolutions inside each of those spaces, and concatenates. The stated hypothesis: cross-channel correlations and spatial correlations are sufficiently *decoupled* that it is preferable not to map them jointly. Inception V3 refines this with factored spatial convolutions (e.g. 7×1 followed by 1×7) and batch normalization. Empirically Inception modules learn richer features with fewer parameters than VGG-style stacks of plain convolutions. The gap each Inception module leaves open: it only *partially* decouples — it commits to a small, hand-chosen number of channel segments (3 or 4), and the modules are intricate, with many branches and ad-hoc design choices that make the architecture hard to define and modify.

**A simplified Inception module is a 1×1 conv followed by grouped spatial convs.** Take an Inception module that uses a single spatial kernel size and no pooling branch. It is strictly equivalent to: one large 1×1 convolution over all input channels, followed by spatial convolutions that each operate on a *non-overlapping segment* of that 1×1 output's channels. This reformulation exposes a free parameter that was implicit all along: the number of segments the output channels are partitioned into. A regular convolution (preceded by a 1×1) corresponds to a *single* segment spanning all channels; Inception sits at 3–4 segments. The whole range between these is a continuum that had not been explored.

**Depthwise separable convolution (Sifre, 2013–2014; in TensorFlow/Keras since 2016).** Independently, an operation existed that factors a convolution into a *depthwise convolution* — one `k×k` spatial filter applied independently to each input channel, with no cross-channel mixing — followed by a *pointwise* 1×1 convolution that recombines channels. Its cost is `k²·M·H·W + M·N·H·W`, the sum of the two stages rather than the product, a reduction of roughly `k²` over a full convolution. In its usual framework implementation, the depthwise step comes first and there is no nonlinearity between the depthwise and pointwise operations. This had been used for efficiency (small accuracy gains, large convergence-speed and model-size gains) but not as the organizing principle of a full high-accuracy architecture.

**Residual connections (He et al., 2015).** Adding the input of a block to its output (`y = F(x) + x`) lets very deep networks train, dramatically improving convergence speed and final accuracy. By this point residuals are a standard, near-essential ingredient for deep stacks.

**Batch normalization (Ioffe & Szegedy, 2015).** Normalizing each channel's pre-activations by minibatch statistics, with a learned scale and shift, stabilizes and accelerates training of deep convolutional stacks; it is applied after essentially every convolution in the architectures of the time.

# Baselines

**VGG-16 (Simonyan & Zisserman, 2014).** A uniform stack of 3×3 convolutions and max-pools. Schematically simple — easy to define as a linear stack — but parameter-heavy and less accurate than Inception. Establishes that simple linear stacks of one repeated motif are attractive from an engineering standpoint.

**Inception V3 (Szegedy et al., 2015).** The strongest practical model of its scale: ~23.6M parameters, state-of-the-art ImageNet accuracy. Built from partially-decoupled Inception modules with several spatial kernel sizes, factored convolutions, an auxiliary classification tower, and batch normalization. Its limitation as a *design*: the modules are complex and hand-tuned, the decoupling is only partial, and it was carefully optimized for ImageNet (potentially over-fit to that benchmark). It is the natural yardstick because a new architecture can be built at the *same* parameter count, so any accuracy difference reflects a more (or less) efficient use of parameters rather than added capacity.

**ResNet-50/101/152 (He et al., 2015).** Deep residual stacks; strong ImageNet baselines at comparable or larger scale. Included as reference points for classification accuracy.

# Evaluation settings

- **ImageNet (ILSVRC 2012)**: 1000-class single-label classification, ~1.3M training images. Standard top-1 and top-5 accuracy on the validation set, single crop, single model. Input 299×299. The natural primary benchmark; Inception V3 was designed for it.
- **A large-scale internal dataset (JFT)**: ~350M images, 17,000 classes, multi-label. Evaluated via an auxiliary held-out set with Mean Average Precision over the top-100 predictions (MAP@100), with class contributions weighted by how common the label is. This larger, less-tuned regime probes whether gains hold beyond a single carefully-tuned benchmark.
- **Optimization protocols available**: SGD with momentum 0.9 (ImageNet: initial LR 0.045, decayed 0.94 every 2 epochs) and RMSprop (JFT: momentum 0.9, initial LR 0.001, decayed 0.9 every 3M samples). Polyak (exponential) averaging of weights at inference. Regularization: L2 weight decay, dropout before the classifier, optionally an auxiliary loss tower.
- **Size/speed metric**: parameter count and training steps/second on a fixed GPU cluster, used to confirm that two architectures are matched in capacity.

# Code framework

The primitives below already exist in a high-level deep-learning library (Conv2D, a depthwise-separable `SeparableConv2D` that does a per-channel spatial conv then a 1×1 conv, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling, and additive residual merges). What does not yet exist is the architecture that organizes them. The scaffold lays out the empty slot.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Already-available primitives:
#   layers.Conv2D(filters, k, strides, padding, use_bias)
#   layers.SeparableConv2D(filters, k, padding, use_bias)  # per-channel spatial conv, then 1x1
#   layers.BatchNormalization()
#   layers.ReLU()
#   layers.MaxPooling2D(pool, strides, padding)
#   layers.GlobalAveragePooling2D()
#   layers.add([...])   # elementwise residual merge


def conv_bn(x, filters, k, strides=1, padding="same"):
    # Standard primitive: convolution -> batch-norm. ReLU added by caller where wanted.
    x = layers.Conv2D(filters, k, strides=strides, padding=padding, use_bias=False)(x)
    return layers.BatchNormalization()(x)


def feature_block(x, filters, stride):
    # TODO: one repeating feature-extraction unit of the network.
    # The whole contribution lives here: what operation does the spatial+channel
    # work, whether a nonlinearity sits inside it, and how the residual is wired.
    raise NotImplementedError


def build_network(input_shape=(299, 299, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)
    # TODO: stem.
    x = inputs
    # TODO: a stack of feature_block(...) units forming entry / middle / exit stages,
    #       with residual connections around them.
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)  # logistic regression head
    return Model(inputs, outputs)
```
