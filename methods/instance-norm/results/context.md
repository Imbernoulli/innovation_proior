## Research question

A feed-forward generator network is trained to apply a fixed artistic style to any content image in a single pass, by minimizing a perceptual loss that matches the output's style statistics to a style image and its content statistics to the input. These networks are fast, but their generated images are visibly worse than the slow optimization-based stylization they are meant to replace, and they show two stubborn pathologies: training on *more* content images makes the qualitative results *worse* (a generator trained on a handful of images beats one trained on thousands, and the best results need early stopping), and the heaviest artifacts appear along the image border, where zero-padding is added before every convolution, resisting fancier padding schemes. The objective behaves as though it were too hard for a standard convolutional architecture to learn. The question is *why* it is hard, and whether the difficulty can be removed by a change to the architecture's building blocks rather than to the training data or schedule.

## Background

**Optimization-based style transfer.** Gatys et al. (2016) stylize an image by iterative optimization: starting from noise, an image is optimized so its deep-CNN feature statistics simultaneously match a content image (deeper layers, preserving spatial structure) and a style image (shallower layers, spatially averaged into Gram matrices that capture texture). The results are excellent but cost minutes per 512×512 image, because every stylization is its own optimization.

**Feed-forward generators.** To make this real-time, Texture Networks (Ulyanov et al., 2016) and Johnson et al. (2016) instead *learn* a generator g(x, z) that maps a content image x (with a noise seed z) to the stylized output in one forward pass, trained on many content images to minimize the same Gatys-style perceptual loss, min_g (1/n) Σ_t L(x_0, x_t, g(x_t, z_t)). These generators are built from convolution, pooling, upsampling, and batch normalization. They are fast but qualitatively inferior to Gatys, and exhibit the more-data-hurts and border-artifact pathologies above.

**A property of stylization: contrast comes from the style.** The style loss is designed so the stylized image's contrast tracks the *style* image's contrast and is essentially *independent* of the content image's contrast — a high-contrast and a low-contrast version of the same content stylize to nearly the same output. This means a correct generator must *discard* the content image's contrast information. The open question is whether such a normalization of content contrast is something a stack of conv/ReLU/pool can easily represent and learn from data, or whether the difficulty of expressing it cleanly is what the generator stalls on.

**Batch normalization.** Batch normalization (Ioffe & Szegedy, 2015), already present in these generators, normalizes each feature channel using the mean and variance pooled over the *whole minibatch* (all images and all spatial positions of that channel): for x ∈ R^{T×C×W×H} with t the image index, i the channel, j,k spatial,

  y_tijk = (x_tijk − μ_i) / √(σ_i² + ε),
  μ_i = (1/(HWT)) Σ_t Σ_l Σ_m x_tilm,
  σ_i² = (1/(HWT)) Σ_t Σ_l Σ_m (x_tilm − μ_i)².

At training time the statistics are the batch statistics; at test time they are replaced by fixed running (population) estimates accumulated during training, and the layer is often folded away. The statistics thus pool over all images and all spatial positions of each channel, and differ between training and test.

## Baselines

**Gatys optimization (Gatys et al., 2016).** Per-image gradient optimization to match style Gram matrices and content features. Core idea and math as above. Gap: minutes per image — not real-time.

**Texture Networks generator (Ulyanov et al., 2016).** A feed-forward multi-scale generator trained to reproduce Gatys' result in one pass, using conv/pool/upsample/batch-norm. Core idea: amortize the optimization into a learned network. Gap: qualitatively worse than Gatys; more training data degrades quality; severe border artifacts from zero padding.

**Johnson generator (Johnson et al., 2016).** A feed-forward generator with a residual architecture trained on the same perceptual loss. Core idea: as Texture Nets, with residual blocks; somewhat more efficient. Gap: the same shortcomings — inferior to Gatys, the same artifacts.

**Batch normalization (Ioffe & Szegedy, 2015).** Normalize each channel by minibatch statistics; affine rescale; running statistics at test. Core idea and math as above. Gap *for this setting*: its statistics are pooled across the whole batch and frozen to population values at test time; with this layer in place the generator still struggles on the stylization objective, the more-data-hurts and border pathologies persist.

## Evaluation settings

The yardstick is feed-forward artistic style transfer. The natural setup: a generator (the Texture Networks multi-scale architecture or the Johnson residual architecture) is trained per fixed style image on a corpus of content images, minimizing a perceptual loss computed through a pretrained image-classification CNN (VGG) — a Gram-matrix style loss from shallow layers plus a feature-reconstruction content loss from deep layers. Quality is judged qualitatively against the Gatys optimization result (the gold standard) and against the same generator using batch normalization, at resolutions such as 256, 512, and 1080, with attention to border artifacts and to how quality changes with the amount of training data.

## Code framework

The available ingredients are a feed-forward generator built from convolution, pooling, upsampling, a per-channel normalization layer, and ReLU, trained with a VGG-based perceptual loss. The open slot is the *normalization layer* dropped between convolutions.

```python
import torch
import torch.nn as nn


class Normalization(nn.Module):
    """Per-channel normalization layer inserted between generator convolutions.
    Input/expected shape: (N, C, H, W)."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))   # gamma
            self.bias = nn.Parameter(torch.zeros(num_features))    # beta

    def forward(self, x):
        # TODO: define how this layer normalizes the activation tensor.
        pass


def build_generator():
    # conv -> Normalization -> ReLU -> ... (Texture-Nets or residual generator)
    # TODO: stack blocks using the Normalization layer above.
    pass

# trained per fixed style image with a pretrained-VGG perceptual loss
# (Gram-matrix style loss + deep-feature content loss), Adam, standard loop.
```
