# Context

## Research question

We can now train generative adversarial networks that produce convincing human faces at 1024×1024. The generator takes a latent vector `z` sampled from a fixed prior, pushes it through a stack of convolutions, and an image comes out. The quality is high, but the internal structure of the generator — how it maps `z` to image space, how different scales of structure are handled, and how the latent space relates to perceptible attributes — is not well understood or controllable.

The open question: how should a GAN generator be designed so that different scales of image structure can be controlled independently, and so that the latent representation is more interpretable? And, since existing disentanglement metrics require an encoder or known ground-truth factors, what encoder-free quantitative measures of latent-space structure are possible?

## Background

**Generative adversarial networks** (Goodfellow et al., 2014). A generator `G: z → x` is trained against a discriminator `D` that scores real versus generated images. In practice the generator is trained with the non-saturating loss `softplus(-D(G(z)))` rather than the original saturating one. Everything here is downstream of this setup, and none of it needs to change.

**The state of high-resolution GANs.** The prevailing recipe for megapixel synthesis is *progressive growing* (Karras et al., 2017): start training at 4×4 and add resolution blocks one at a time up to 1024×1024, fading each new block in smoothly. Training low resolutions first lets the network nail large-scale structure before worrying about fine detail, which is what makes high resolution stable at all. That work also contributed several tricks that are now standard: an *equalized learning rate* (initialize weights from N(0,1) and multiply them by the He constant `c = gain/√(fan_in)` at runtime, so that the per-parameter dynamic range is uniform and an adaptive optimizer like Adam advances every weight at the same effective rate), *pixelwise feature normalization* (rescale each pixel's feature vector to unit length after each convolution, to stop the generator and discriminator from driving activation magnitudes into an arms race), and a *minibatch standard-deviation* layer near the end of the discriminator (append a feature map holding the average per-pixel feature standard deviation across the minibatch, which fights mode collapse by making the discriminator aware of intra-batch variation). Parallel lines of work improve the discriminator (multiple discriminators, multi-resolution discrimination, self-attention) or the loss/regularizer (WGAN-GP, spectral normalization, R1). The R1 regularizer (Mescheder et al., 2018) penalizes the discriminator's gradient on real samples, `(γ/2)·E[‖∇_x D(x)‖²]`, and is notable in that it lets FID keep improving for far longer than WGAN-GP does.

**Style transfer and instance normalization.** A separate body of work, on neural style transfer, concerns separating what an image depicts from how it looks. Gatys et al. (2016) established that the *style* of an image is captured by the statistics of deep convolutional feature maps — the Gram matrix, and equivalently (Li et al., 2017, "Demystifying neural style transfer") the per-channel mean and variance — while spatially varying features carry content. *Instance normalization* normalizes each sample and each channel over its spatial extent, `IN(x) = γ·(x − μ(x))/σ(x) + β`, with `μ_nc, σ_nc` taken over the H×W of channel `c` of sample `n`. Instance norm dramatically outperforms batch norm for feedforward style transfer, and there is a clean reason why: an experiment training matched IN and BN models on original images, on contrast-normalized images, and on *style-normalized* images shows that IN's advantage survives contrast normalization but largely disappears once the images are already style-normalized. The conclusion is that **instance normalization is itself performing a kind of style normalization** — it strips an instance's style (carried in the channel-wise mean/variance) so the rest of the network can focus on content.

*Conditional instance normalization* (Dumoulin et al., 2016) exploited this: a single network handles `S` styles by learning a separate affine pair `(γ^s, β^s)` per style, `CIN(x; s) = γ^s·(x − μ(x))/σ(x) + β^s`. Swapping only those affine parameters — same convolution weights — produces completely different output styles, which is strong evidence that **the affine parameters of an instance-norm layer can by themselves dictate the output style**.

*Adaptive instance normalization*, AdaIN (Huang & Belongie, 2017), removes the learned affine entirely and instead computes the affine parameters on the fly from a style input `y`'s own feature statistics:

```
AdaIN(x, y) = σ(y) · (x − μ(x)) / σ(x) + μ(y)
```

i.e. normalize the content `x` to zero mean and unit variance per channel, then re-scale and re-bias to match the style `y`'s per-channel `(σ(y), μ(y))`. It has no learnable parameters, works for arbitrary styles, and is essentially free. The intuition: a channel that fires on a particular feature has a mean activation that says "how strongly this feature is present" — exactly a style knob — and its variance carries finer style information. In the original setting both `x` and `y` are feature maps of real images.

**Why a fixed latent prior forces entanglement.** A common notion of disentanglement is a latent space made of linear subspaces, each controlling one factor of variation. There is a structural obstacle to achieving this with the *input* latent space of a GAN. The input latent `z` is sampled from a fixed prior (a round Gaussian), and the generator must reproduce the training data's density — the probability of each combination of factors in latent space has to match its frequency in the data. If the data manifold has holes (some combinations of attributes simply never occur), then mapping a hole-free round Gaussian onto a manifold with holes forces the mapping `z → features` to *curve*, so that the missing region acquires no preimage. A curved mapping is an entangled one: moving along a single `z` coordinate moves several factors at once, non-linearly. This curvature is unavoidable for *any* fixed input distribution that is constrained to match the data density.

**Truncation of the sampling region.** Regions of low prior density are poorly learned and yield poor samples; shrinking the sampling region toward its mean (Marchesi, 2017; Brock et al., 2018; Kingma & Dhariwal, 2018) trades some variation for higher average quality. Brock et al. found this only works for some networks and leaned on orthogonal regularization to make it reliable.

**Perceptual distance.** The LPIPS metric (Zhang et al., 2018) measures image dissimilarity as a learned weighted L2 between deep VGG-16 (Simonyan & Zisserman, 2014) embeddings, with weights fit to human judgments. It behaves quadratically for small perturbations, which makes it suitable for differentiating a smooth interpolation path.

## Baselines

- **Progressive-growing GAN (Karras et al., 2017).** The direct predecessor and natural implementation base. Generator: latent `z` enters at the first layer, then a stack of conv blocks each doubling resolution, ending in a 1×1 "toRGB" convolution; trained with progressive growing, equalized learning rate, pixelwise normalization in the generator, and minibatch-stddev in the discriminator. It produces state-of-the-art megapixel faces. Its discriminator, minibatch sizes, Adam settings, and generator weight EMA form the practical starting point.
- **BigGAN (Brock et al., 2018).** Large-scale GAN that, among other things, popularized the truncation trick and explored feeding parts of the latent to multiple generator layers. Its truncation uses orthogonal regularization.
- **Self-modulation (Chen et al., 2018).** Parallel work that modulates generator feature maps with AdaIN-style affine parameters derived from the latent — closely related in spirit — aimed at training stability.
- **Encoder-based disentanglement metrics (β-VAE; Kim & Mnih; Chen et al.; Eastwood & Williams).** The existing way to *measure* disentanglement. They all require an encoder mapping images back to latents and/or known ground-truth factors.

## Evaluation settings

- **Datasets.** CelebA-HQ (Karras et al., 2017), 1024² aligned celebrity faces; LSUN Bedroom/Car/Cat for non-face generality. A new, more varied high-quality face dataset at 1024² is also part of the setting. The 40 binary CelebA attributes (male/female, glasses, etc.) are available for attribute-based analysis.
- **Quality metric.** Fréchet Inception Distance (Heusel et al., 2017), computed between Inception features of real and generated images; lower is better. Here it is computed from 50,000 generated images against the training set, reporting the best value over training.
- **Disentanglement yardsticks.** Latent-space interpolation is the qualitative habit; LPIPS provides a perceptual ruler that a quantitative metric can be built on. For an attribute-based linear analysis, auxiliary attribute classifiers with the discriminator's architecture (minibatch-stddev removed), trained on CelebA-HQ labels, are the natural tool.
- **Protocol.** Progressive growing from a low resolution up to 1024². Adam optimizer with resolution-dependent minibatch sizes; an exponential moving average of the generator weights used for evaluation; mirror augmentation for faces. Non-saturating logistic loss with R1 for the face dataset, WGAN-GP for CelebA-HQ.

## Code framework

The primitives below are a generic progressive-growing GAN harness with the generator and measurement slots left open.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight(shape, gain=np.sqrt(2), lrmul=1.0):
    # Equalized learning rate: store N(0,1) weights, scale by He constant at runtime.
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)
    w = nn.Parameter(torch.randn(*shape) / lrmul)
    return w, he_std * lrmul              # caller multiplies w by the runtime coef

class EqLinear(nn.Module):                # fully-connected with equalized LR
    def __init__(self, fin, fout, gain=np.sqrt(2), lrmul=1.0, bias=True): ...
    def forward(self, x): ...

class EqConv2d(nn.Module):                # 3x3 / 1x1 conv with equalized LR
    def __init__(self, fin, fout, k, gain=np.sqrt(2)): ...
    def forward(self, x): ...

def leaky_relu(x): return F.leaky_relu(x, 0.2)

def pixel_norm(x, eps=1e-8):             # unit-length feature vector per pixel
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)

def blur2d(x, f=(1, 2, 1)):             # separable binomial low-pass for resampling
    ...

def upscale2d(x): ...
def downscale2d(x): ...

def minibatch_stddev(x, group_size=4):  # discriminator variation feature
    ...

class Discriminator(nn.Module):
    # progressive-growing discriminator (fromRGB, conv blocks, minibatch-stddev, score)
    def forward(self, img, lod): ...

def d_logistic_r1(D, reals, fakes, gamma=10.0):
    # non-saturating logistic loss + R1 gradient penalty on reals
    ...

class LatentBlock(nn.Module):
    """TODO: prepare the sampled latent for the image-producing network."""
    def forward(self, z):
        pass

class GeneratorBody(nn.Module):
    """TODO: define the image-producing network."""
    def forward(self, prepared_latent):
        pass

class CandidateGenerator(nn.Module):
    """TODO: compose the latent block and image-producing network."""
    def forward(self, z):
        pass

def latent_path_metric(candidate, distance_model, space):
    """TODO: measure interpolation behavior in a latent space."""
    pass

def attribute_direction_metric(candidate, classifiers, space):
    """TODO: measure whether attributes are simple directions in a latent space."""
    pass
```
