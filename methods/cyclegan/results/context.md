# Context: Translating between image domains without paired examples

## Research question

We want to learn to translate an image from one domain into another — grayscale to color, an edge map to a photograph, a semantic-label map to a street scene, a summer landscape to a winter one, a horse to a zebra, a photograph into the style of a painter — when **no aligned input/output pairs are available**.

In the supervised setting we are handed examples {(x_i, y_i)} where x_i and y_i are the *same* scene rendered two ways, and we fit a function that maps x_i to its known partner y_i. Aligned pairs come from sources like segmentation datasets with paired label maps and photos, or stylizations hand-authored by an artist. For object transfiguration — turning a horse into a zebra — the desired output is not even well defined: there is no photograph of *this* horse as a zebra anywhere in the world.

What we *do* have, cheaply and in quantity, is two unaligned collections: a set {x_i} drawn from domain X and a separate set {y_j} drawn from domain Y, with no information about which x matches which y. The goal is to learn a mapping X→Y from this set-level supervision alone, so that an output looks like a real image of the target domain while remaining a translation of the specific input that produced it.

## Background

**Adversarial training matches distributions.** A generative adversarial network pits a generator against a discriminator: the discriminator is trained to tell real samples from generated ones, the generator is trained to fool it. With a generator G that maps inputs to the target domain and a discriminator D_Y trained to separate real y from generated G(x), the value being optimized is

  E_y[log D_Y(y)] + E_x[log(1 − D_Y(G(x)))], min over G, max over D_Y.

For a fixed G the optimal discriminator is D_Y*(y) = p_data(y) / (p_data(y) + p_g(y)), and substituting it back reduces the generator's objective to 2·JSD(p_data ‖ p_g) − log4, the Jensen–Shannon divergence between the real target distribution and the generated one. This is minimized exactly when p_g = p_data. So adversarial training drives the *distribution* of generated images to match the target domain's distribution, which is what lets us demand "looks like domain Y" without ever writing down what domain Y is.

The adversarial objective constrains the marginal over outputs. A common observed behavior when optimizing it in isolation is mode collapse, where the generator maps many inputs onto a single output the discriminator currently rates as real.

**Consistency constraints as supervision.** Across vision and language, constraints that hold without labels have been used as a training signal: structure-from-motion, 3D shape matching, co-segmentation, and dense correspondence all exploit such constraints, and Zhou et al. (2016) and Godard et al. (2016) train CNNs without labels using them as a supervision signal.

**Building blocks that already exist.** Residual blocks (He et al. 2016) make deep networks trainable by learning a perturbation off the identity. Instance normalization (Ulyanov et al. 2016) normalizes each image's per-channel statistics independently and was found to help feed-forward stylization by removing instance-specific contrast. The fast feed-forward image-transformation architecture of Johnson et al. (2016) — downsampling convolutions, a stack of residual blocks, fractionally-strided upsampling convolutions — gave high-quality results for style transfer and super-resolution. The least-squares GAN (Mao et al. 2017) replaces the saturating cross-entropy adversarial loss with a squared-error loss and is more stable. Training GANs on a history of previously generated samples (Shrivastava et al. 2016) reduces oscillation.

## Baselines

**pix2pix — conditional GAN + L1 on paired data (Isola et al. 2016).** A conditional GAN whose generator maps x to y and whose discriminator sees (x, output) pairs, trained with an adversarial term plus an L1 reconstruction term ‖y − G(x)‖₁ against the *known* paired target. The discriminator is a 70×70 PatchGAN that judges local patches rather than the whole image. It learns the translation loss instead of hand-engineering it, and requires aligned {(x_i, y_i)} pairs.

**CoGAN (Liu et al. 2016) and cross-modal scene networks (Aytar et al. 2016).** Learn a joint representation across two domains by *sharing weights* between two generators (and/or two discriminators) in their early or late layers, so that a shared latent code renders into both domains. Translation is indirect: find the latent that produces a given x, then render it in Y.

**Content-sharing with a predefined similarity term (Shrivastava et al. 2016 SimGAN; Taigman et al. 2016 DTN; Bousmalis et al. 2016).** These keep an adversarial term but add a regularizer forcing output to stay close to input in some *predefined* space: pixel space (SimGAN adds ‖x − G(x)‖₁, used when input and output already nearly coincide, e.g. refining synthetic images), a deep feature space (DTN matches a fixed pretrained embedding), or a class-label space (Bousmalis et al.).

**BiGAN / ALI (Donahue et al. 2016; Dumoulin et al. 2016).** Jointly learn a generator from latent to data and an inverse encoder from data to latent, inverting a generator into a latent code.

**Neural style transfer (Gatys et al. 2015; Johnson et al. 2016).** Synthesizes an image combining the content of one image with the style of another by matching Gram-matrix statistics of pretrained deep features. It transfers the style of a *single* exemplar image, given a matched style image.

## Evaluation settings

The natural yardsticks are tasks where unaligned collections are easy to assemble:

- **Cityscapes labels↔photo** (Cordts et al. 2016): 2975 training images at 128×128, val set for testing; supports a "translate then score with an off-the-shelf segmenter" protocol (FCN-score: run a pretrained segmentation network on label→photo outputs and measure per-pixel accuracy, per-class accuracy, and mean IoU against ground-truth label maps) and the inverse photo→labels.
- **Maps↔aerial photo** (≈1096 images scraped from Google Maps, 256×256), split about the median latitude with a buffer so no training pixel leaks into test.
- **Architectural facades labels↔photo** (CMP Facade Database, 400 images).
- **Edges→shoes** (UT Zappos50K, ≈50,000 images).
- **Object transfiguration** horse↔zebra, apple↔orange (ImageNet *wild horse*, *zebra*, *apple*, *navel orange* synsets, scaled to 256×256).
- **Season transfer** summer↔winter Yosemite (Flickr, 256×256).
- **Collection style transfer** photo↔art for Monet, Van Gogh, Cézanne, Ukiyo-e (Wikiart + Flickr landscape photos, 256×256).
- **Photo enhancement** smartphone→DSLR shallow-depth-of-field flower photos (Flickr).

For perceptual quality where no ground truth exists, a real-vs-fake human study (e.g. an AMT "which is real?" perceptual test) is the established protocol; for label↔photo, the FCN-score and standard segmentation metrics give an automatic proxy.

## Code framework

The available scaffold is a feed-forward image transform, a local discriminator, Adam, L1/MSE losses, and a single-direction adversarial training step.

```python
import torch
import torch.nn as nn

# --- known building blocks (residual feed-forward image transformer) ---
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim),
        )
    def forward(self, x):
        return x + self.conv_block(x)   # learn a residual off the identity

class Generator(nn.Module):
    # downsample convs -> residual blocks -> fractionally-strided upsample convs -> Tanh
    def __init__(self, in_nc, out_nc, ngf=64, n_blocks=9):
        super().__init__()
        # TODO: build the feed-forward transformer body
        pass
    def forward(self, x):
        pass

class PatchDiscriminator(nn.Module):
    # fully-convolutional 70x70 patch classifier: C64-C128-C256-C512 -> 1-channel map
    def __init__(self, in_nc, ndf=64):
        super().__init__()
        pass
    def forward(self, x):
        pass

# --- known losses / optimizer ---
gan_criterion = nn.MSELoss()       # least-squares adversarial loss (stable form)
recon_criterion = nn.L1Loss()      # L1 reconstruction (sharper than L2 for images)

# --- known adversarial training scaffold (one direction, paired-style) ---
def train_step(x, y, G, D_Y, opt_G, opt_D):
    fake_y = G(x)
    # adversarial loss makes fake_y look like domain Y from unordered collections.
    #
    # TODO: relate each generated output to the input that produced it, using
    #       only the two unordered collections {x_i} and {y_j}.
    pass
```
