# Context: Translating between image domains without paired examples

## Research question

We want to learn to translate an image from one domain into another — grayscale to color, an edge map to a photograph, a semantic-label map to a street scene, a summer landscape to a winter one, a horse to a zebra, a photograph into the style of a painter — when **no aligned input/output pairs are available**.

In the supervised setting the problem is well posed: we are handed examples {(x_i, y_i)} where x_i and y_i are the *same* scene rendered two ways, and we fit a function that maps x_i to its known partner y_i. But aligned pairs are scarce or impossible to obtain. Semantic-segmentation datasets with paired label maps and photos are small and expensive to annotate. Artistic stylization would require an artist to hand-author the target for every input. And for object transfiguration — turning a horse into a zebra — the desired output is not even well defined: there is no photograph of *this* horse as a zebra anywhere in the world.

What we *do* have, cheaply and in quantity, is two unaligned collections: a set {x_i} drawn from domain X and a separate set {y_j} drawn from domain Y, with no information about which x matches which y. The goal is to learn a mapping X→Y from this set-level supervision alone, such that an output is (a) indistinguishable from a real image of the target domain and (b) still recognizably a translation of *the specific input that produced it* — its structure preserved, only its appearance changed. A solution must somehow recover, from two unordered piles of images, the correspondence that paired supervision would otherwise have provided.

## Background

**Adversarial training matches distributions.** A generative adversarial network pits a generator against a discriminator: the discriminator is trained to tell real samples from generated ones, the generator is trained to fool it. With a generator G that maps inputs to the target domain and a discriminator D_Y trained to separate real y from generated G(x), the value being optimized is

  E_y[log D_Y(y)] + E_x[log(1 − D_Y(G(x)))], min over G, max over D_Y.

For a fixed G the optimal discriminator is D_Y*(y) = p_data(y) / (p_data(y) + p_g(y)), and substituting it back reduces the generator's objective to 2·JSD(p_data ‖ p_g) − log4, the Jensen–Shannon divergence between the real target distribution and the generated one. This is minimized exactly when p_g = p_data. So in principle adversarial training drives the *distribution* of generated images to match the target domain's distribution. This is the engine that lets us demand "looks like domain Y" without ever writing down what domain Y is.

**But distribution-matching alone is loose.** Matching the output distribution constrains only the marginal over outputs, not the joint over (input, output). There are infinitely many maps G that all push p_data(X) onto a distribution identical to p_data(Y): compose a distribution-matching G with any permutation of target images that leaves p_data(Y) fixed, and you get a different input→output correspondence with exactly the same output distribution. The objective cannot tell these apart, so it gives no guarantee that an individual input x maps to a *meaningful* output. This is a structural, knowable-in-advance fact about what an adversarial loss can and cannot pin down.

**And in practice it collapses.** Optimizing the adversarial objective in isolation is notoriously unstable. The well-documented failure mode is mode collapse: the generator funnels many or all inputs onto a single output the discriminator currently rates as real, the gradient signal flattens, and optimization stops making progress. So adversarial training is both under-constrained in theory and fragile in practice.

**Prior consistency-checking techniques in vision and language.** In visual tracking, forward-backward consistency — track a point forward then backward and check it returns — has been a standard reliability check for decades (Kalal et al. 2010; Sundaram et al. 2010). In machine and human translation, "back-translation and reconciliation" verifies a translation by translating it back and comparing to the source (Brislin 1970; He et al. 2016 use this as a learning signal in *dual learning*). Higher-order consistency constraints have appeared in structure-from-motion, 3D shape matching, co-segmentation, and dense correspondence; Zhou et al. (2016) and Godard et al. (2016) train CNNs without labels using such constraints as a supervision signal.

**Building blocks that already exist.** Residual blocks (He et al. 2016) make deep networks trainable by learning a perturbation off the identity. Instance normalization (Ulyanov et al. 2016) normalizes each image's per-channel statistics independently and was found to help feed-forward stylization by removing instance-specific contrast. The fast feed-forward image-transformation architecture of Johnson et al. (2016) — downsampling convolutions, a stack of residual blocks, fractionally-strided upsampling convolutions — gave high-quality results for style transfer and super-resolution. The least-squares GAN (Mao et al. 2017) replaces the saturating cross-entropy adversarial loss with a squared-error loss and is more stable. Training GANs on a history of previously generated samples (Shrivastava et al. 2016) reduces oscillation.

## Baselines

**pix2pix — conditional GAN + L1 on paired data (Isola et al. 2016).** The supervised reference point. A conditional GAN whose generator maps x to y and whose discriminator sees (x, output) pairs, trained with an adversarial term plus an L1 reconstruction term ‖y − G(x)‖₁ against the *known* paired target. The discriminator is a 70×70 PatchGAN that judges local patches rather than the whole image. The core idea — learn the translation loss instead of hand-engineering it — is exactly what we want to keep. The gap: it requires aligned {(x_i, y_i)} pairs, the very thing unavailable here.

**CoGAN (Liu et al. 2016) and cross-modal scene networks (Aytar et al. 2016).** Learn a joint representation across two domains by *sharing weights* between two generators (and/or two discriminators) in their early or late layers, so that a shared latent code renders into both domains. Translation is indirect: find the latent that produces a given x, then render it in Y. The gap: it leans on a weight-sharing assumption that a single shared latent suffices, and the translation is round-about rather than a direct learned map.

**Content-sharing with a predefined similarity term (Shrivastava et al. 2016 SimGAN; Taigman et al. 2016 DTN; Bousmalis et al. 2016).** These keep an adversarial term but add a regularizer forcing output to stay close to input in some *predefined* space: pixel space (SimGAN adds ‖x − G(x)‖₁, suitable only when input and output already nearly coincide, e.g. refining synthetic images), a deep feature space (DTN matches a fixed pretrained embedding), or a class-label space (Bousmalis et al.). The gap: each bakes in a hand-chosen metric of what should be preserved, which presumes input and output live in the same low-dimensional space and breaks down under large appearance change.

**BiGAN / ALI (Donahue et al. 2016; Dumoulin et al. 2016).** Jointly learn a generator from latent to data and an inverse encoder from data to latent. The gap: it is built for inverting a generator into a latent code, not for relating two image domains.

**Neural style transfer (Gatys et al. 2015; Johnson et al. 2016).** Synthesizes an image combining the content of one image with the style of another by matching Gram-matrix statistics of pretrained deep features. The gap: it transfers the style of a *single* exemplar image, not a whole *collection*; it needs a matched style image and does not learn a mapping between two domains.

Every one of these either needs paired data or assumes a predefined notion of similarity between input and output. A method that needs neither — only two unordered collections plus a structural assumption — would be a general-purpose solution.

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

The available scaffold is a feed-forward image transform, a local discriminator, Adam, L1/MSE losses, and a single-direction adversarial training step. The unresolved slot is a generic constraint that makes an unpaired output remain tied to the input that produced it instead of merely matching the target marginal.

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
    # adversarial loss makes fake_y look like domain Y ...
    # ... but matching the Y distribution does NOT tie fake_y to this specific x.
    #
    # TODO: add the missing unpaired constraint that pins fake_y to this
    #       specific x without access to any paired target.
    pass
```

The single open slot is the missing constraint that ties each generated output to the input that produced it using only unordered collections.
