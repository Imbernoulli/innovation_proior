# Context

## Research question

Single-image super-resolution (SISR) asks for a high-resolution image `I^HR` given only a
low-resolution input `I^LR`. The problem is fundamentally underdetermined: many distinct
high-resolution images, differing in their fine texture, downsample to the same low-resolution image,
so for a given input there is a whole *set* of equally-plausible sharp solutions. This
ill-posedness is mild at small upscaling factors but severe at large ones — at `4×` (a `16×` increase
in pixel count) the high-frequency texture detail is essentially absent from the input and must be
*invented* plausibly rather than recovered exactly.

The dominant supervised approach minimizes the per-pixel mean squared error (MSE) between the recovered
image and the ground-truth high-resolution image. This is convenient because minimizing MSE also
maximizes the peak signal-to-noise ratio (PSNR), the standard quantitative metric. But the resulting
images, while high in PSNR, are perceptually unsatisfying: they are overly smooth and lack
high-frequency texture. The precise question: **how do we recover *photo-realistic* fine texture when
super-resolving at large (`4×`) upscaling factors — producing an image that looks like a real
photograph at the higher resolution, rather than a blurred compromise — given that the standard
per-pixel objective is exactly what produces the blur?** What is wanted is an output that is one
sharp, natural-looking image rather than the average of the plausible set, judged by a criterion that
tracks human perception rather than raw pixel distance — but how to formulate such an objective is open.

## Background

The field state (mid-2010s deep super-resolution): convolutional networks have pushed SISR accuracy
and speed forward, but all the leading methods optimize a pixel-wise loss and inherit its perceptual
limitations. The load-bearing concepts and the key diagnostic observation:

- **Why pixel-wise MSE blurs (the diagnostic insight).** For a given low-resolution input, the
  high-resolution images consistent with it form a set of plausible sharp solutions. Minimizing the
  expected squared pixel error drives the estimate toward the *pixel-wise average* of that set — the
  conditional mean `E[I^HR | I^LR]`. Averaging many sharp, differently-textured images produces a
  smooth image that may sit *off* the manifold of natural images entirely. So the smoothing is not a
  modeling deficiency to be fixed with a bigger network; it is the direct, unavoidable consequence of
  the per-pixel objective. Pixel-wise losses, being defined on pixel differences, cannot capture
  perceptually relevant differences such as texture.

- **The natural-image manifold.** Real photographs occupy a thin manifold in pixel space. The
  centroid between several manifold points generally sits *off* the manifold, in the smooth region
  between them — which is where the per-pixel average lands.

- **The adversarial framework (Goodfellow et al., 2014).** A generator and a discriminator play a
  minimax game; the discriminator learns to distinguish real images from generated ones and thereby
  supplies a learned signal for "does this look like a real image?" This was already being used to
  generate plausible natural images (Mathieu et al., 2015; Denton et al., 2015) and to learn mappings
  between manifolds for style transfer and inpainting. A stable convolutional recipe for it
  (Radford et al., 2015) — strided convolutions instead of pooling, leaky ReLU in the discriminator,
  batch normalization — exists and would guide a discriminator design.

- **Perceptual / feature-space losses (Gatys et al., 2015; Johnson et al., 2016; Bruna et al., 2016).**
  Instead of comparing images pixel by pixel, compare the *feature maps* they induce in a pre-trained
  convolutional network (VGG19; Simonyan & Zisserman, 2014). These deep features are more invariant to
  pixel-space changes, so matching them lets two images agree on content while differing in exact
  pixels — allowing texture that a pixel loss would penalize. This had already produced more convincing
  super-resolution and style-transfer results than pixel losses.

- **Residual learning and learned upscaling.** Very deep networks are hard to train, but residual
  blocks with skip connections (He et al., 2015, 2016) relieve the network of modeling the identity
  mapping and let depth grow — and depth has been shown to help SISR. Batch normalization (Ioffe &
  Szegedy, 2015) eases training of deep nets. Separately, learning the upscaling filters *inside* the
  network via sub-pixel convolution (Shi et al., 2016, ESPCN) — operating in low-resolution feature
  space and rearranging channels into spatial resolution at the end — is more accurate and far cheaper
  than first bicubically upsampling the input and then convolving.

## Baselines

The prior methods a new procedure would be measured against and reacts to:

- **Interpolation (bicubic, Lanczos; Duchon, 1979).** Fast filtering. *Gap:* oversimplifies SISR,
  yielding overly smooth textures.

- **Example- and patch-based / sparse-coding / self-exemplar methods (Freeman et al., 2000; Glasner et
  al., 2009; Huang et al., 2015, SelfExSR).** Establish a complex LR↔HR mapping from training pairs or
  cross-scale self-similarity. *Gap:* computationally heavy optimization; limited high-frequency
  synthesis.

- **SRCNN (Dong et al., 2014, 2016).** A three-layer fully-convolutional network trained end-to-end on
  bicubically-upsampled inputs with an MSE loss; state of the art in PSNR at its time. *Gap:* shallow,
  operates in HR space (expensive), and MSE-smooth.

- **ESPCN (Shi et al., 2016).** Introduces sub-pixel convolution to learn the upscaling in LR space —
  fast enough for real-time video. *Gap:* still optimizes pixel-wise MSE.

- **DRCN (Kim et al., 2016).** A deeply-recursive convolutional network capturing long-range pixel
  dependencies with few parameters. *Gap:* MSE objective, hence the same perceptual ceiling.

- **Feature-space perceptual SR (Johnson et al., 2016; Bruna et al., 2016).** Replace pixel MSE with
  the Euclidean distance between VGG19 feature maps — the closest ancestors in spirit, and visibly more
  convincing than pixel losses. *Gap:* the reconstructions still do not reach photo-realistic texture
  at large upscaling factors.

## Evaluation settings

The benchmarks, datasets, metrics, and protocol that form the natural yardstick:

- **Datasets.** Set5 (Bevilacqua et al., 2012), Set14 (Zeyde et al., 2012), and BSD100 (the test set of
  BSD300; Martin et al., 2001). All experiments at a `4×` scale factor. Training is done on a large
  generic image corpus (ImageNet), with `I^LR` produced by bicubically downsampling `I^HR` by `r = 4`.

- **Quantitative metrics.** PSNR [dB] and SSIM (Wang et al., 2004), computed on the luminance
  (y-) channel of center-cropped images (a few border pixels removed) via a standard package, for fair
  comparison across methods. These are the conventional yardsticks — and their key, known limitation is
  that, being pixel-based, they do not capture perceptual quality (the highest-PSNR image is often not
  the perceptually best one).

- **Perceptual metric.** Because PSNR/SSIM miss perceptual quality, the appropriate yardstick is a mean
  opinion score (MOS) test: human raters assign each super-resolved image an integer score from 1 (bad)
  to 5 (excellent), with raters calibrated on anchor images, and significance assessed by paired
  two-sided Wilcoxon signed-rank tests.

## Code framework

The available substrate is a deep-learning library with convolutions, residual blocks, batch
normalization, sub-pixel-convolution upscaling, an adversarial-training harness, and a frozen,
pre-trained VGG19 feature extractor. What exists: a residual super-resolution generator that maps an
LR image to an HR estimate, a convolutional discriminator, and a fixed VGG network to extract features.
What is missing: the *objective* — what loss to train the generator on so that the output is sharp and
photo-realistic rather than smooth.

```python
import torch
import torch.nn as nn
import torchvision

upscale_factor = 4

class Generator(nn.Module):
    # LR image -> HR estimate (deep residual network with learned upscaling)
    def __init__(self, n_blocks=16):
        super().__init__()
        # TODO: first conv; n_blocks residual blocks with a long skip; sub-pixel upscaling; output conv
        self.net = None
    def forward(self, lr):
        pass

class Discriminator(nn.Module):
    # HR image -> probability "this is a real photograph"
    def __init__(self):
        super().__init__()
        # strided-conv / leaky-ReLU / batchnorm classifier, no pooling
        self.net = None
    def forward(self, img):
        pass

# frozen feature extractor available for measuring similarity in feature space
vgg = torchvision.models.vgg19(pretrained=True).features.eval()
for p in vgg.parameters():
    p.requires_grad = False

def content_loss(sr, hr):
    # TODO: how should generator output be compared to ground truth?
    pass

def generator_objective(sr, hr, d_pred_on_sr):
    # TODO: the objective that makes the output sharp and photo-realistic
    #       rather than the smooth pixel-wise average of plausible solutions
    pass

def train_step(lr, hr, opt_g, opt_d):
    # alternate: update D (real HR vs generated SR), then update G on its objective
    pass
```

This harness can run a residual generator inside an adversarial game with a VGG feature extractor on
hand, but the choice of objective — pixel versus feature space, and how to incorporate the adversarial
signal — is the open problem.
