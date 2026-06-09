# Context

## Research question

A huge fraction of problems in image processing, graphics, and vision share one shape: take an input image and produce a corresponding output image of the same scene in a different representation. Semantic label map to photo; photo to semantic labels; edge map to photo; grayscale to color; aerial photo to map and back; daytime photo to nighttime; thermal to RGB; a photo with a hole to an inpainted photo. In every case the setting is identical — predict pixels from pixels — yet each task has historically been attacked with its own special-purpose machinery and its own hand-designed loss function.

The goal is a single general-purpose recipe: one architecture, one objective, that a practitioner can point at any paired image-to-image dataset and train, with no task-specific loss engineering. The deeper pain point is that, while convolutional networks have automated the learning of the *mapping*, we still hand-design the *loss* — we still have to tell the network what to minimize. A naive choice (Euclidean distance to the ground-truth pixels) is easy to write down but produces blurry, washed-out results. What a solution would have to achieve is to *learn* an appropriate loss for the task from the data, so that "make the output look like a real example of the target domain, and make it correspond to this particular input" can be specified at a high level rather than encoded by hand for each new problem.

## Background

**Per-pixel regression and why it blurs.** The simplest way to learn an image-to-image map is to train a CNN to minimize a per-pixel distance to the ground truth, typically the Euclidean (L2) distance `E[‖y − f(x)‖_2^2]`. This has a well-understood failure mode. When the conditional distribution `p(y|x)` is multimodal — many distinct output images are plausible for a given input — the predictor that minimizes expected squared error is the conditional mean `E[y|x]`. Averaging many sharp, mutually inconsistent plausible images produces a blur. This is why L2-regression outputs are smooth and desaturated (Pathak et al. 2016 on inpainting; Zhang et al. 2016 on colorization both report this). The L1 distance has the same qualitative weakness but is milder: its minimizer is the conditional *median* rather than the mean, which tends to select a single plausible value instead of blending them, and is more robust to outliers — so L1 blurs less than L2, though it still cannot synthesize sharp high-frequency detail on its own.

**Structured vs. unstructured losses.** Per-pixel classification or regression treats the output space as *unstructured*: each output pixel is modeled as conditionally independent of all others given the input. Such a loss cannot penalize the joint configuration of the output — whether the pixels together form a coherent object boundary, a plausible texture, a realistic global layout. A *structured* loss instead penalizes the joint configuration. A large body of work builds structured losses by hand: conditional random fields (Chen et al. 2014), the SSIM perceptual metric (Wang et al. 2004), feature matching in a fixed deep network (Dosovitskiy & Brox 2016), nonparametric losses, the convolutional pseudo-prior, and losses that match covariance/Gram statistics of deep features (Johnson et al. 2016, "perceptual loss"). Each of these fixes a particular notion of structure in advance. None is general; choosing the right one is itself expert work.

**Generative adversarial networks.** Goodfellow et al. (2014) introduced a way to *learn* a loss. A generator `G: z → y` maps a noise vector to an image; a discriminator `D` is trained to classify images as real (drawn from data) or fake (produced by `G`). They play a minimax game

```
min_G max_D  E_y[log D(y)] + E_z[log(1 − D(G(z)))].
```

`D` is, in effect, a learned loss for `G` that adapts to the data distribution: anything that looks obviously fake — a blur, for instance — is easy for `D` to reject, so `G` is pushed away from it. In practice the generator's half of the objective saturates early (when `D` confidently rejects, `log(1 − D(G(z)))` has vanishing gradient), so the standard fix is the *non-saturating* form: train `G` to maximize `log D(G(z))` instead of to minimize `log(1 − D(G(z)))`. Because the loss is learned rather than hand-specified, the same GAN machinery can in principle serve tasks that would otherwise demand very different hand-built losses.

**Conditional GANs.** Mirza & Osindero (2014) extended GANs so that both `G` and `D` are conditioned on side information — a class label, text, or an image. The generator learns a conditional generative model `p(y | conditioning)`, and the discriminator judges samples in the context of that conditioning. Conditioning on an *input image* is the natural fit for image-to-image translation. A separate line of work applied GANs to image outputs but kept the discriminator *unconditional* (it saw only the output), relying on an auxiliary L2 term to tie the output to the input — used for inpainting (Pathak et al. 2016), super-resolution (Ledig et al. 2016), user-guided manipulation (Zhu et al. 2016), and style transfer (Li & Wand 2016). Each of these was engineered for one application.

**Encoder-decoder generators.** Many image-prediction networks (Hinton & Salakhutdinov 2006; Pathak et al. 2016) use an encoder-decoder: the input is progressively downsampled to a small bottleneck, then progressively upsampled back to full resolution. All information must pass through the narrow bottleneck. But in translation tasks the input and output, though differing in surface appearance, are renderings of the *same* underlying scene — they share a great deal of low-level structure (the location of prominent edges, for instance). Forcing that shared structure through the bottleneck and reconstructing it from scratch is wasteful and lossy.

**U-Net.** Ronneberger et al. (2015) addressed exactly this in biomedical segmentation by adding skip connections to an encoder-decoder: the activations of encoder layer `i` are concatenated onto decoder layer `n − i` (with `n` the total number of layers), letting low-level information shortcut directly across the network rather than squeezing through the bottleneck.

**DCGAN backbone.** Radford et al. (2015) established a stable convolutional GAN recipe: build generator and discriminator from modules of the form convolution–BatchNorm (Ioffe & Szegedy 2015)–ReLU, use strided convolutions to down/upsample rather than pooling, and use leaky ReLUs in the discriminator. This is the architectural starting point on the table.

**Markov-random-field / texture models.** A long tradition models images as Markov random fields, assuming pixels separated by more than a small neighborhood are independent. Texture synthesis (Efros & Leung 1999; Gatys et al. 2015) and patch-based style transfer (Li & Wand 2016, who proposed a patch-wise "Markovian" discriminator to capture local style statistics) exploit this: realistic *local* statistics, repeated, produce realistic texture.

**Two empirical observations about existing systems.** First, per-pixel L2/L1 outputs are not uniformly wrong: the coarse layout and rough color come out roughly right, while the visible failure is the blur in fine detail. Second, when noise is fed to a conditional image generator as an explicit Gaussian input alongside the conditioning image, the generator tends to learn to *ignore* it (also observed by Mathieu et al. 2015): the conditioning is informative enough that the optimum is nearly deterministic, so the noise channel carries no gradient signal and dies.

## Baselines

**L1 / L2 per-pixel regression.** Train `G` (a CNN) to minimize `E_{x,y}[‖y − G(x)‖_p]`. Simple, stable, and it does get the coarse, low-frequency content right. But it is an unstructured loss and its optimum under output uncertainty is an average (L2: mean) or a robust central value (L1: median) of all plausible outputs, so it cannot synthesize sharp high-frequency detail and produces blurry, desaturated images. Gap: no mechanism to commit to a single sharp, realistic mode; no penalty on joint output structure.

**Unconditional GAN + reconstruction term.** Use a discriminator that sees only the output, `D(y)`, plus an auxiliary L2/L1 term to anchor the output to the input. The adversarial term enforces realism (so outputs are sharp), and the reconstruction term is what actually carries the correspondence to the input. Gap: the discriminator never sees the input, so it cannot directly penalize a mismatch between input and output — it only knows whether the output looks like *some* real image. Empirically a generator trained this way can collapse toward producing the same realistic output regardless of input; correspondence rests entirely on the regression term.

**Conditional GAN (discrete/text/image conditioning).** Condition both `G` and `D` on side information so the discriminator judges samples *in context*. Prior conditional-GAN work conditioned on class labels, text, or images, but each instance was specialized to a particular application, and it remained unclear how well an image-conditional GAN could work as a *general* image-to-image solution across many tasks with a single architecture and objective. Gap: no general framework; no systematic analysis of the architectural choices (what the discriminator should look at, what the generator should look like) that make it work across tasks.

**Encoder-decoder generator.** A downsample-to-bottleneck-then-upsample network. Gap: forces all information, including the low-level structure shared between input and output, through the bottleneck, losing detail that ought to be passed straight across.

## Evaluation settings

Paired image-to-image datasets, where each example is an aligned `(input, output)` image pair, both 1–3 channel images, typically at `256 × 256` resolution. Tasks and corpora that exist as natural yardsticks: Cityscapes (Cordts et al. 2016) for semantic labels↔photo; CMP Facades (Tyleček & Šára 2013) for architectural labels→photo; map↔aerial photo pairs scraped from Google Maps; ImageNet (Russakovsky et al. 2015) for grayscale→color; edge↔photo on handbags (Zhu et al. 2016) and UT Zappos50K shoes (Yu & Grauman 2014), with edges from the HED detector (Xie & Tu 2015); day→night (Laffont et al. 2014); thermal→color (Hwang et al. 2015); Paris StreetView (Doersch et al. 2012) for inpainting.

Evaluating synthesized-image quality is itself hard, and per-pixel mean-squared error does not measure joint structure. Two protocols are available. A "real vs. fake" perceptual study on Amazon Mechanical Turk: a human is shown a real and a generated image briefly and must say which is fake; the metric is the fraction of trials on which the generated image fools the observer. And an "FCN-score": run an off-the-shelf semantic segmentation network (FCN-8s, Long et al. 2015, trained on real data) on the synthesized images and measure how well it classifies them (per-pixel accuracy, per-class accuracy, class IOU) against the labels the images were synthesized from — the intuition being that a realistic synthesis should be parseable by a recognizer trained on real images. Standard data augmentation: random jitter (resize `256→286`, random crop back to `256`) and horizontal mirroring.

## Code framework

The pieces that already exist: a data pipeline yielding aligned image pairs, the conv–norm–ReLU module from the DCGAN recipe, the Adam optimizer, the GAN minimax training loop with its non-saturating generator update, and the L1/L2 regression losses. What does not yet exist is the architecture of the generator, what the discriminator looks at and at what spatial scale, how noise enters, and how the adversarial and reconstruction terms combine. Those are the empty slots.

```python
import torch
import torch.nn as nn

def conv_norm_relu(in_ch, out_ch, down=True, leaky=False, norm=True):
    # DCGAN-style module: 4x4 stride-2 conv (down) or transposed conv (up),
    # optional BatchNorm, ReLU (leaky 0.2 in the down/critic path).
    layers = []
    if down:
        layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm))
    else:
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not norm))
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True))
    return nn.Sequential(*layers)


class ImageToImageNet(nn.Module):
    """Maps an input image to an output image of the same scene.
    The internal architecture is the open question."""
    def __init__(self, in_ch, out_ch, ngf=64):
        super().__init__()
        # TODO: how should information flow from input grid to output grid,
        #       given that input and output share low-level structure?
        pass

    def forward(self, x):
        # TODO: produce G(x); inject stochasticity somehow (Gaussian input? dropout?)
        raise NotImplementedError


class Critic(nn.Module):
    """Scores how 'good' an output is — a learned loss for the generator.
    Open questions: does it see the input too? at what spatial scale does it judge?"""
    def __init__(self, in_ch, ndf=64):
        super().__init__()
        # TODO: what is the input to the critic, and what does its output represent?
        pass

    def forward(self, *args):
        raise NotImplementedError


def critic_realism_loss(critic, real_pair, fake_pair):
    # Standard GAN real/fake classification loss (BCEWithLogits / cross-entropy).
    # TODO: what exactly is a "pair" here — output alone, or input together with output?
    raise NotImplementedError


def reconstruction_loss(fake_out, real_out):
    # TODO: which norm anchors the output to the ground truth, and why that one?
    raise NotImplementedError


def train_step(G, D, opt_G, opt_D, x, y):
    fake = G(x)
    # --- update D ---
    opt_D.zero_grad()
    loss_D = critic_realism_loss(D, real_pair=(x, y), fake_pair=(x, fake.detach()))
    loss_D.backward(); opt_D.step()
    # --- update G ---
    opt_G.zero_grad()
    loss_G = None   # TODO: realism term (fool D) + reconstruction term, weighted by lambda
    loss_G.backward(); opt_G.step()
```
