# pix2pix synthesis notes

## Pain point / research question
Many image tasks = "predict pixels from pixels": semantic labels->photo, edges->photo,
BW->color, map<->aerial, day->night, inpainting, thermal->color. Historically each
solved with a bespoke, hand-engineered loss/pipeline. Goal: ONE general framework,
same architecture+objective, just retrain on different paired data. Key insight: we
no longer hand-engineer the *mapping* (CNNs learn it), but we still hand-engineer the
*loss*. Can we learn the loss too?

## The central diagnostic fact (pre-method, knowable)
Naive CNN regression to minimize Euclidean (L2) distance to ground truth pixels =>
BLURRY outputs. Reason: under uncertainty (many plausible outputs y consistent with x),
the L2-optimal prediction is the conditional MEAN E[y|x]; averaging many sharp plausible
images = a blur. Per-pixel regression also treats output pixels as conditionally
independent given x ("unstructured" loss) -> no penalty on joint/structural correctness.

## Load-bearing ancestors
- **Per-task hand-designed translation losses** (Efros texture synthesis, image analogies
  Hertzmann 2001, colorization-specific losses Zhang 2016, FCN seg Long 2015): each works
  but is application-specific; no general recipe.
- **CNN regression with L2/L1**: unstructured per-pixel loss; L2 -> blur (averages modes);
  Pathak 2016 (context encoders), Zhang 2016 colorful note blur explicitly.
- **Structured losses**: CRFs (Chen 2014), SSIM, feature matching (Dosovitskiy 2016),
  nonparametric losses, convolutional pseudo-prior, covariance/Gram matching (Johnson 2016
  perceptual loss). They penalize JOINT configuration, but each is a fixed hand-built
  structure. cGAN instead *learns* the structured loss.
- **GANs** (Goodfellow 2014): G: z->y, D classifies real vs fake. minimax
  min_G max_D E_y[log D(y)] + E_z[log(1-D(G(z)))]. The discriminator IS a learned loss
  that adapts to data; blurry => obviously fake => penalized. Non-saturating trick:
  train G to maximize log D(G(z)) instead of minimize log(1-D(G(z))).
- **Conditional GANs** (Mirza 2014; conditioned on labels/text/images): G,D both see a
  conditioning variable. For image translation, condition on input image x.
- **Image-conditional GANs used UNconditionally** (Pathak inpainting, Ledig SRGAN,
  Zhou, Zhu iGAN, Li-Wand style): they applied GAN to output realism but relied on a
  separate L2 term to tie output to input -> D doesn't police correspondence.
- **Encoder-decoder** (Hinton 2006): downsample to bottleneck, upsample back. All info
  must pass through bottleneck. Problem: input/output in translation share low-level
  structure (edges, locations) that gets crushed at bottleneck.
- **U-Net** (Ronneberger 2015): encoder-decoder + skip connections layer i <-> n-i,
  concatenate channels. Lets low-level info shortcut the bottleneck.
- **DCGAN** (Radford 2015): conv-BatchNorm-ReLU module recipe, strided conv down/up,
  the architectural backbone adapted here.
- **MRF / texture / style models** (Efros-Leung texture, Gatys, Li-Wand precomputed
  Markovian GAN / "patch" discriminator capturing local style stats): model image as a
  Markov random field, independence beyond a patch diameter. Basis for PatchGAN.
- Batch norm (Ioffe 2015); instance norm (Ulyanov 2016) = batchnorm with batch stats at
  test, batch size 1; Adam (Kingma 2014).

## Derivation chain (discovery order)
1. Regression blurs because it averages modes. A GAN can instead pick a single sharp mode
   that lives on the data manifold (D rejects blur as fake). So make the loss adversarial.
2. But plain GAN only cares output looks real, not that it MATCHES the input x. Empirically
   G collapses to ~same output regardless of x. Fix: condition D on x too -> D(x,y) judges
   the (input,output) PAIR / correspondence. This is the conditional GAN.
   L_cGAN(G,D) = E_{x,y}[log D(x,y)] + E_{x,z}[log(1 - D(x,G(x,z)))].
   (Unconditional variant for ablation: L_GAN uses D(y), D(G(x,z)) — no x.)
3. Pure GAN sharp but has artifacts / doesn't nail location. Add a reconstruction term so
   output stays near ground truth. Use L1 not L2: L_L1(G)=E_{x,y,z}[||y - G(x,z)||_1].
   WHY L1: L2 optimum = conditional mean (blur); L1 optimum = conditional MEDIAN, which is
   one plausible value not an average -> sharper, and L1 is robust. (Colorfulness: L1
   picks median color -> grayish/desaturated; cGAN restores color distribution.)
   Final: G* = argmin_G max_D L_cGAN(G,D) + lambda * L_L1(G), lambda=100.
4. Low/high frequency SPLIT (the key insight). L1 (and L2) already capture LOW frequencies
   well even though they blur HIGH frequencies. So we don't need the GAN to police the
   whole image — only the high-frequency / fine-detail structure. L1 handles low freq.
5. High-frequency structure is LOCAL -> it suffices for D to look at local N×N patches.
   PatchGAN: D classifies each N×N patch real/fake, run convolutionally, average all
   responses. Models image as MRF (independence beyond patch diameter) = a texture/style
   loss. Benefits: fewer params, faster, applies to arbitrarily large images, no need to
   model long-range structure (L1 does that). Ablation: 1x1 PixelGAN (only color), 16x16,
   70x70 (best), 286x286 ImageGAN (too many params, harder, worse).
6. Generator: input grid -> output grid, same underlying structure, roughly aligned. Plain
   encoder-decoder forces everything through bottleneck and loses shared low-level info.
   Add skip connections (U-Net): concat layer i to layer n-i. Low-level structure (edges,
   locations) shuttles directly across.
7. Noise z: if provided as Gaussian input, G learns to ignore it (deterministic anyway).
   So supply noise only as DROPOUT, applied at several decoder layers at train AND test
   time. Output still nearly deterministic (open problem).

## Optimization details
- Alternate: one SGD step on D, then one on G.
- Non-saturating: train G to maximize log D(x,G(x,z)).
- Divide D objective by 2 (slows D relative to G).
- Adam lr=2e-4, beta1=0.5, beta2=0.999.
- Weights init N(0, 0.02). Random jitter: resize 256->286, crop back to 256. Mirroring.
- conv-BatchNorm-ReLU modules. All convs 4x4 stride 2. Encoder/D ReLU leaky 0.2;
  decoder ReLU not leaky. No BatchNorm on first C64.
- Test time: keep dropout ON; use batch stats for batchnorm (instance norm w/ bs=1).

## Architecture specifics (appendix)
Ck = Conv-BatchNorm-ReLU k filters. CDk = + Dropout 0.5. Convs 4x4 stride2.
- Encoder: C64-C128-C256-C512-C512-C512-C512-C512 (no BN on first C64).
- Decoder (enc-dec): CD512-CD512-CD512-C512-C256-C128-C64, then conv->out channels, Tanh.
- U-Net decoder (channels doubled by skips): CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128.
- 70x70 D: C64-C128-C256-C512 -> conv to 1 ch -> Sigmoid (no BN on first C64, leaky 0.2).
- 1x1 D (PixelGAN): C64-C128 with 1x1 convs. 16x16: C64-C128. 286x286: C64...C512x3.
- D input channels = input_nc + output_nc (sees concat[x,y]).

## Code grounding
junyanz/pytorch-CycleGAN-and-pix2pix: pix2pix_model.py (loss assembly, cat[A,B] for D,
loss_D*0.5, GAN + lambda_L1*L1), networks.py (UnetSkipConnectionBlock recursive U-Net,
NLayerDiscriminator PatchGAN, PixelDiscriminator 1x1, GANLoss BCEWithLogits/vanilla).
GANLoss uses BCEWithLogitsLoss (vanilla) — equivalent to the non-saturating real/fake
target labels. lambda_L1 default 100.

## Scaffold (pre-method, generic conditional image generation)
Bare paired-image harness: data loader yielding (input_image, target_image); a generic
ImageToImageNet (TODO architecture); an adversarial Critic that sees... (TODO what it
sees, what scale); a Loss (TODO: realism term + which reconstruction norm); training loop
alternating critic/generator. No U-Net, no PatchGAN, no method names. Baselines section
gets cGAN, L1-regression, encoder-decoder.
