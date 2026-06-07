# SRGAN synthesis (Phase 1.5)

Verified: arXiv 1609.04802, "Photo-Realistic Single Image Super-Resolution Using a GAN", Ledig et al. (Twitter), CVPR 2017.
Canonical impl: original Theano/Lasagne; faithful PyTorch = sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution (saved in code/). 16 residual blocks, 9x9 first/last conv, PixelShuffle, VGG i=5 j=4 after-activation. Matches paper exactly.

## Pain point / research question
- Single-image super-resolution (SISR): recover HR image I^SR from LR input I^LR. Ill-posed/underdetermined — many HR images map to the same LR; especially at large upscale (4x = 16x pixels).
- The reigning objective is per-pixel MSE between recovered and ground-truth HR. MSE maximizes PSNR (the standard metric). BUT minimizing MSE = finding the PIXEL-WISE AVERAGE of all plausible HR solutions → overly smooth, washed-out textures, no high-frequency detail. High PSNR ≠ perceptually good.
- So: how to recover photo-realistic fine texture at 4x, where the recovered image lies on the natural-image manifold rather than at the blurry centroid of plausible solutions?

## The MSE-averaging diagnosis (the central insight)
- For an LR input, the set of HR images consistent with it is a manifold of equally-plausible sharp solutions. MSE penalizes pixel distance to the single ground truth; the MSE-minimizing estimate (the conditional mean E[HR|LR]) is the AVERAGE over that set → a smooth blur that may be far off the manifold (Fig 3: MSE solution = pixel-wise average, lies between/off the manifold; GAN solution = on the manifold).
- Pixel-wise losses cannot capture perceptual differences (texture) since defined on pixel differences.

## Two ideas to fix it
1. ADVERSARIAL LOSS (manifold pull): train a discriminator D to tell real HR from super-resolved SR; train generator G to fool it. The discriminator's "is this a real photo?" signal pushes G's outputs toward the natural-image manifold — toward ONE sharp plausible solution, not the average. (Goodfellow 2014 GAN; manifold idea from Mathieu 2015, Denton 2015 LAPGAN, Radford 2015 DCGAN.)
2. CONTENT LOSS in FEATURE SPACE not pixel space (perceptual/VGG loss): instead of pixel MSE, compare feature maps of a pre-trained VGG19 network — these are more invariant to pixel-space changes, so matching them allows texture to differ while content matches. (Gatys 2015, Johnson 2016, Bruna 2016.)

## The perceptual loss (THE objective)
l^SR = content loss + 1e-3 * adversarial loss.
   l^SR = l^SR_X + 1e-3 * l^SR_Gen.

Content loss choices:
- MSE content loss: l^SR_MSE = (1/(r^2 W H)) Σ_{x=1}^{rW} Σ_{y=1}^{rH} (I^HR_{x,y} - G(I^LR)_{x,y})^2. (r = upscale factor.)
- VGG content loss: l^SR_VGG/i.j = (1/(W_{i,j} H_{i,j})) Σ_x Σ_y (φ_{i,j}(I^HR)_{x,y} - φ_{i,j}(G(I^LR))_{x,y})^2.
  φ_{i,j} = feature map from j-th convolution (AFTER activation) before the i-th maxpooling layer of VGG19.
  Variants: VGG22 (φ_{2,2}, low-level features) and VGG54 (φ_{5,4}, high-level features). VGG54 = the final "SRGAN".

Adversarial (generative) loss: l^SR_Gen = Σ_{n=1}^N -log D(G(I^LR)).
  NON-SATURATING form: minimize -log D(G(I^LR)) instead of log(1 - D(G(I^LR))) for better gradient behavior (Goodfellow 2014).

Adversarial game (eq minmax): min_θG max_θD E_{I^HR~p_train}[log D(I^HR)] + E_{I^LR~p_G}[log(1 - D(G(I^LR)))].

## Generator architecture (SRResNet, B=16; from text + figure + canonical code)
- First: Conv k9 n64 s1 + PReLU.
- B=16 identical residual blocks; each: Conv k3 n64 s1 → BN → PReLU → Conv k3 n64 s1 → BN → elementwise sum with block input. (Block layout from Gross & Wilber 2016, via Johnson 2016.)
- After residual blocks: Conv k3 n64 s1 → BN, then a BIG SKIP CONNECTION: elementwise add the output of the first conv (pre-residual-blocks) — skip over all residual blocks.
- Upsampling: 2 sub-pixel convolution (PixelShuffle) layers, each ×2 → total ×4. Each: Conv k3 n256 s1 → PixelShuffle(×2) → PReLU. (Sub-pixel/ESPCN, Shi 2016: learn upscaling filters in LR space, rearrange channels to spatial — efficient, avoids LR-space bicubic pre-upsample.)
- Final: Conv k9 n3 s1 → Tanh (HR in [-1,1]).
- Why PReLU: learnable negative slope; the paper uses ParametricReLU (He 2015) in the generator.
- Why ResNet/skip: skip-connections relieve the net from modeling the trivial identity, enabling very deep generators (he2015deep, he2016identity). Found deeper beneficial (unlike SRCNN/Dong).

## Discriminator architecture (DCGAN-guided)
- 8 conv layers, 3x3 kernels, channels 64→128→256→512 doubling, increasing as in VGG.
- Strided convolutions reduce resolution each time #features doubles; NO max-pooling (DCGAN guideline, Radford 2015).
- LeakyReLU(α=0.2) throughout.
- Then two dense layers + final sigmoid → P(real). (Canonical code uses BCEWithLogits, sigmoid folded in.)
- BN after conv layers (except first), DCGAN-style.

## Training details (grounded, line 512-517)
- Data: 350k random ImageNet images. LR via bicubic downsample r=4 (Gaussian filter then downsample per method section). 16 random 96x96 HR crops per minibatch. Fully convolutional → arbitrary test size.
- Scaling: LR input → [0,1]; HR → [-1,1]; MSE computed in [-1,1]. VGG feature maps rescaled by 1/12.75 (≈ multiply eq by ≈0.006) to match MSE scale.
- Adam, β1=0.9. SRResNet: lr 1e-4, 1e6 iterations. SRGAN: pre-init generator with the trained MSE SRResNet (avoid bad local optima); then 1e5 iters at lr 1e-4 then 1e5 at lr 1e-5.
- Alternate G/D updates, k=1.
- Test time: BN in eval mode (deterministic on input).
- Theano + Lasagne.

## Baselines (prior art to elaborate)
- Bicubic / Lanczos interpolation: fast, oversmooth.
- Example/patch methods, sparse coding, neighborhood embedding, self-exemplars (Glasner 2009, Huang 2015), regression (RF, GP).
- SRCNN (Dong 2014/2016): 3-layer CNN, bicubic-preupsample input, MSE. Gap: shallow, MSE-smooth.
- ESPCN (Shi 2016): sub-pixel conv learns upscaling in LR space, fast. Gap: MSE objective.
- DRCN (Kim 2016): deeply-recursive CNN, long-range deps, few params. MSE.
- Johnson 2016 / Bruna 2016: VGG feature-space (perceptual) loss for SR & style transfer. Gap: no adversarial term → still not photo-realistic textures; closest ancestors.
- GAN (Goodfellow 2014), DCGAN (Radford 2015): adversarial framework + stable conv arch guidelines.
- VGG19 (Simonyan & Zisserman 2014): the fixed feature extractor for content loss.
- ResNet (He 2015/2016): residual blocks/skip; BatchNorm (Ioffe 2015).

## Evaluation settings (no outcomes)
- Datasets: Set5, Set14, BSD100 (test of BSD300). All 4x.
- Metrics: PSNR [dB], SSIM on y-channel of center-cropped (4px border removed) images via daala. Known limitation: PSNR/SSIM are pixel-based, fail to capture perceptual quality.
- MOS test: 26 raters score 1-5; the perceptual yardstick the paper argues for. Wilcoxon signed-rank, p<0.05.

## Design-decision → why table
- Drop MSE-only objective → MSE = pixel-wise average of plausible HR → smooth, off-manifold, perceptually poor.
- Add adversarial loss → pulls solution onto natural-image manifold (one sharp plausible solution).
- Content loss in VGG feature space (not pixel) → feature maps invariant to pixel-shifts; lets texture vary while content matches.
- VGG54 (deep φ_{5,4}) chosen → higher-level features = content/abstraction, leave texture to the adversarial term; deeper layers more invariant to pixel space.
- weight 1e-3 on adversarial loss → keep adversarial subordinate; balance content fidelity vs manifold pull.
- VGG rescale 1/12.75 → put VGG loss on same scale as MSE.
- Generator = deep ResNet (B=16) + skip → enable very deep generator; skip avoids identity-mapping burden.
- PReLU in G → learnable negative slope.
- Sub-pixel conv (PixelShuffle) ×2 ×2 → learn upscaling in LR space; efficient, no bicubic pre-upsample.
- Tanh output, HR in [-1,1] → bounded output.
- Discriminator DCGAN-style: strided conv, no maxpool, LeakyReLU 0.2, BN → stable adversarial training (Radford 2015).
- Non-saturating -log D(G) generator loss → better gradients when D confident.
- Pre-train generator as MSE SRResNet then GAN → avoid undesirable local optima / high-freq artifacts.
- Adam β1=0.9 (NOT 0.5) → here generator pretrained + content loss dominates, so standard momentum fine.
```
