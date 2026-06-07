# CycleGAN — Synthesis (Phase 1.5)

## The pain point / research question
Image-to-image translation (grayscale→color, edges→photo, labels→photo, photo→painting, summer→winter,
horse→zebra) had become learnable end-to-end with **pix2pix** (Isola et al. 2016): a conditional GAN +
L1 supervised on **aligned pairs** {(x_i, y_i)}. But aligned pairs are expensive or impossible:
- semantic-segmentation datasets are tiny (Cityscapes),
- artistic stylization needs an artist to hand-author each output,
- object transfiguration (horse↔zebra) has **no well-defined ground-truth output** at all — you cannot
  photograph the same horse as a zebra.

Goal: learn G: X→Y from two **unpaired** sets {x_i}, {y_j} with no correspondence info, such that
translations are both (a) realistic in the target domain and (b) meaningfully tied to the specific input.

## Why an adversarial loss ALONE is under-constrained (the core derivation)
GAN theory (Goodfellow 2014): with enough capacity, training G against a discriminator D_Y drives the
distribution of G(X) to match p_data(Y). The minimax value with optimal D is
2·JSD(p_g ‖ p_data) − log4, minimized (=−log4) iff p_g = p_data. So adversarial training matches
**distributions**.

But matching the output *distribution* says nothing about the *pairing*. There are infinitely many maps G
that all push p_data(X) onto p_data(Y): compose any distribution-matching G with any permutation π of
target images that preserves p_data(Y) — G' = π∘G induces the *same* output distribution but a completely
different input→output correspondence. The objective cannot distinguish them. So x's content is not
preserved; an x can map to an arbitrary y.
Worse, in practice the degenerate solution the optimizer actually reaches is **mode collapse**: all x map
to a single highly-realistic y* — that single output is in-distribution, so D is fooled, gradient progress
stalls. Adversarial loss alone is both theoretically under-constrained and practically unstable.

## The fix: cycle consistency (insight)
Need extra structure that pins down the pairing without paired data. Borrow from human/machine translation
**back-translation** (Brislin 1970; He et al. 2016 dual learning) and from forward-backward consistency in
tracking/optical flow (Kalal 2010; Sundaram 2010) and CNN cycle-supervision (Zhou 2016; Godard 2016):
a good translation should be **invertible**. Introduce the inverse map F: Y→X and require
F(G(x)) ≈ x (forward cycle) and G(F(y)) ≈ y (backward cycle).

Why this kills the degeneracy: a permutation/collapsed G loses information about x, so no F can reconstruct
x from G(x). Requiring reconstruction forces G to be (approximately) injective and content-preserving.
Mode collapse is impossible: if many x's map to the same G(x), F cannot recover the distinct x's, so the
cycle loss is large.

### Cycle-consistency loss
L_cyc(G,F) = E_x[ ‖F(G(x)) − x‖_1 ] + E_y[ ‖G(F(y)) − y‖_1 ].
- **L1 not L2**: L1 produces less blurring than L2 for image reconstruction (well known from pix2pix; L2
  averages over plausible outputs → blur). Tried replacing L1 with an *adversarial* loss on (F(G(x)), x):
  no improvement, so keep the simpler L1.
- **Both directions, not one**: one-directional cycle leaves the other mapping unregularized → instability /
  mode collapse in the unregularized direction (ablation: GAN+backward fails). Symmetric coupling.

### Full objective
L(G,F,D_X,D_Y) = L_GAN(G,D_Y,X,Y) + L_GAN(F,D_X,Y,X) + λ·L_cyc(G,F),
solved as G*,F* = argmin_{G,F} max_{D_X,D_Y} L.
- λ = 10 — cycle term carries the content constraint; it must dominate so reconstruction is not sacrificed
  for adversarial realism, but not so large it freezes the translation. λ=10 with L1 (range ~[0,2] per
  pixel) vs GAN term ~O(1) makes cycle the primary signal.
- Interpretation: two coupled "autoencoders" F∘G: X→X and G∘F: Y→Y whose bottleneck is an image in the
  other domain, regularized adversarially to look like that domain — a special case of adversarial
  autoencoders (Makhzani 2015).

## Adversarial loss form
L_GAN(G,D_Y,X,Y) = E_y[log D_Y(y)] + E_x[log(1 − D_Y(G(x)))], min_G max_{D_Y}.
In practice replaced by **least-squares (LSGAN, Mao 2017)**:
- D minimizes E_y[(D(y)−1)^2] + E_x[(D(G(x)))^2]; G minimizes E_x[(D(G(x))−1)^2].
- Why: vanilla sigmoid-CE saturates — fake samples on the correct side of the decision boundary but far
  from the data give vanishing gradient to G. LSGAN penalizes by *distance* to the boundary, so even
  "classified-correct" fakes that are far from real get a gradient → more stable, sharper outputs.
  (LSGAN corresponds to minimizing a Pearson χ² divergence.)

## Stabilization tricks
- **Image buffer (history, Shrivastava 2016)**: update D on a buffer of 50 previously generated images, not
  only the current batch. Reduces oscillation — D can't overfit to the generator's latest single output;
  decorrelates D updates. (batch size = 1, so without history D sees one fake per step.)
- **Divide D objective by 2**: slows D relative to G so D doesn't overpower G early.

## Architecture (design + why)
Generator (from Johnson et al. 2016 fast-style-transfer / super-resolution):
c7s1-64, d128, d256, [6 or 9 × R256 ResNet blocks], u128, u64, c7s1-3, Tanh.
- 3 convs (1 stride-1 + 2 stride-2 downsample) → residual blocks at low res → 2 fractional-stride
  (transpose) convs upsample → 7×7 conv to RGB.
- **ResNet blocks (He 2016)**: translation = input + residual; identity is a good prior since output keeps
  most of the input's structure (geometry/layout), only appearance changes. Residual blocks make learning
  "mostly identity + small change" easy and trainable at depth.
- 6 blocks for 128², 9 blocks for 256²+ (more capacity / receptive field at higher res).
- **Instance normalization (Ulyanov 2016)** not batch norm: normalize per-image (per-sample, per-channel),
  removing instance-specific contrast/style statistics; output style should depend on the *target domain*,
  not the input image's contrast. Also batch size = 1 makes BatchNorm degenerate. IN ≈ style normalization.
- **Reflection padding** to reduce boundary artifacts.
- Tanh output → pixels in [−1, 1].

Discriminator: **70×70 PatchGAN** (from pix2pix): C64-C128-C256-C512 then 1-channel conv.
- 4×4 stride-2 conv, InstanceNorm (none on first C64), LeakyReLU(0.2).
- Classifies each 70×70 overlapping patch real/fake, averaged.
- Why patches: appearance/texture realism is local; a patch discriminator has far fewer params, models high
  frequencies (low-freq handled by cycle/L1 elsewhere), is fully convolutional → works on any image size.
- 70 chosen as the receptive field giving best texture quality vs artifact tradeoff (from pix2pix study).

## Identity loss (application-specific, painting→photo)
L_idt(G,F) = E_y[‖G(y) − y‖_1] + E_x[‖F(x) − x‖_1], weight 0.5λ.
- Motivation: without it G/F are free to shift the overall tint (e.g. Monet daytime → photo sunset) since
  such a shift is equally valid under GAN + cycle. Feeding a *target-domain* image into G should leave it
  unchanged (it's already in the target domain). Anchors the color/tint. (Adapted from Taigman 2016 DTN.)

## Training details
- Adam, lr 2e-4, β1=0.5, batch size 1.
- Constant lr first 100 epochs, linear decay to 0 over next 100.
- Weights N(0, 0.02).
- λ = 10; identity weight 0.5λ when used.

## Baselines (context.md)
- **pix2pix (Isola 2016)** — cGAN + L1 on *paired* data; the supervised upper bound; the thing being freed
  from pairing.
- **CoGAN (Liu 2016)** — two GANs with tied early/late weights → shared latent; translate by finding latent
  that makes x then rendering in Y. Gap: weight-sharing assumption, indirect translation.
- **SimGAN (Shrivastava 2016)** — adversarial + ‖x−G(x)‖_1 pixel regularizer; assumes input/output close in
  *pixel* space (only refinement). Gap: predefined pixel similarity; can't do large appearance change.
- **Feature-loss + GAN** — SimGAN variant with L1 in VGG feature space. Gap: predefined feature similarity.
- **BiGAN/ALI (Donahue/Dumoulin 2016)** — learn inverse encoder F: X→Z alongside G: Z→X. Repurposed for
  image→image. Gap: built for latent inversion, not domain translation pairing.
- **DTN/Taigman 2016, BousmalisDomain 2016** — share content features in a predefined metric/label space.
- **Neural style transfer (Gatys 2015)** — match Gram statistics of *one* style image; per-image, not a
  *collection*; needs a matched style exemplar.

All baselines either need pairs or assume a *predefined* similarity metric between input/output. CycleGAN
needs neither — only set-level supervision + invertibility.

## Canonical implementation
junyanz/pytorch-CycleGAN-and-pix2pix. Key files:
- models/cycle_gan_model.py — losses, two G's two D's, identity, buffers, optimize_parameters.
- models/networks.py — ResnetGenerator, NLayerDiscriminator (PatchGAN), GANLoss (lsgan=MSE), instance norm.
- util/image_pool.py — 50-image history buffer, 50% return-old/insert-current.
Naming: code G_A=G (X→Y), G_B=F (Y→X), D_A=D_Y, D_B=D_X. lambda_A=lambda_B=10, lambda_identity=0.5.

## Code-scaffold (pre-method) ↔ final correspondence
Pre-method known primitives: Conv/InstanceNorm/ReLU/ResBlock layers, Adam, L1Loss, MSELoss, a base GAN
training loop with a single generator + single PatchGAN discriminator (pix2pix-style). The empty slots are:
the *coupled* two-generator/two-discriminator structure, the cycle reconstruction term, the combined
objective. Scaffold = pix2pix-style harness with a generator stub, a discriminator stub, a GAN loss, and a
TODO "extra structure that ties output to input without pairs".
