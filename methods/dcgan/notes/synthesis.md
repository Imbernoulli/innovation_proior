# DCGAN synthesis (Phase 1.5)

Verified: arXiv 1511.06434, "Unsupervised Representation Learning with Deep Convolutional GANs", Radford, Metz, Chintala (ICLR 2016).
Canonical impl: pytorch/examples/dcgan/main.py (grounded, saved in code/). Original was Theano/torch (soumith/dcgan.torch).

## Pain point / research question
- Discriminative CNNs are booming (supervised). Unsupervised feature learning lags. Want to reuse the practically-unlimited unlabeled image data to learn reusable features.
- GANs (Goodfellow 2014) are an attractive ML-free generative model — no explicit density, no pixelwise MSE — but: (a) known to be UNSTABLE to train; naive CNN GANs produce nonsense or collapse; (b) it's a black box — nobody knows what multi-layer GANs learn.
- So: can we find a CNN architecture family that makes convolutional GANs train STABLY across datasets, and then reuse the learned features (discriminator) for supervised tasks?
- Historical attempts to scale GANs with CNNs failed — this is why LAPGAN (Denton 2015) went the Laplacian-pyramid route (chain of small models upscaling) instead of one CNN generator. We want one CNN that trains directly.

## The three CNN-architecture ingredients adopted/modified (the heart)
1. **All-convolutional net (Springenberg et al. 2014, "Striving for Simplicity").** Replace deterministic spatial pooling (maxpool) with *strided* convolutions, so the net learns its own spatial downsampling. DCGAN applies this in the discriminator (strided conv downsamples) AND in the generator (fractionally-strided conv UPsamples — learn own upsampling). Why: a fixed maxpool throws away spatial info with a hand-chosen operator; a strided conv is a learnable downsample. For the generator you need the transpose: a fractionally-strided ("transposed") convolution learns the upsample instead of a fixed unpool/nearest-neighbor. (Paper note: these are "in some recent papers wrongly called deconvolutions".)
2. **Eliminate fully-connected hidden layers on top of conv features.** Strongest example = global average pooling (used in NIN / Inceptionism-era SOTA classifiers). They tried GAP: increased stability but HURT convergence speed. Compromise: directly connect highest conv features to in/out. Generator's first layer takes uniform noise Z (100-dim) — it IS a matrix multiply (could be called FC) but the result is RESHAPED into a 4D tensor (small spatial extent, many feature maps) and used as the start of the conv stack. Discriminator: last conv layer is flattened → single sigmoid output. So: no FC *hidden* layers; the only "FC-like" things are the projection-reshape at the gen input and the flatten-to-sigmoid at the disc output.
3. **Batch Normalization (Ioffe & Szegedy 2015).** Normalizes input to each unit to zero mean/unit var. Critical to get deep generators to START learning, and prevents the classic GAN failure mode where the generator COLLAPSES all samples to a single point (mode collapse). BUT applying BN to ALL layers → sample oscillation / instability. Fix: do NOT apply BN to the generator OUTPUT layer and the discriminator INPUT layer. (Rationale: BN at the gen output forces the generated batch statistics, fighting the tanh range / data stats; BN at disc input normalizes away the actual real-vs-fake signal at the pixel level.)

## Activations (design decisions → why)
- Generator: ReLU everywhere EXCEPT output, which uses **Tanh**. Why Tanh out: a *bounded* activation lets the model learn quickly to saturate and cover the color space of the training distribution (images are pre-scaled to [-1,1], the tanh range).
- Discriminator: **LeakyReLU** (slope 0.2) for all layers. Found to work well especially for higher-resolution modeling. Contrast: original GAN used maxout. Why leaky: a plain ReLU in the discriminator can have dead units / kills gradient on the negative side; the generator depends on the discriminator's gradient w.r.t. the fake — if discriminator units are dead the generator gets no signal. LeakyReLU keeps a small gradient on the negative side → healthier gradient flow back to G.

## Training details (all grounded in tex line 104)
- Preprocessing: scale images to [-1,1] (tanh range). No other preprocessing.
- Minibatch SGD, batch size 128.
- Weights init: zero-centered Normal, std 0.02. (Canonical code: conv weights N(0,0.02); BN weight N(1,0.02), bias 0.)
- LeakyReLU leak slope = 0.2.
- Optimizer: **Adam** (Kingma & Ba 2014), not plain momentum SGD. lr=0.0002 (the suggested 0.001 was too high). β1=0.5 (the suggested 0.9 caused oscillation/instability; lowering to 0.5 stabilized). β2=0.999 (default; canonical code uses (0.5,0.999)).
- Latent Z: 100-dim uniform distribution.

## Architecture concrete (LSUN generator, fig 1; canonical 64x64 code)
Generator (nz=100 → 64x64x3): project+reshape z to 4x4x(ngf*8); then 4 fractionally-strided convs (k=4,s=2,p=1) doubling spatial / halving channels: 4→8→16→32→64; channels ngf*8→ngf*4→ngf*2→ngf→nc. BN+ReLU after each except output (Tanh, no BN). ngf=64.
Discriminator (mirror): Conv k=4 s=2 p=1: 64→32→16→8→4 spatial, channels nc→ndf→ndf*2→ndf*4→ndf*8; LeakyReLU(0.2) all; BN all except input layer; final Conv k=4 s=1 p=0 → 1 → Sigmoid. ndf=64.

## Objective (unchanged from GAN — this is reused, not invented here)
min_G max_D V = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1 - D(G(z)))]. Trained with BCE: D wants D(real)→1, D(fake)→0; G wants D(fake)→1 (non-saturating: maximize log D(G(z))). Canonical code trains D on real (label 1) + fake (label 0), then G with fake labelled 1.

## Representation-learning payoff (motivating but is method-result; keep evaluation outcomes out of reasoning except as forward intent)
- Use discriminator conv features (all layers, maxpool each to 4x4 grid, flatten+concat → 28672-d) + L2-SVM as a feature extractor for CIFAR-10/SVHN.
- Latent-space walks → smooth semantic transitions (no sharp jumps = not memorizing).
- Vector arithmetic in Z (à la word2vec, Mikolov 2013): average Z of 3 exemplars per concept, then arithmetic (smiling woman - neutral woman + neutral man → smiling man; "turn" vector for pose). Single-sample arithmetic unstable; averaging 3 stabilizes.
- Guided backprop (Springenberg) shows disc features fire on beds/windows; dropping "window" feature maps makes generator forget to draw windows → disentangled scene/object representation.

## Ancestors to elaborate in context (prior art)
- GAN (Goodfellow 2014): the adversarial framework + minimax objective. Gap: unstable, black-box, naive CNN versions fail.
- LAPGAN (Denton 2015): Laplacian pyramid of conditional GANs, upscale low-res step by step. Gap: objects look "wobbly" due to noise from chaining multiple models; doesn't reuse generator for supervised tasks.
- VAE (Kingma & Welling 2013): blurry samples.
- All-conv net (Springenberg 2014): strided conv replaces pooling.
- NIN / global average pooling (Lin et al; used in Inceptionism-era classifiers): remove FC.
- BatchNorm (Ioffe & Szegedy 2015).
- ReLU (Nair & Hinton 2010), LeakyReLU (Maas 2013, Xu 2015), maxout (Goodfellow 2013).
- Adam (Kingma & Ba 2014).
- word2vec arithmetic (Mikolov 2013).
- Zeiler & Fergus 2014 (deconv visualization), guided backprop (Springenberg 2014).

## Design-decision → why table (no holes)
- Strided conv (disc) / fractional-strided conv (gen) instead of pooling/unpooling → learn own down/up-sampling; fixed pool is a hand-chosen lossy operator.
- Remove FC hidden layers → GAP most stable but slow; compromise = project-reshape at gen in, flatten-sigmoid at disc out. Fewer params, more stable deep training.
- BN both nets → start learning + prevent mode collapse to a point.
- BN NOT on gen output / disc input → applying everywhere → oscillation; these two layers must see/produce true statistics (tanh-range output; raw pixel input).
- ReLU in gen → standard, good gradients for deep gen.
- Tanh gen output → bounded, saturates to cover color space, matches [-1,1] data scaling.
- LeakyReLU(0.2) in disc → avoid dead units / preserve negative-side gradient so G keeps getting signal; better at high res.
- Adam lr 2e-4 → 1e-3 too high (unstable). β1=0.5 → 0.9 oscillates; 0.5 stabilizes (GAN is a moving non-stationary target, lower momentum tracks it better).
- Weight init N(0,0.02) → small init for stable GAN start.
- Z uniform 100-d → low-dim latent manifold; uniform prior.
- batch 128, SGD/Adam minibatch.
```
