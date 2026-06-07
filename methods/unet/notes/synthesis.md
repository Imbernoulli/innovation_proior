# U-Net — Synthesis Notes (Phase 1.5)

## The pain point (problem setup, ~2015)
- Deep CNNs (AlexNet 2012, VGG 2014) crushed *classification*: image → one label. Their typical design throws away spatial resolution (pooling + fully connected head) precisely because classification only needs "what", not "where".
- Many real tasks — especially **biomedical** — need a label *per pixel* (segmentation): "where". Two extra hardships specific to biomedicine:
  1. **Tiny labeled corpora.** Thousands of annotated images are out of reach; expert pixel annotation is expensive. Often only ~30 images.
  2. **Touching objects of the same class** (cells packed against each other) must be separated into instances — a thin background ridge between them must be predicted, which a frequency-balanced pixel loss ignores.

## Baseline 1 — Ciresan et al. 2012 (sliding-window patch CNN)
- Core idea: to classify pixel x, feed the CNN a square **patch centered on x**; output is the class of the center pixel. Slide this window over the whole image → a label per pixel.
- Two virtues that made it win ISBI 2012 EM:
  - It **localizes** (gives per-pixel output), unlike a plain classifier.
  - The number of **training patches** ≫ number of images, so scarce-image regime is partly mitigated.
- Two drawbacks (explicitly the springboard):
  - **Slow + hugely redundant.** Run the full net once per pixel; adjacent patches overlap almost entirely, so the same convolutions are recomputed millions of times. A 512×512 image = 262k forward passes.
  - **Localization-vs-context tradeoff governed by patch size.** Big patch ⇒ more max-pooling ⇒ more context but coarser localization (pooling discards where). Small patch ⇒ sharp localization but little context. You cannot have both with a single patch size.
- Seyedhosseini 2013 / Hariharan "hypercolumns" 2014: use features from *multiple layers* in the classifier to get context + localization together — points in the right direction but still bolted onto a classifier.

## Baseline 2 — FCN, Long/Shelhamer/Darrell 2014 (the direct parent)
- Insight: a classification CNN, with its FC layers reinterpreted as 1×1 convolutions, is already a **fully convolutional** net — it maps an arbitrary-size image to a coarse spatial grid of class scores in **one** forward pass (no per-patch redundancy). This fixes drawback #1 of Ciresan.
- But the output grid is coarse (downsampled by the pooling stride, e.g. 32×). To get dense pixel output they **upsample inside the network** with a *backwards-strided convolution* ("deconvolution" / transposed conv) — a learnable upsampling layer, initialized to bilinear.
- Upsampling alone is blurry (semantic-but-coarse). Fix: **skip connections** — fuse the upsampled coarse prediction with finer feature maps from shallower pooling stages, by **summing** score maps (FCN-32s → 16s → 8s, fusing pool4 and pool3). Coarse-deep = "what", shallow-fine = "where".
- Limitations U-Net reacts to:
  - FCN's expansive side is thin — upsampling layers carry few channels; coarse semantic context is not richly propagated up to high resolution.
  - Fusion is by **summation** of class-score maps at a couple of stages, not a full symmetric decoder.
  - FCN was built for natural images with large datasets (PASCAL); not tuned for the few-image biomedical regime.

## Load-bearing background concepts
- **Max-pooling discards localization**: pooling builds translation-tolerant, larger-receptive-field features (good context) but you lose where within the pooled window the feature was. This is the root of the context/localization tension.
- **Transposed convolution / "up-convolution"**: a learnable upsampling. In U-Net it's a 2×2 up-conv with stride 2 that doubles spatial size and *halves* channels.
- **Data augmentation as a prior**: if labeled data is scarce, synthesize plausible variations so the net learns the right invariances. Dosovitskiy 2014 showed augmentation alone teaches strong invariances in unsupervised feature learning. For tissue, the dominant natural variation is **elastic deformation** — so simulate it.
- **He init (2015)**: with ReLU, draw weights ~ N(0, 2/N), N = fan-in, so each feature map keeps ~unit variance through the deep net; otherwise some paths explode and others die. For 3×3 conv with 64 in-channels, N = 9·64 = 576.

## The U-Net method (what gets derived)
### Architecture
- **Contracting path** (encoder): repeated [3×3 conv (unpadded/valid) → ReLU] ×2, then 2×2 max-pool stride 2. **Double channels at each downsampling** (64→128→256→512→1024).
- **Expansive path** (decoder): each step = 2×2 up-conv (halves channels) → **concatenate** with the cropped feature map from the mirror contracting level → [3×3 conv → ReLU] ×2.
  - Key modification vs FCN: the decoder has **many channels** (symmetric to encoder) so context propagates to high resolution; fusion is **concatenation** (not summation), letting the following convs *learn* how to combine; whole thing is a symmetric **U shape**.
- Final layer: **1×1 conv** maps the 64-component feature vector at each pixel to K class scores.
- 23 conv layers total. **No fully-connected layers.**
- **Cropping** before each concat: valid convs shrink the map by 2 px per 3×3 conv, so the encoder feature map is larger than the decoder map at the same level; center-crop the encoder map to match.
- **Valid convolutions only** ⇒ every output pixel has full receptive-field support in the input (no fabricated padding at borders). Output is smaller than input by a fixed border.

### Tile arithmetic (worked)
- Each 3×3 valid conv removes a 1-px ring ⇒ −2 px per conv. Two convs per block ⇒ −4 px before each pool.
- Original: input **572×572** → after the U → output **388×388**. The output is the seamless interior.
- Constraint: choose input tile so that **every** 2×2 max-pool sees an even-sized map (so up/down sizes line up exactly for cropping).

### Overlap-tile strategy (for arbitrarily large images)
- Because output < input and only valid convs are used, you tile: to predict an output tile, feed the larger input tile that contains the needed context. Adjacent output tiles **abut seamlessly** (no border artifacts).
- Border of the *whole* image lacks context ⇒ **extrapolate by mirroring** the image across the edge to fill the missing input ring. This is what lets the valid-conv net segment image borders without inventing data.

### Weighted boundary loss (the touching-cells fix)
- Per-pixel soft-max over the K final channels: p_k(x) = exp(a_k(x)) / Σ_{k'} exp(a_{k'}(x)).
- Energy = weighted cross-entropy: **E = Σ_x w(x) · log p_{ℓ(x)}(x)** (ℓ(x) = true label; maximize, or equivalently minimize the negative).
- Weight map: **w(x) = w_c(x) + w_0 · exp( −(d_1(x)+d_2(x))² / (2σ²) )**
  - w_c(x): class-frequency balancing weight.
  - d_1(x): distance to border of nearest cell; d_2(x): distance to border of 2nd-nearest cell.
  - The Gaussian term spikes exactly in the **thin gap between two touching cells** (where both d_1 and d_2 are small) ⇒ forces the net to learn the separating ridge. w_0 = 10, σ ≈ 5 px.
  - Separation borders pre-computed via morphological operations on the instance ground truth.

### Training details
- Caffe SGD. **Batch = 1 image** (favor large input tiles over batch size to use GPU memory) ⇒ compensate with **high momentum 0.99** so many past samples shape each update.
- He init (std √(2/N)).
- Drop-out at the end of the contracting path = further implicit augmentation.

### Data augmentation (the few-image enabler)
- Need shift/rotation invariance + robustness to deformation + gray-value variation.
- **Random elastic deformation**: displacement vectors on a coarse 3×3 grid, sampled from N(0, 10 px), per-pixel displacements via bicubic interpolation. This is "the key concept" for training from very few images.

## Design-decision → why table
| Choice | Why this and not the alternative |
|---|---|
| Fully convolutional, one pass | Kills Ciresan's per-patch redundancy/slowness; whole image in one forward pass. |
| Encoder–decoder U (not classifier) | Need per-pixel output ("where"), not one label. |
| Symmetric **many-channel decoder** | FCN's thin decoder under-propagates context; rich decoder carries context to high res. |
| **Concatenate** skip features (vs FCN's sum) | Sum forces a fixed combination of scores; concat lets the next convs *learn* how to mix fine+coarse. |
| **Skip connections** at every level | Pooling destroyed localization; the mirror encoder map restores high-frequency "where". |
| Double channels per downsample | Standard CNN capacity scaling: as spatial res halves, channel count doubles to keep representational capacity. |
| **Valid (unpadded) convs** + cropping | Every output pixel has true support; no padded border garbage; enables seamless overlap-tiling. |
| **Overlap-tile + mirroring** | Lets a valid-conv net segment arbitrarily large images and their borders, limited only by memory not image size. |
| **Weighted CE with Gaussian boundary term** | Plain CE ignores the thin inter-cell gap (few pixels, swamped by class balance); the exp(−(d1+d2)²/2σ²) term up-weights exactly that gap → separates touching instances. |
| **Class-balance weight w_c** | Background ≫ membrane; without balancing the net predicts majority class. |
| **Elastic deformation aug** | Tissue's dominant real variation; teaches deformation invariance without more labels — the core trick for ~30 images. |
| **Batch=1 + momentum 0.99** | Big tiles use memory better than big batches; high momentum recovers a stable gradient estimate from many past single-image steps. |
| **He init √(2/N)** | Keeps unit variance through many ReLU layers + multiple paths; else some branches explode, others die. |
| **1×1 conv head** | Maps per-pixel feature vector → K class scores; no FC layer needed (keeps it fully convolutional / size-agnostic). |

## Canonical implementation grounding
- `code/Pytorch-UNet/` (milesial). Structure: `DoubleConv` ([conv→BN→ReLU]×2), `Down` (maxpool→DoubleConv), `Up` (upsample/transpose-conv → pad/crop → concat → DoubleConv), `OutConv` (1×1). Model wires inc/down1-4/up1-4/outc; channels 64→128→256→512→1024.
- Note: this modern variant uses **padded** 3×3 convs (so it pads instead of crops) and adds **BatchNorm** — neither in the 2015 original. My final code keeps the original's valid-conv + center-crop + no-BN design (so output<input), but mirrors this repo's module decomposition. Original optimizer = SGD momentum 0.99, batch 1; this repo uses RMSprop — I follow the original.
