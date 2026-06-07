# VGG — Synthesis Notes (Phase 1.5)

## The pain point / research question
By 2013–2014 ConvNets dominate ImageNet (AlexNet won ILSVRC-2012). Everyone is tweaking the AlexNet
recipe to squeeze out accuracy. Two main lines of "improvement" by ILSVRC-2013 winners (ZF net, OverFeat):
(1) smaller receptive window + smaller stride in the FIRST conv layer; (2) dense/multi-scale train+test.
Both change *many* things at once. The unexplored axis is **depth itself** — does simply making the net
deeper (more weight layers) help, holding everything else fixed? The difficulty: you cannot cleanly add
depth to AlexNet because its layers use big filters (11x11/7x7) with big strides — adding such layers
blows up parameters and collapses resolution fast. So the question "is depth good?" is entangled with
"what filter size lets you add depth cleanly?"

## Load-bearing ancestors

### AlexNet (Krizhevsky, Sutskever, Hinton, NIPS 2012) — the thing being improved
- 8 weight layers: 5 conv + 3 FC. First conv = 96 filters of 11x11x3, stride 4. Then 5x5, then 3x3.
- ReLU nonlinearity (key speedup over tanh/sigmoid). Dropout 0.5 on FC layers. Local Response
  Normalization (LRN) after some conv layers. Max pooling (overlapping 3x3 stride 2).
- Trained with SGD+momentum 0.9, weight decay 5e-4, batch 128, data augmentation (crops + flips +
  PCA color jitter). 224/227 crops.
- Limitation for the depth question: the big stride-4 first layer throws away spatial info immediately;
  big filters are parameter-heavy. The architecture isn't a clean substrate for "just add more layers."

### ZF net (Zeiler & Fergus 2013) and OverFeat (Sermanet et al. 2014) — ILSVRC-2013 winners
- ZF: reduced AlexNet's first layer to 7x7 stride 2 (from 11x11 stride 4) → better. Showed via
  deconv visualization that finer first-layer sampling helps.
- OverFeat: 7x7 stride 2 first layer; introduced **dense evaluation** — convert FC layers to conv layers
  so the net becomes fully convolutional and can be applied to an arbitrary-size image in one pass,
  producing a spatial grid of class scores (efficient sliding window). Also multi-scale.
- Limitation: both still use relatively large first-layer filters; neither isolates depth.

### Network in Network (Lin, Chen, Yan, ICLR 2014) — source of the 1x1 conv idea
- A convolution is a generalized linear model over a local patch; only linearly separable patches are
  cleanly handled. NIN inserts a tiny MLP ("mlpconv") at each location = stacked 1x1 convolutions with
  ReLU between them, i.e. a learnable nonlinear cross-channel mixing at each spatial location.
- For VGG: a 1x1 conv = linear projection across channels; followed by ReLU it adds nonlinearity to the
  decision function WITHOUT enlarging the receptive field. This is config C's extra knob.

### Ciresan et al. 2011 — small filters, deep-ish, but small datasets (MNIST/NORB/traffic signs), not ILSVRC.
### Goodfellow et al. 2014 (street numbers) — 11 weight layers helped → depth signal.
### LeCun 1989 — backprop conv nets; the classical architecture VGG stays loyal to.
### Glorot & Bengio 2010 — Xavier init; relevant to the init problem of deep nets (found post-submission to remove need for pretraining).
### Caffe (Jia 2013) — the implementation substrate.

## The core derivations (must be lived inline in reasoning.md)

### 1. Stacking 3x3 convs = larger effective receptive field
- A single 3x3 conv (stride 1, pad 1) sees a 3x3 window. Stack two (no pooling between): a unit in
  layer 2 sees a 3x3 window of layer-1 units, each of which sees 3x3 of the input → input footprint is
  5x5 (3 + (3-1) = 5). Stack three → 7x7 (5 + 2). General rule: k stacked 3x3 layers → (2k+1)x(2k+1).
  So 3 layers ≡ one 7x7's reach; 5 layers ≡ 11x11. 3x3 is the smallest filter that still encodes
  left/right, up/down, center.

### 2. Parameter count: stack of 3x3 << one big filter
- Assume C input and C output channels throughout. One conv layer with f×f filters: f²·C² weights
  (ignoring bias). One 7x7 layer: 49 C². Three stacked 3x3 layers: 3·(3²·C²) = 3·9·C² = 27 C².
  27 vs 49 → the 7x7 has 49/27 ≈ 1.81× as many params, i.e. the single 7x7 needs **81% more** params.
  Equivalently the 3x3 stack is a regularized/factored 7x7 with nonlinearity injected between factors.
- Two 3x3 (= one 5x5): 2·9·C² = 18 C² vs 25 C² → 5x5 has 25/18 ≈ 1.39× → ~39% more.

### 3. More nonlinearity
- The 7x7 layer has ONE ReLU. The three-3x3 stack has THREE ReLUs → a more discriminative (more
  expressive, more nonlinear) decision function for the same receptive field. Empirical confirmation in
  the paper: a 5-layer net of 5x5 convs derived from net B (each 3x3 pair → one 5x5) had ~7% higher
  top-1 error than B on a center crop.

### 4. The architecture family A–E (config table)
- Fixed generic layout: input 224x224x3, mean-RGB subtracted. Stacks of conv3 (stride 1, pad 1, so
  resolution preserved) separated by 5 max-pools (2x2 stride 2 → /2 each, total /32: 224→7). Width
  starts 64, doubles after each pool: 64→128→256→512→512 (capped at 512). Then FC-4096, FC-4096,
  FC-1000, softmax. ReLU everywhere. No LRN (A-LRN shows it doesn't help, costs memory/time).
- A: 11 wt layers (8 conv+3 FC). B: 13 (adds a conv per of first two blocks). C: 16 (adds a 1x1 conv in
  last three blocks). D: 16 (those extra convs are 3x3 not 1x1). E: 19 (two extra 3x3 in last three blocks).
- Param counts: A/A-LRN 133M, B 133M, C 134M, D 138M, E 144M. Depth grows but params stay ~flat because
  the conv stacks are cheap (most params live in the first FC: 512·7·7·4096 ≈ 102.7M).
- C vs D: C is better than B (1x1 adds nonlinearity → helps) but worse than D (3x3 also captures spatial
  context → better). So nonlinearity helps, but spatial receptive field matters more.

### 5. Training details
- Multinomial logistic regression (softmax CE) objective, mini-batch SGD, batch 256, momentum 0.9,
  weight decay L2 5e-4, dropout 0.5 on first two FC. LR init 1e-2, /10 when val acc plateaus (3 times
  total), stop at 370K iters (74 epochs).
- Init problem: deep nets stall with bad init. Solution at the time: train shallow A with random init
  (N(0, 1e-2 variance), biases 0); then init first 4 conv + last 3 FC of deeper nets from A, rest random.
  Later found Glorot init removes the need for this pre-training.
- Input: random 224 crop from isotropically rescaled image (smallest side S), per SGD iter; random
  horizontal flip + RGB color shift.

### 6. Multi-scale training (scale jittering)
- S = smallest side of rescaled training image (training scale), S ≥ 224. Single-scale: fix S=256 or
  S=384 (train 384 by fine-tuning from 256 with smaller LR 1e-3). Multi-scale: sample S uniformly from
  [S_min, S_max] = [256, 512] per image → one model robust to a range of object sizes; train by
  fine-tuning all layers of the S=384 model. This is train-set augmentation by scale.

### 7. Dense / fully-convolutional testing + multi-scale testing
- Test scale Q (smallest side), not necessarily = S. Convert FC layers to conv: first FC → 7x7 conv
  (because last feature map is 7x7), last two FC → 1x1 conv. Net is now fully convolutional → apply to
  whole uncropped image → class score map (HxWx1000), spatially average (sum-pool) → fixed 1000-vector.
  Average with horizontally-flipped image. No need to sample many crops (one pass, efficient).
- Multi-crop is complementary (different boundary/padding conditions: crop pads with zeros, dense gets
  real-image context at borders) — averaging dense+multicrop helps a bit. 50 crops/scale (5x5 grid x2 flips).
- Multi-scale test: run several Q and average posteriors. For fixed S use Q={S-32,S,S+32}; for jittered
  S use Q={S_min, 0.5(S_min+S_max), S_max}.

## Design-decision → why table
- 3x3 filters everywhere: smallest filter with directional notion; stacking gives any receptive field
  with fewer params + more nonlinearity; lets depth grow cleanly. Alt (big 7x7/11x11 first layer): more
  params, fewer nonlinearities, can't stack deep.
- stride 1, pad 1 on conv: preserve spatial resolution so depth doesn't collapse the map; only pooling
  downsamples. Alt (stride>1): loses info early (the AlexNet problem ZF already flagged).
- 5 max-pools 2x2/2: controlled /32 downsample (224→7), standard. Width doubling after each pool:
  keeps compute roughly balanced as resolution halves.
- 1x1 conv (config C): add nonlinearity without changing receptive field. Alt (3x3, config D): better,
  because it ALSO adds spatial context — so D > C.
- No LRN: A-LRN ≈ A, LRN costs memory/time → drop it. (Reacts against AlexNet's LRN.)
- ReLU: fast, non-saturating (from AlexNet).
- Dropout on FC, weight decay 5e-4: regularize the huge FC params.
- Pre-init from shallow A: deep nets stall on bad init; warm-start stabilizes gradient. Alt: Glorot init
  (found later to also work from scratch).
- Multi-scale train/test: objects appear at many scales; jitter = scale augmentation; dense multi-scale
  test aggregates more decisions. Alt (single fixed scale): worse, less robust.
- Dense fully-conv test (from OverFeat): efficient whole-image eval, captures border context. Multi-crop
  complementary via different padding.

## Canonical implementation (code/vgg_torchvision.py)
- `cfgs` dict maps A/B/D/E → list of channel ints and 'M' (maxpool). `make_layers` builds conv3(pad1)+ReLU
  stacks, in_channels updated, 'M' → MaxPool2d(2,2). `VGG` module = features + AdaptiveAvgPool2d(7,7) +
  classifier (Linear 512*7*7→4096, ReLU, Dropout, Linear 4096→4096, ReLU, Dropout, Linear 4096→1000).
  Init: conv kaiming_normal (orig: N(0,1e-2)), linear N(0,0.01) bias 0. (Config C with 1x1 not in
  torchvision; I'll include it to match the paper.)
