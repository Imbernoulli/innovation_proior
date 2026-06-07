# ResNeXt — synthesis (grounded in arXiv 1611.05431 source + facebookresearch/ResNeXt lua + torchvision)

## Source
- arXiv 1611.05431 (verified). LaTeX read in full.
- Canonical code: facebookresearch/ResNeXt models/resnext.lua (form C grouped conv); torchvision resnet.py supports groups+base_width (resnext50_32x4d).

## Pain point / research question (in-frame, late 2016)
- Vision moved from feature engineering → network engineering. But designing architectures gets harder with growing #hyperparameters (width, filter sizes, strides) especially with many layers.
- Two design philosophies:
  - VGG/ResNet: stack building blocks of the SAME shape/topology. Simple rule → fewer free hyperparameters, depth exposed as essential dimension; reduces risk of over-adapting hyperparameters to a dataset; robust across tasks.
  - Inception: carefully designed topologies → compelling accuracy at low theoretical complexity, via split-transform-merge (split input into low-dim embeddings by 1x1, transform by specialized filters 3x3/5x5, merge by concatenation). BUT complicating factors: filter numbers/sizes tailored per transformation, modules customized stage-by-stage → unclear how to adapt to new datasets/tasks.
- Goal: a simple architecture that adopts VGG/ResNet's repeat-same-block strategy WHILE exploiting split-transform-merge in an easy, extensible way. Increase accuracy while MAINTAINING (or reducing) complexity — rare in the literature.

## Key idea: aggregated transformations / cardinality
- Revisit the simple neuron = inner product Σ_{i=1}^D w_i x_i. This is itself splitting (slice x into single-dim subspaces x_i), transforming (scale w_i x_i), aggregating (Σ).
- Replace the elementary transformation w_i x_i with a generic function T_i(x) that is itself a network ("Network-in-Neuron"). Aggregated transformation:
  F(x) = Σ_{i=1}^C T_i(x)    (Eq. general)
- C = "cardinality" = size of the set of transformations. C is in the position of D in the inner product, but need not equal D and can be arbitrary. Width relates to #simple transformations; cardinality controls #more-complex transformations.
- Each T_i projects x into a (low-dim) embedding then transforms it. SIMPLE choice: all T_i have the SAME topology (extends VGG repeat-same-shape). Set T_i = bottleneck-shaped (1x1 reduce → 3x3 → 1x1 restore); the first 1x1 produces the low-dim embedding.
- Aggregated transformation as residual function:
  y = x + Σ_{i=1}^C T_i(x)    (Eq. resnext)

## Three equivalent forms (figure 3)
- (a) aggregated residual transformations: C separate bottleneck paths, each 1x1(256→d)→3x3(d→d)→1x1(d→256), summed, + identity.
- (b) early concatenation: C paths each 1x1(256→d)→3x3(d→d), concatenate the d-dim outputs into Cd, then ONE 1x1(Cd→256). Resembles Inception-ResNet but all paths same topology.
  - Proof of (a)≡(b): A_1 B_1 + A_2 B_2 = [A_1,A_2][B_1;B_2] (horiz concat of weights, vert concat of responses). A_i = weight of last layer, B_i = output of second-last layer. For C=2 the elementwise add in (a) = A_1B_1+A_2B_2; in (b) the last-layer weight is [A_1,A_2] and concatenated second-last outputs are [B_1;B_2]. So equal.
- (c) grouped convolution: replace all C low-dim-embedding 1x1 layers with a SINGLE wider 1x1 (256→Cd, e.g. 128-d); the 3x3 is a grouped conv with C groups (input/output Cd channels, each group operates on d=Cd/C channels); then 1x1 (Cd→256). Grouped conv does the splitting. Looks like original bottleneck but wider+sparsely connected.
- Reformulations give nontrivial topology only for block depth ≥3. Depth-2 block → trivially wide dense module.
- All three forms strictly equivalent (with BN/ReLU handled correctly). Train all three → same results. Implement (c) — more succinct + faster.

## Template & two rules (Sec Template) — from VGG/ResNet
- (i) blocks producing same-size spatial maps share same hyperparameters (width, filter sizes).
- (ii) each time spatial map downsampled by 2, block width ×2. → roughly same FLOPs per block on all stages.
- Only need to design a template module; rest determined.

## Architecture (Table 1: ResNeXt-50, 32×4d)
- conv1: 7×7, 64, stride 2 → 112×112.
- conv2 (56×56): 3×3 maxpool stride 2; then [1×1,128; 3×3,128,C=32; 1×1,256] ×3.
- conv3 (28×28): [1×1,256; 3×3,256,C=32; 1×1,512] ×4.
- conv4 (14×14): [1×1,512; 3×3,512,C=32; 1×1,1024] ×6.
- conv5 (7×7): [1×1,1024; 3×3,1024,C=32; 1×1,2048] ×3.
- global avg pool → 1000-d fc → softmax.
- 25.0M params, 4.2 GFLOPs (vs ResNet-50: 25.5M, 4.1 GFLOPs). Input/output width of template fixed 256-d (at conv2); widths double each subsample.

## Model capacity (Sec, Eq capacity)
- Original ResNet bottleneck: 256·64 + 3·3·64·64 + 64·256 ≈ 70k params (on a given map size).
- ResNeXt template with bottleneck width d, cardinality C: C·(256·d + 3·3·d·d + d·256) params and proportional FLOPs.
- C=32, d=4 → ≈ 70k (preserves complexity).
- Table 2 (conv2 template, ~70k params, ~0.22 GFLOPs = #params×56×56):
  | C | 1 | 2 | 4 | 8 | 32 |
  | d (bottleneck width) | 64 | 40 | 24 | 14 | 4 |
  | width of group conv (Cd) | 64 | 80 | 96 | 112 | 128 |
- To vary C at preserved complexity, adjust bottleneck width d (it's isolated from block input/output) — no change to depth or block I/O width.

## Implementation details (Sec)
- Follows He 2016 + fb.resnet.torch (Gross 2016). Input 224×224 random crop from resized image; scale+aspect-ratio augmentation (Szegedy GoogLeNet style).
- Shortcuts: identity, except dimension-increasing ones = projections (type B in He 2016).
- Downsampling of conv3,4,5: stride-2 in the 3×3 layer of the first block of each stage.
- SGD, minibatch 256 on 8 GPUs (32/GPU). Weight decay 1e-4, momentum 0.9. LR start 0.1, ÷10 three times. He 2015 weight init.
- Ablation eval: single 224×224 center crop from shorter-side-256 image.
- Implement form (c). BN right after the convs in (c). ReLU right after each BN, except block output where ReLU is after adding to shortcut (following He 2016). [For form (a): BN after aggregating transformations, before adding shortcut.]

## Diagnostic / motivating findings (context, about ResNet baseline + capacity dimensions)
- Increasing accuracy by increasing capacity (deeper/wider) is easy; increasing accuracy while maintaining/reducing complexity is rare.
- Depth/width give diminishing returns for existing models — this is the gap cardinality attacks.
- Veit et al. 2016: a single ResNet can be interpreted as an ensemble of shallower networks (from ResNet's additive behavior). But aggregating transformations is NOT ensembling — members trained jointly, not independently.
- Grouped convs date to AlexNet (Krizhevsky 2012), motivated by distributing model over 2 GPUs; an "engineering compromise". Little prior evidence of using grouped convs to IMPROVE accuracy. Channel-wise conv (groups=channels) part of separable convs (Sifre 2014).
- (The proposed ResNeXt's win numbers — 32×4d ResNeXt-50 22.2% vs ResNet-50 23.9%, lower train error too — are proposed-method results; keep OUT of context.md, may appear in reasoning only as "what I'd want to validate".)

## Load-bearing ancestors (baselines)
- VGG (Simonyan & Zisserman 2015): stack same-shape blocks; depth as essential dimension; simplicity → robustness.
- ResNet (He et al. 2016): residual y = x + F(x); bottleneck 1×1→3×3→1×1; two-branch (one = identity); the base being modified.
- Inception (Szegedy 2015 / Ioffe-Szegedy 2015 / Szegedy 2016): split-transform-merge; low theoretical complexity; but per-module hand customization. Inception-ResNet (Szegedy 2016): branch+concat in residual function.
- Grouped convolutions (Krizhevsky 2012): AlexNet 2-GPU split. Sifre 2014: separable/channel-wise.
- BatchNorm (Ioffe & Szegedy 2015). He init (He 2015).
- Network-in-Network (Lin 2014a): increases depth dimension (contrast: NiN-euron expands new dimension).

## Design-decision → why table
| Decision | Why this, not alternative |
|---|---|
| All T_i share the SAME topology | extends VGG repeat-same-shape; isolates #paths (cardinality) as a single clean factor; extensible to any C with no per-path design; vs Inception's per-branch customization which doesn't transfer to new datasets. |
| Aggregate by SUMMATION, set as residual y = x + ΣT_i | matches the inner-product's Σ aggregation; reuses ResNet's residual conditioning (identity shortcut); summation lets the C paths combine into one additive residual. |
| T_i = bottleneck (1×1 reduce → 3×3 → 1×1 restore) | first 1×1 makes the low-dim embedding (split-transform); needs depth ≥3 for the reformulations to be nontrivial; reuses ResNet's affordable bottleneck. |
| Expose cardinality C as a new dimension | C sits where D sits in the inner product but is free; controls #complex transformations; empirically more effective than depth/width when those give diminishing returns. |
| Vary bottleneck width d to hold complexity when changing C | d is isolated from block I/O (fixed 256-d), so changing it doesn't touch depth or block I/O width → clean controlled study of C. params = C·(256d + 9d² + 256d). |
| C=32, d=4 default | gives ≈70k params matching ResNet bottleneck; bottleneck width ≥4d (accuracy saturates as width shrinks below that — not worthwhile to keep trading width for C). |
| Implement form (c) grouped convolution | strictly equivalent to (a),(b) but more succinct + faster; grouped conv does the splitting natively. |
| Two template rules (same map→same width; halve map→double width) | from VGG/ResNet; equalizes per-block FLOPs across stages; narrows design space to the template. |
| Shortcut: identity, projection only when dims increase (type B) | from ResNet; keeps most skips free, projects only at stage transitions. |
| Downsample by stride-2 in the 3×3 of the first block per stage | fb.resnet.torch convention. |
| BN after each conv, ReLU after each BN except final (after add) | from ResNet; final ReLU on the sum so the residual branch can carry negative corrections. |
