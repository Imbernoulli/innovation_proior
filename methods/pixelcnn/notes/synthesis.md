# PixelRNN / PixelCNN — synthesis notes (Phase 1.5)

## The pain point / research question
Model p(x) over natural images so that (a) we can compute exact likelihood of held-out images
(tractable density), (b) we can sample new images, and (c) it scales to large datasets. The
expressive/tractable/scalable trade-off is the whole game.

- Latent-variable models (VAE 2013/14, DBM, DBN, DLGM, DRAW) extract latents but the marginal
  p(x) = ∫ p(x|z)p(z) dz is intractable; training maximizes an ELBO, not the true likelihood, and
  evaluation needs estimators (AIS) or bounds. So you never have an exact, comparable likelihood,
  and the latent bottleneck imposes conditional-independence assumptions across pixels.
- Undirected models (RBM/DBM) have an intractable partition function Z; exact likelihood unavailable.
- Continuous-density autoregressive models (RIDE/spatial-LSTM Theis&Bethge 2015, RNADE Uria 2013,
  Deep GMM van den Oord 2014) get exact likelihood via the chain rule but assume a continuous
  parametric conditional (Gaussian / mixture of Gaussians / GSM). Problems: (1) you must pick a
  functional form; a mixture is unimodal-ish unless you add many components; (2) probability mass
  leaks outside [0,255]; (3) pixel data is really discrete (0..255 integers), so a density on a
  continuum is a modeling mismatch.

## The tractable handle: chain rule
For ANY joint, p(x) = Π_i p(x_i | x_{<i}). No approximation — it's exact by repeated conditioning.
Choose a fixed ordering: raster scan (row by row, left to right). Then x_1..x_{n^2}, and
  p(x) = Π_{i=1}^{n^2} p(x_i | x_1,...,x_{i-1}).
This converts density estimation into a sequence-of-classifiers problem (this is exactly the
fully-visible Bayes net of Neal 1992 / Bengio&Bengio 2000 and NADE of Larochelle&Murray 2011 —
NADE ties weights across the conditionals with a single shared hidden layer). MADE (Germain 2015)
realizes the same idea inside a single autoencoder by MASKING the weight matrices so output i only
connects to inputs <i — one forward pass gives all conditionals, trained in parallel.

### RGB sub-pixel factorization
Each pixel has 3 channels. Treat them as 3 sub-steps in the order R,G,B:
  p(x_i | x_{<i}) = p(x_{i,R}|x_{<i}) · p(x_{i,G}|x_{<i}, x_{i,R}) · p(x_{i,B}|x_{<i}, x_{i,R}, x_{i,G}).
G sees R of the same pixel; B sees R and G. This keeps full color dependence inside a pixel.

### Discrete output, not continuous
Model each conditional as a 256-way softmax (multinomial) over the integer intensities 0..255.
- arbitrarily multimodal, no shape prior, no leakage outside [0,255].
- no continuity assumption between 51 and 52; the net rediscovers ordinal structure (it learns
  smooth/peaked/long-tailed shapes itself, and puts extra mass on 0 and 255 which are frequent).
- empirically beats a continuous mixture (MCGSM) head: Row LSTM 3.06 vs 3.22 bits/dim on CIFAR-10.
- Loss = sum of per-pixel (per-channel) cross-entropies = exact negative log-likelihood. Trainable
  fully in parallel (all conditionals at once) because targets are the known input pixels;
  generation is sequential (sample x_i, feed back).

## Making a CONV net respect the ordering — masked convolutions
A conv at position i mixes a whole neighborhood, including pixels at/after i → it would peek at the
answer. Fix: zero out the kernel weights that touch x_i and any x_j with j ≥ i.

For a kH×kW kernel centered at the output position, in raster order "allowed" = strictly-above rows
+ same row strictly-left. So:
  - all rows BELOW center → 0
  - center row, columns at/after center → 0 (and the center itself handled by mask type)
- **Mask A** (first layer only): also zeroes the CENTER weight → x_i cannot see its own value.
  Required because at the input layer the center "pixel" is the actual value being predicted.
- **Mask B** (all later layers): KEEPS the center weight → a feature at position i may read the
  position-i feature of the previous layer (which, by induction, only summarizes x_{<i}). Without
  B you'd needlessly shrink the receptive field by one each layer.

### RGB-aware masking inside the channel dimension
Split the feature channels into 3 groups (R,G,B). The center-pixel connectivity is then a
within-pixel triangular rule:
  - mask A center: R←(nothing of itself), G←R, B←R,G  (no self-connections; group g reads only groups <g)
  - mask B center: R←R, G←R,G, B←R,G,B               (self allowed; group g reads groups ≤g)
Off-center spatial weights are fully on (subject to the above-row/left rule). This is what enforces
the R→G→B sub-pixel order in eq for p(x_i|x_<i).

(jzbontar's minimal MNIST code is single-channel, so it uses the scalar version:
  mask.fill_(1); mask[:,:,kH//2, kW//2 + (type=='B'):] = 0; mask[:,:,kH//2+1:] = 0 — exactly the
  above-row/left rule with A excluding / B including the center.)

## PixelRNN: two 2-D recurrent layers (unbounded receptive field)
Conv stacks have a BOUNDED receptive field (grows linearly with depth). RNNs give an unbounded one.
Two designs that parallelize a 2-D LSTM:

### Row LSTM
Process row by row, top→bottom. For a whole row at once: a k×1 (k≥3) MASKED convolution computes the
input-to-state contribution for all positions in parallel (4h channels = the 4 gates). The
state-to-state is a k×1 conv of the previous row's hidden state h_{i-1}. LSTM recurrence:
  [o,f,i,g] = activation( K^ss ⊛ h_{i-1} + K^is ⊛ x_i )   (σ for o,f,i ; tanh for g)
  c_i = f ⊙ c_{i-1} + i ⊙ g
  h_i = o ⊙ tanh(c_i)
Receptive field = a TRIANGLE above the pixel (a k-wide cone), because each row only pulls a k-wide
window from the row above. Misses the pixels far to the side on the same/earlier rows. Fast (n
sequential steps over rows) but context is occluded.

### Diagonal BiLSTM
Want the FULL context (everything left+above). Scan along diagonals. Trick: SKEW the map — offset
row j to the right by j positions → an n×(2n-1) map. Now a diagonal of the original becomes a
COLUMN of the skewed map, so a single column-wise 2×1 conv state-to-state propagates along the
diagonal and parallelizes over the whole diagonal. input-to-state is a 1×1 conv (4h). Compute one
direction (top-left→bottom-right), skew back. The second direction (top-right→bottom-left) is the
same on a flipped map; to keep causality the right map is shifted DOWN by one row before being added
to the left map (so it contributes only strictly-previous pixels). Two directions together → each
pixel sees the ENTIRE valid context (full receptive field) for any image size. Kernel 2×1 is minimal
and maximally nonlinear; bigger kernels don't help since the field is already global.

CIFAR-10 ordering of likelihood tracks receptive field: Diagonal BiLSTM (global) 3.00 <
Row LSTM (triangle) 3.07 < PixelCNN (bounded box) 3.14 bits/dim → "capturing a large receptive
field matters."

## Residual / skip connections
Train up to 12 LSTM layers. Use residual connections around each LSTM layer (He 2015) to speed
convergence and propagate signal. In a PixelRNN block: input has 2h features; input-to-state reduces
to h per gate; after recurrence, a 1×1 conv upsamples back to 2h and the input is added. Also
optional layer-to-output skip connections. Ablation (12-layer Row LSTM, CIFAR-10 val):
  no-res/no-skip 3.22, skip-only 3.09, res-only 3.07, both 3.06. Depth helps monotonically:
  1→3.30, 2→3.20, 3→3.17, 6→3.09, 9→3.08, 12→3.06.

## PixelCNN (the fast, fully-convolutional variant)
Drop recurrence. Stack masked convs (first layer 7×7 mask A; then 3×3 mask B), preserve spatial
resolution (no pooling), end with ReLU + 1×1 convs (mask B) then 256-way softmax per channel. Bounded
receptive field but everything computes in ONE parallel pass at train/eval time (generation still
sequential). 15 layers, h=128 on CIFAR-10.

## The BLIND SPOT (limitation of the masked-conv PixelCNN)
Stacking the standard above-row/left mask does NOT actually give the full "left+above" context.
Walk it: the allowed region of one masked conv is a flat-topped trapezoid (the rows above, plus the
left part of the current row). Compose L of them and the reachable region grows, but a triangular
wedge of pixels just to the upper-RIGHT of the current pixel is never reached — they're above the
current pixel (so they're legitimately in x_{<i}) yet they fall outside the cone that the stacked
left-leaning masks can propagate. That wedge is the BLIND SPOT: information the model is ALLOWED to
use but architecturally CANNOT see. It arises precisely because the single mask is left-biased and
the only way conv composition carries information rightward-and-up is blocked by the same-row mask.
Row LSTM has a (different) triangular blind region for the same reason; Diagonal BiLSTM has none
(its two skewed scans cover everything).

### Fix (two-stream gated PixelCNN)
Split into a VERTICAL stack (sees all rows strictly above — an unmasked conv on rows above,
implemented as a (k//2+1)×1 conv with top padding then crop/shift down by one so it never includes
the current row) and a HORIZONTAL stack (sees the current row strictly to the left — a 1×(k//2+1)
conv). The vertical stack feeds INTO the horizontal stack (a 1×1 "link") but not vice versa, so
causality holds and together they cover the whole valid context with no blind spot. Replace ReLU
with a GATED activation tanh(W_f * x) ⊙ σ(W_g * x) (mimics LSTM gates, more expressive). Residual
connection on the horizontal stack only (and NOT on the causal input layer, else future leak). This
two-stream/gated design is what removes the blind spot while staying convolutional and parallel.

## Code grounding
- jzbontar/pixelcnn-pytorch: the canonical ~90-line MNIST PixelCNN; exact MaskedConv2d A/B. (code/pixelcnn_jzbontar.py)
- EugenHotaj/pytorch-generative GatedPixelCNN: two-stream vstack/hstack + GatedActivation, blind-spot
  fix via (k//2+1) convs with padding+crop. (code/gated_pixel_cnn.py)

## Evaluation settings (pre-method facts)
- MNIST (binarized, Salakhutdinov&Murray 2008) — NLL in nats.
- CIFAR-10 (Krizhevsky 2009) — bits/dim = NLL / (32·32·3).
- ImageNet 32×32 and 64×64 — bits/dim. Dequantize continuous baselines with uniform [0,1] noise so
  discrete and continuous likelihoods are comparable (Theis 2015 note); a discrete model corresponds
  to a piecewise-uniform density with the same likelihood on noised data. Optimizer RMSProp; batch 16
  for MNIST/CIFAR, larger for ImageNet; scale+center input only.
</content>
</invoke>
