# PixelRNN / PixelCNN synthesis (grounding notes)

arXiv 1601.06759 — "Pixel Recurrent Neural Networks", van den Oord, Kalchbrenner, Kavukcuoglu (Google DeepMind), ICML 2016. Verified by title search.
Canonical impl: jzbontar/pixelcnn-pytorch (MaskedConv2d + 8-layer stack + 256-way softmax), grounds PixelCNN code.

## Pain point / research question
Estimate the distribution of natural images p(x) such that it is (a) tractable to compute the likelihood, (b) able to generate new images, and (c) expressive enough to capture the highly nonlinear, long-range, multimodal correlations in natural images. Want a model that is BOTH expressive AND tractable+scalable.

## Background / tools and where they fall short
- Latent-variable models (VAE: Kingma&Welling 2013, Rezende 2014): extract representations but inference is intractable/approximate (ELBO), which hinders exact likelihood; and they make independence assumptions in the latent structure.
- Autoregressive factorization (NADE - Larochelle&Murray 2011; fully visible sigmoid belief nets - Neal 1992, Bengio&Bengio 1999): cast the joint as a product of conditionals p(x)=prod p(x_i|x_<i). TRACTABLE exact likelihood, no latent. But to model the complex conditionals you need a very expressive sequence model.
- 2D LSTM for images/textures (Theis & Bethge 2015): promising on grayscale; a 2D LSTM from top-left to bottom-right handles long-range deps. Slow.
- RNNs (Graves handwriting, Sutskever char prediction, machine translation): great at hard sequence problems, compact shared parametrization of a series of conditionals.
- Prior pixel models used CONTINUOUS pixel distributions (Theis 2015, Uria 2013). Limits multimodality / needs a prior on shape.

## THE FRAMEWORK
Factorize joint over n x n image, pixels in raster order (row by row, left to right):
  p(x) = prod_{i=1}^{n^2} p(x_i | x_1,...,x_{i-1}).  (eq 1)
Each pixel has 3 color channels R,G,B. Condition channels in order R->G->B within a pixel:
  p(x_i|x_<i) = p(x_{i,R}|x_<i) p(x_{i,G}|x_<i, x_{i,R}) p(x_{i,B}|x_<i, x_{i,R}, x_{i,G}).  (eq 2)
KEY CHOICE: pixels as DISCRETE variables. Each channel takes one of 256 values; each conditional is a MULTINOMIAL via a softmax (256-way). vs continuous: discrete softmax is representationally simple, arbitrarily MULTIMODAL with no prior on shape, easier to learn, better performance. For MNIST (binary) use sigmoid.
During training/eval the conditionals are computed IN PARALLEL (teacher forcing: all true pixels known); generation is SEQUENTIAL (each sampled pixel fed back).

## The masking technique (makes the conditioning exact + parallel)
A network that preserves spatial resolution and outputs a conditional at each location must NOT let pixel i see itself or future pixels. Enforce with masked convolutions.
- Channels split into 3 groups (R,G,B). For R channel of current pixel: only pixels left+above. For G: also R of current pixel. For B: also R and G of current pixel.
- Mask A: first conv layer ONLY. Restricts to already-predicted neighbors AND already-predicted colors of current pixel (excludes connection from a color to ITSELF -> center pixel of its own channel is zeroed).
- Mask B: all subsequent layers. Relaxes A to also allow a color->itself connection (center pixel kept), since the first layer already removed direct self-access.
- Implement: zero out the corresponding weights in the input-to-state / conv layers (multiply weight by a 0/1 mask). In the simple grayscale impl, mask zeros (i) the whole rows below center, and (ii) in center row, columns at/after center (A) or strictly after center (B).

## Row LSTM (sect 4.1)
Unidirectional, processes image row by row top->bottom, computing a whole row at once with a 1D convolution. Receptive field = roughly TRIANGULAR above the pixel (does NOT capture full context — misses pixels far to the sides).
- input-to-state: a k x 1 convolution (k>=3) over the whole 2D input map, MASKED, producing 4h x n x n (four gate vectors per position). Computed once for the whole map.
- state-to-state: recurrent over rows. Given h_{i-1}, c_{i-1} (each h x n x 1):
   [o_i, f_i, i_i, g_i] = sigma( K^ss (*) h_{i-1} + K^is (*) x_i )
   c_i = f_i ⊙ c_{i-1} + i_i ⊙ g_i
   h_i = o_i ⊙ tanh(c_i)            (eq 3)
  where (*) = convolution, ⊙ = elementwise. sigma = logistic for o,f,i gates; tanh for content gate g. K^ss kernel size 3x1; K^is precomputed (the i-s conv). Each step computes the new state for an ENTIRE ROW.
  (Larger k = broader triangular context. Weight sharing => translation invariance along the row.)

## Diagonal BiLSTM (sect 4.2)
Designed to capture the ENTIRE available context for any image size, and to parallelize.
- Two directions, each scans diagonally from a top corner to the opposite bottom corner; each step computes the LSTM state along a DIAGONAL.
- SKEWING: offset each row by one position relative to the previous row -> map of size n x (2n-1). Makes diagonals into columns so convolutions apply along them.
- input-to-state: 1x1 conv K^is -> 4h x n x n.
- state-to-state: COLUMN-wise conv K^ss with kernel size 2x1 (eq 3 dynamics). Kernel >2x1 not useful (already global receptive field).
- Skew output back to n x n by removing offsets.
- Combine two directions: shift the RIGHT (second-direction) output map DOWN by one row, add to LEFT output map, to prevent seeing future pixels.
- Advantage: full dependency field; 2x1 kernel = minimal info per step => highly non-linear computation.

## Residual connections (sect 4.3)
Up to 12 LSTM layers. Residual connections from one LSTM layer to next: input map has 2h features; input-to-state reduces to h per gate; after recurrent layer, output upsampled back to 2h via 1x1 conv; add input map to output. Increases convergence speed, propagates signals. Also optional learnable skip connections layer->output. (vs depth-gating: no extra gates needed.)

## PixelCNN (sect 4.5)
Replace unbounded-but-sequential-to-compute LSTM with standard conv layers giving a BOUNDED but large receptive field, computed for all positions AT ONCE. Fully convolutional, preserves spatial resolution, NO pooling. Masks (A then B) avoid future context. 15 layers, h=128 (CIFAR/ImageNet). Faster than PixelRNN in training/eval (parallel), but generation still sequential. Tradeoff: bounded receptive field ("blind spot" not addressed in this version).

## Multi-Scale PixelRNN (sect 4.6)
An unconditional PixelRNN generates a small s x s subsampled image; one or more conditional PixelRNNs take the s x s image as extra input and generate n x n. Conditional net: each layer biased by an upsampled version of the small image. Upsample s x s -> c x n x n via deconv net; bias = 1x1 unmasked conv mapping c x n x n -> 4h x n x n added to the input-to-state map of each layer.

## Architecture table (single-scale)
First layer: 7x7 conv mask A. Then multiple residual blocks:
- PixelCNN: 3x3 conv mask B.
- Row LSTM: i-s 3x1 mask B; s-s 3x1 no mask.
- Diagonal BiLSTM: i-s 1x1 mask B; s-s 1x2 no mask.
Then ReLU + 1x1 conv mask B (2 layers); for CIFAR/ImageNet these have 1024 feature maps, MNIST 32.
Output: 256-way softmax per RGB color (natural images), or sigmoid (MNIST binary).
Hyperparams: MNIST Diagonal BiLSTM 7 layers h=16. CIFAR-10 Row & Diagonal BiLSTM 12 layers h=128; PixelCNN 15 layers h=128. ImageNet32 12-layer Row LSTM h=384. ImageNet64 4-layer Row LSTM h=512 (no residual).

## Canonical PixelCNN code (jzbontar, grayscale)
class MaskedConv2d(nn.Conv2d):
  __init__(mask_type,...): register_buffer mask = weight.clone(); fill_(1);
    mask[:,:, kH//2, kW//2 + (mask_type=='B'):] = 0   # center row, at/after center
    mask[:,:, kH//2 + 1:] = 0                          # all rows below center
  forward: self.weight.data *= self.mask; return super().forward(x)
net = Sequential(
  MaskedConv2d('A',1,fm,7,1,3,bias=False),BN,ReLU,
  7x [MaskedConv2d('B',fm,fm,7,1,3,bias=False),BN,ReLU],
  Conv2d(fm,256,1))
target = (input[:,0]*255).long(); loss = F.cross_entropy(net(input), target). Adam.

## Design-decision -> why
- Autoregressive factorization (not latent): exact tractable likelihood, no approximate inference, models full dependency structure with no independence assumptions.
- Discrete 256-way softmax (not continuous): arbitrary multimodality without shape prior; easier training; better likelihood.
- RGB channels conditioned R->G->B: within-pixel color dependencies preserved; matched by 3-group masking.
- Mask A (first layer) vs Mask B (rest): A removes self/future so no information leak about the predicted value; B re-allows self-connection in deeper layers (safe since layer 1 already blocked direct self-access) to keep capacity.
- Row LSTM 1D conv along rows: parallelize the input-to-state across the whole map; triangular field is cheap but incomplete.
- Diagonal BiLSTM skewing + 2x1 column conv: turns diagonals into columns so a tiny conv reaches the FULL context; 2x1 minimal => highly nonlinear; bidirectional+shift covers entire context without leaking future.
- Residual connections: train up to 12 layers, faster convergence, no extra gates.
- PixelCNN: trade unbounded receptive field for parallel all-position computation -> fastest to train; generation still sequential.
- Multi-scale: condition a high-res PixelRNN on a cheaply-generated low-res image to inject global structure.

NO unsourced facts. All hyperparams from "Specifications of Models" + Table.
