# Context: fast small-filter convolution for convolutional networks

## Research question

Deep convolutional networks built almost entirely from small 3×3 convolution layers have become the state of the art for image recognition. Their cost is dominated by the convolution layers, and that cost is paid not only across the GPU-days of training but at every inference. Two regimes matter and both are awkward for the existing fast methods: very small batch sizes (a single image for low-latency detection; small per-node minibatches when training is sharded across a cluster, because large batches hurt convergence) and small filters (3×3 almost exclusively). The precise problem: compute a convolution layer — correlate K filters of C channels and size R×S against N images of C channels and size H×W — using far fewer multiplications per output than the R·S of direct convolution, while keeping the heavy stage a dense matrix multiply large enough to run efficiently *even when the batch is one*, and keeping the working memory small. A solution must reduce arithmetic for *small* filters (where the existing fast method does not help) without trading away the matrix-multiply structure that makes GPUs fast.

## Background

A convnet layer computes, for image i, filter k, position (x,y):
Y[i,k,x,y] = Σ_c Σ_v Σ_u D[i,c,x+u,y+v] · G[k,c,u,v], i.e. a sum over C channels of 2D correlations of an H×W image channel with an R×S filter. Direct evaluation spends r multiplications per output in 1D and R·S in 2D (3×3 ⇒ 9 multiplications per output pixel per channel). The arithmetic is regular but multiply-heavy.

A relevant classical result is the **arithmetic complexity of FIR filters**. Computing m outputs of an r-tap FIR filter is written F(m,r). It has been known since at least 1980 that the minimum number of *general* multiplications required is m+r-1 — exactly the number of distinct input samples the m-output window touches, "one multiply per input" — versus the m·r that direct computation spends, so for short filters there is a wide gap between the direct count and the known minimum. The mathematics behind such complexity results lives in the algebra of polynomials over a field: the standard toolkit (polynomial multiplication and division, evaluation/interpolation at chosen points, and the Chinese Remainder Theorem for polynomials) is the machinery in which exact bilinear convolution identities are stated and proved. This 1D theory is established for filtering a single sequence; how (or whether) it bears on a 2D, multi-channel, multi-filter convnet layer run on matrix-multiply hardware is not something the classical literature settles.

A second body of background is the **FFT route to fast convolution**, recently brought to convnets. The convolution theorem turns convolution into pointwise multiplication in the Fourier domain; transforming each image channel and each filter once and reusing the transforms across all filter/image pairings amortizes the transform cost when the number of feature maps is large. This was demonstrated for convnet training and inference (Mathieu, Henaff & LeCun 2013), refined with a GPU-tuned batched FFT (Vasilache et al. 2014), and shipped in a vendor library. Its known structure and limits: only m×n of the (m+r-1)×(n+s-1) cyclic-convolution outputs of a tile are valid, so tiles must overlap (overlap-and-save) and be large to amortize; the pointwise stage multiplies *complex* numbers (4 real multiplies, reducible to 3 with a fast complex-multiply trick, or roughly halved by exploiting Hermitian symmetry of a real signal's transform), so even at best it costs well above 1 real multiply per input; and a large tile (e.g. 64×64) inflates each transformed filter channel enormously (a 3×3 filter becomes 64×64 = 4096 units) and demands a large memory workspace and a large batch to generate enough tiles for an efficient pointwise stage.

A third relevant thread is **reducing the number of convolutions** rather than the cost of each: Strassen-style fast matrix multiplication has been applied to the K×C structure of a layer to cut total arithmetic (Cong & Xiao 2014), with each recursion halving all three matrix dimensions for an 8/7 reduction in multiplies. And a body of work reduces convnet cost by **quantization / low precision**: convnets train and infer correctly with surprisingly few bits (Courbariaux et al. 2014; Gupta et al. 2015), a fact that becomes important whenever a fast algorithm trades numerical accuracy for speed.

Two diagnostic facts about the existing fast methods frame the problem. First, FFT-based convnet layers underperform badly at moderate batch sizes — a vendor library that switches to FFT for intermediate N can drop under 2 TFLOPS — symptomatic of large tiles or one-tile-per-image schemes whose pointwise stage is too small to be efficient unless N is large. Second, the FFT workspace can balloon (multiple gigabytes), which is hostile to GPUs with limited on-chip memory.

## Baselines

- **Direct convolution.** Compute each output as the explicit Σ_c Σ_{u,v} D·G. R·S multiplies per output (1D: m·r per m-window). Simple, exact, no workspace, and the natural way to cast a layer as a matrix multiply (im2col + GEMM). Gap: 9 multiplies per output for 3×3 is pure arithmetic that the minimal-filtering bound shows is unnecessary; nothing about it is minimal for small filters.

- **FFT-based convolution (Mathieu, Henaff & LeCun 2013; refined by Vasilache et al. 2014; vendor cuDNN).** Transform image channels and filters to the Fourier domain once, multiply pointwise, inverse-transform; reuse transformed feature maps across all pairings; sum over channels in the frequency domain. Core math: tile the image, take a 2D FFT of each tile and filter, pointwise-multiply (yielding cyclic convolution — overlap-and-save to recover linear outputs), inverse-FFT. Fast for *large* filters and large feature-map counts. Gaps: the pointwise stage costs a complex multiply per input (≥1.5–4 real multiplies even after Hermitian-symmetry and fast-complex-multiply tricks), strictly worse than 1; large tiles are needed to amortize the transform, which (a) wastes computation on unwanted pixels for image sizes not near a multiple of the tile, (b) blows up filter storage and workspace memory, and (c) needs a large batch to have enough tiles for an efficient pointwise matrix multiply — exactly the small-batch, small-filter regime that matters is where it is weakest.

- **Strassen recursion on the layer (Cong & Xiao 2014).** Treat the layer's convolutions as a matrix product and apply Strassen to cut the number of convolutions, an 8/7 arithmetic reduction per recursion. Gap: each recursion halves all three matrix dimensions, shrinking the very dimensions a GEMM needs to stay efficient, and it reduces the *count* of convolutions, not the cost of each small convolution.

- **Quantization / low-precision layers (Courbariaux et al. 2014; Gupta et al. 2015).** Reduce arithmetic cost by computing the convolution in fewer bits. Orthogonal to algebraic restructuring: it changes the datatype, not the number of multiplications. Relevant mainly as evidence that some numerical accuracy can be sacrificed without hurting the network.

## Evaluation settings

The natural yardstick is a deep 3×3 convnet for ImageNet-scale recognition — the VGG family, whose configuration E uses 3×3 filters exclusively across layers of shapes ranging from 3×224×224 with 64 filters up to 512×14×14 with 512 filters (about 39 GFLOPs of direct convolution per image at N=1). The protocol of interest sweeps batch size across the hard regime — N from 1 to 64 — and measures per-layer and whole-network throughput as Effective TFLOPS (direct-convolution GFLOPs ÷ runtime), so an arithmetic-reducing algorithm can exceed the device's nominal peak. The comparison point is the vendor convolution library on a single high-end GPU. Numerical accuracy is assessed by maximum element error against a double-precision direct-convolution ground truth, in single (fp32) and half (fp16) precision, on the same VGG layer shapes. Memory is measured as workspace footprint.

## Code framework

The pre-existing primitives: dense matrix multiply (GEMM/SGEMM, including a batched form and a β-accumulate form), elementwise operations, and a symbolic-math toolkit (polynomial arithmetic, rational arithmetic, and matrix construction over an exact field) sufficient to build and *symbolically verify* small exact linear-algebra identities. A convnet layer is, abstractly, a per-channel small-window operation summed over channels.

```python
import numpy as np

# --- baseline for checking ---
def direct_conv_valid(D, W):
    # TODO: explicit valid R x S correlation, summed over channels.
    pass

# --- the slot: a full convolution layer with fewer multiplies than direct ---
def conv_layer(D, W):
    # D: N,C,H,W   W: K,C,R,S   ->  Y: N,K,H',W'
    # TODO: compute the exact layer with fewer multiplications per output than
    #       direct R*S, keeping the heavy stage a dense matrix multiply that is
    #       efficient even at batch size 1, with a small workspace.
    pass
```
