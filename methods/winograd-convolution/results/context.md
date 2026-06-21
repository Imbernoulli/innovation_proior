# Context: fast small-filter convolution for convolutional networks

## Research question

Deep convolutional networks built almost entirely from small 3×3 convolution layers have become the state of the art for image recognition. Their cost is dominated by the convolution layers, and that cost is paid across both training and inference. Two regimes are of particular interest: very small batch sizes (a single image for low-latency detection; small per-node minibatches when training is sharded across a cluster) and small filters (3×3 almost exclusively). The problem: compute a convolution layer — correlate K filters of C channels and size R×S against N images of C channels and size H×W — using as few multiplications per output as possible, while keeping the heavy stage a dense matrix multiply that is efficient across a range of batch sizes, and keeping the working memory small.

## Background

A convnet layer computes, for image i, filter k, position (x,y):
Y[i,k,x,y] = Σ_c Σ_v Σ_u D[i,c,x+u,y+v] · G[k,c,u,v], i.e. a sum over C channels of 2D correlations of an H×W image channel with an R×S filter. Direct evaluation spends r multiplications per output in 1D and R·S in 2D (3×3 ⇒ 9 multiplications per output pixel per channel). The arithmetic is regular but multiply-heavy.

A relevant classical result is the **arithmetic complexity of FIR filters**. Computing m outputs of an r-tap FIR filter is written F(m,r). It has been known since at least 1980 that the minimum number of *general* multiplications required is m+r-1 — exactly the number of distinct input samples the m-output window touches, "one multiply per input" — versus the m·r that direct computation spends. The mathematics behind such complexity results lives in the algebra of polynomials over a field: the standard toolkit (polynomial multiplication and division, evaluation/interpolation at chosen points, and the Chinese Remainder Theorem for polynomials) is the machinery in which exact bilinear convolution identities are stated and proved.

A second body of background is the **FFT route to fast convolution**, recently brought to convnets. The convolution theorem turns convolution into pointwise multiplication in the Fourier domain; transforming each image channel and each filter once and reusing the transforms across all filter/image pairings amortizes the transform cost when the number of feature maps is large. This was demonstrated for convnet training and inference (Mathieu, Henaff & LeCun 2013), refined with a GPU-tuned batched FFT (Vasilache et al. 2014), and shipped in a vendor library. Its structure: tile the image, take a 2D FFT of each tile and filter, pointwise-multiply (yielding cyclic convolution — overlap-and-save to recover linear outputs), inverse-FFT. The pointwise stage multiplies complex numbers (4 real multiplies, reducible to 3 with a fast complex-multiply trick, or roughly halved by exploiting Hermitian symmetry of a real signal's transform). A large tile (e.g. 64×64) transforms a 3×3 filter to 64×64 = 4096 units.

A third relevant thread is **reducing the number of convolutions** rather than the cost of each: Strassen-style fast matrix multiplication has been applied to the K×C structure of a layer to cut total arithmetic (Cong & Xiao 2014), with each recursion halving all three matrix dimensions for an 8/7 reduction in multiplies. And a body of work reduces convnet cost by **quantization / low precision**: convnets train and infer correctly with surprisingly few bits (Courbariaux et al. 2014; Gupta et al. 2015), a fact that becomes important whenever a fast algorithm trades numerical accuracy for speed.

## Baselines

- **Direct convolution.** Compute each output as the explicit Σ_c Σ_{u,v} D·G. R·S multiplies per output (1D: m·r per m-window). Simple, exact, no workspace, and the natural way to cast a layer as a matrix multiply (im2col + GEMM).

- **FFT-based convolution (Mathieu, Henaff & LeCun 2013; refined by Vasilache et al. 2014; vendor cuDNN).** Transform image channels and filters to the Fourier domain once, multiply pointwise, inverse-transform; reuse transformed feature maps across all pairings; sum over channels in the frequency domain. Core math: tile the image, take a 2D FFT of each tile and filter, pointwise-multiply (yielding cyclic convolution — overlap-and-save to recover linear outputs), inverse-FFT. Fast for *large* filters and large feature-map counts.

- **Strassen recursion on the layer (Cong & Xiao 2014).** Treat the layer's convolutions as a matrix product and apply Strassen to cut the number of convolutions, an 8/7 arithmetic reduction per recursion.

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
