# Context: pure convolution for general time series analysis (circa 2023-2024)

## Research question

A time series presents, at each step `t`, an `M`-channel vector `x_t in R^M`, observed over a window
of length `L`. Five tasks share this input — long- and short-term forecasting, imputation, anomaly
detection, and classification — and the field has fragmented into task-specific architectures. The live
problem is that **convolution**, the workhorse that powers vision, has been written off for time series:
the dominant view is that 1-D temporal convolution has too small an effective receptive field (ERF) to
capture the long-range dependencies a Transformer's attention captures, so attention-based and
MLP-based models have taken over the benchmarks. The question is whether that verdict is wrong — whether
a *modern* convolution design, importing the architectural lessons that made ConvNeXt competitive with
vision Transformers (large depthwise kernels, pointwise channel mixing, an inverted-bottleneck FFN),
can match or beat the attention/MLP models across all five tasks while keeping convolution's efficiency.
A solution must (1) achieve a genuinely large ERF along time so it can model long dependencies; (2)
handle the multivariate axis explicitly — model cross-variable dependence, which channel-independent
forecasters drop; (3) be one backbone that serves all five tasks by swapping only the head; (4) stay
efficient (convolutional cost, not quadratic attention); and (5) be trainable from scratch on the
modest sizes of these benchmarks without overfitting.

## Background

By this point three lines dominate time-series analysis. **Transformer forecasters** (Informer,
Autoformer, FEDformer, and later PatchTST and the inverted iTransformer) relate sequence elements by
attention; PatchTST in particular showed that patching the series into sub-series tokens and processing
each channel independently with a shared backbone beats the channel-mixing Transformers, at the cost of
dropping cross-variable modelling. **MLP models** (DLinear, the decomposition-linear baseline; TSMixer;
TimeMixer) showed a plain linear or MLP map over the temporal axis is shockingly competitive, which cast
doubt on whether attention's inductive bias is even needed. **TimesNet** (Wu et al., ICLR 2023) is the
strongest task-general baseline: it reshapes the 1-D series into 2-D by discovered periods so that
intraperiod and interperiod variation both become 2-D locality, then runs a parameter-efficient
Inception block of multi-scale 2-D convolutions — notably, it succeeds by going *to* convolution, but in
a 2-D reshaped space rather than along the native time axis.

In vision, the parallel story had already resolved: after Vision Transformers briefly displaced CNNs,
**ConvNeXt** (Liu et al., 2022) showed a pure ConvNet, modernized with large depthwise kernels (7×7), an
inverted-bottleneck pointwise FFN, fewer activations/norms, and BatchNorm→LayerNorm choices, matches or
beats Swin Transformers. The key modern-conv idea is **separation of concerns**: a depthwise conv mixes
only spatially (large kernel for a big ERF, cheap because per-channel), and 1×1 (pointwise) convs do all
the cross-channel mixing. The open question this work takes up is whether that recipe transfers to time
series — where the "spatial" axis is time and the "channel" axis splits into two distinct things (a
learned feature dimension *and* the physical variable dimension) that a naive port would conflate.

## Reference implementation (read-only)

The canonical implementation is the official ModernTCN repository
(https://github.com/luodhhh/ModernTCN, ICLR 2024 Spotlight, "ModernTCN: A Modern Pure Convolution
Structure for General Time Series Analysis", https://openreview.net/forum?id=vpJMJerXHU). The
classification variant lives in `ModernTCN-classification/models/ModernTCN.py`: a `ReparamLargeKernelConv`
(a large depthwise conv branch in parallel with a small one, each with BatchNorm, summed, fusing to one
kernel at inference via `merge_kernel`), a `Block` (DWConv → BatchNorm → ConvFFN1 grouped by `nvars` for
cross-feature mixing → ConvFFN2 grouped by `dmodel` for cross-variable mixing → residual), a multi-stage
backbone with a patch-embedding stem and strided-conv downsampling between stages, and a classification
head (GELU → dropout → flatten → linear to `num_class`). The Time-Series-Library classification protocol
(RAdam, CrossEntropyLoss, accuracy on UEA datasets) is the evaluation harness.
