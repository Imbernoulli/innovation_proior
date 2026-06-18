I start with the constraint, because it decides what kind of architecture can matter. I need a vision model that runs under a mobile or embedded budget. A high-accuracy stack of standard convolutions is not automatically useful here: if it costs billions of multiply-adds per image, it misses the deployment target even if the parameter file can be stored. So I need to reduce arithmetic and memory movement, and I need the reduction to come from structured dense operations that real libraries can execute efficiently.

I look first at one convolutional layer rather than at the whole network. The input has spatial size `D_F x D_F` and `M` channels. The output has the same spatial size and `N` channels. A standard square convolution uses a kernel of shape `D_K x D_K x M x N`, so each output value is

```text
G[k,l,n] = sum_{i,j,m} K[i,j,m,n] F[k+i-1,l+j-1,m].
```

For each of the `D_F^2` output locations and `N` output channels, I sum over `D_K^2` spatial taps and `M` input channels. The cost is therefore

```text
D_K^2 M N D_F^2.
```

This expression exposes the problem. The spatial kernel, input depth, output depth, and feature-map area all multiply. A standard convolution is doing two jobs at once: it filters local spatial neighborhoods and it mixes input channels into output channels. Because those jobs are fused in one weight tensor, the spatial factor `D_K^2` and the output-channel factor `N` multiply each other.

The natural move is to split the two jobs. First I keep only spatial filtering. I apply one `D_K x D_K` filter to each input channel independently:

```text
Ghat[k,l,m] = sum_{i,j} Khat[i,j,m] F[k+i-1,l+j-1,m].
```

This depthwise step has cost

```text
D_K^2 M D_F^2.
```

It drops the factor `N`, but it also cannot create new mixtures of channels. Each output channel still corresponds to one input channel. So I need a second step that mixes channels with no spatial footprint. A `1 x 1` convolution does exactly that: at each spatial location it applies an `M` by `N` matrix. Its cost is

```text
M N D_F^2.
```

Putting the two steps together gives

```text
D_K^2 M D_F^2 + M N D_F^2.
```

Now I divide by the standard convolution cost:

```text
(D_K^2 M D_F^2 + M N D_F^2) / (D_K^2 M N D_F^2)
  = 1/N + 1/D_K^2.
```

There is no sign ambiguity and no hidden constant in that cancellation. The first term is the cost of retaining per-channel spatial filtering; the second is the cost of retaining channel mixing. With `3 x 3` kernels, `1/D_K^2 = 1/9`. Once `N` is at least 64, `1/N` is small beside `1/9`, so the layer uses roughly one eighth to one ninth of the computation of a full `3 x 3` convolution. For a representative internal layer with `D_K=3`, `M=N=512`, and `D_F=14`, the full convolution costs `3*3*512*512*14*14 = 462.4M` multiply-adds and has `2.36M` weights. The split version costs `3*3*512*14*14 + 512*512*14*14 = 52.3M` multiply-adds and has `0.267M` weights. The arithmetic matches the ratio `1/512 + 1/9`.

This split also tells me where the remaining work lives. The depthwise-to-pointwise cost ratio is `D_K^2 : N`; with `3 x 3` filters and hundreds of output channels, almost all work is in the `1 x 1` convolution. That is a useful shape for hardware, because a `1 x 1` convolution is just a dense matrix multiply over the channel dimension at each spatial position. It does not need the same `im2col` lowering that a general spatial convolution often needs. So the saved multiply-adds are not achieved by making the computation irregular or sparse; the remaining heavy operation is a dense GEMM-like channel mixer.

I do not want to factorize the spatial part further unless it buys enough to justify the extra constraint. Flattening a filter into one-dimensional pieces is a stronger assumption. Once the depthwise term is already a small fraction of the total cost, making the spatial filter rank-one can only attack a small part of the remaining compute while risking a larger loss in representation. The cleaner compromise is to keep full two-dimensional spatial filters per channel and a full dense `1 x 1` channel mixer.

The repeating block is therefore fixed: depthwise `3 x 3` convolution, normalization, activation, then pointwise `1 x 1` convolution, normalization, activation. The first layer is a special case. With only three RGB input channels, a depthwise first layer would have only three spatial filters before any channel expansion. I keep the stem as a full `3 x 3` convolution from 3 to 32 channels with stride 2, then use the split block everywhere else.

For the whole network, I use the usual pyramid logic: as spatial resolution shrinks, channel count grows. The stem maps `224 x 224` to `112 x 112`. Strided depthwise steps then create the sequence `112 -> 56 -> 28 -> 14 -> 7`. The channel schedule is `32 -> 64 -> 128 -> 128 -> 256 -> 256 -> 512`, then five more `512 -> 512` blocks, then `512 -> 1024`, then `1024 -> 1024`. The final `1024` block stays at stride 1; a stride-2 label there would contradict the retained `7 x 7` feature size and the implementation. A global average pool reduces the spatial map to `1 x 1`, and a linear classifier produces logits. Counting the first convolution, every depthwise and pointwise convolution separately, and the final classifier gives 28 layers.

Now I need budget knobs, not just one fixed model. Channel width is the first knob. In the split block, the cost is

```text
D_K^2 M D_F^2 + M N D_F^2.
```

If I scale every channel count by `alpha`, then the depthwise term becomes linear in `alpha` and the pointwise term becomes quadratic:

```text
D_K^2 alpha M D_F^2 + alpha M alpha N D_F^2.
```

The exact expression matters: the total is not purely `alpha^2` because the depthwise term is only linear. But the pointwise term dominates, so compute and parameters fall approximately as `alpha^2`. Typical settings such as `1.0`, `0.75`, `0.5`, and `0.25` define separate thinner networks trained from scratch, not pruned copies of a larger model. The single-layer check is consistent: at `alpha=0.75`, the representative layer goes from `52.3M` to `29.6M` multiply-adds and from `0.267M` to `0.151M` weights.

Spatial resolution is the second knob. If the input resolution is scaled by `rho`, the internal feature maps scale with it, so every `D_F^2` becomes `(rho D_F)^2`. With both knobs, the block cost is

```text
D_K^2 alpha M (rho D_F)^2 + alpha M alpha N (rho D_F)^2.
```

The `rho` multiplier changes compute by `rho^2` and leaves parameter count unchanged, because weights do not depend on feature-map size. In the same representative layer, shrinking `14` to `10` gives `rho=10/14=0.714`, so the `alpha=0.75` cost falls from `29.6M` to `15.1M` multiply-adds while the parameter count remains `0.151M`.

I also choose thin over shallow as the main way to reduce the model. Removing whole nonlinear stages gives up representational depth. Uniformly thinning every layer keeps the full sequence of transformations and mostly attacks the dominant pointwise terms. At similar compute, I expect the deep-thin version to retain accuracy better than a shortened version that deletes the run of middle `512` blocks.

The training choices follow from the same size argument. These models have much less capacity than large Inception-like systems, and the depthwise filters contain very few parameters compared with pointwise filters. Heavy regularization aimed at large overparameterized models can become counterproductive. So I use less aggressive augmentation, no auxiliary classifier heads, no label smoothing, and little or no weight decay on depthwise filters. In a faithful implementation I also preserve the code-level details that affect behavior: channel counts are `max(int(d * alpha), 8)`, depthwise convolutions are expressed as per-channel grouped convolutions or as `separable_conv2d` with no pointwise output, and the reference activation is the clipped ReLU6 variant even though the architectural description often says ReLU.

The final construction is a simple branch-free family: a full-convolution stem, thirteen depthwise-separable blocks with stride placed in the depthwise step, average pooling, and logits. The mathematical reason it works is the cancellation from `D_K^2 M N D_F^2` to `D_K^2 M D_F^2 + M N D_F^2`; the engineering reason it is deployable is that the remaining dominant computation is dense `1 x 1` channel mixing; and the practical reason it is useful is that `alpha` and `rho` give independent, predictable controls over width and resolution.
