I start with the constraint, because it decides what kind of architecture can matter. I need a vision model that runs under a mobile or embedded budget. A high-accuracy stack of standard convolutions is not automatically useful here: if it costs billions of multiply-adds per image, it misses the deployment target even if the parameter file can be stored. So I need to reduce arithmetic and memory movement, and I need the reduction to come from structured dense operations that real libraries can execute efficiently.

I look first at one convolutional layer rather than at the whole network. The input has spatial size `D_F x D_F` and `M` channels. The output has the same spatial size and `N` channels. A standard square convolution uses a kernel of shape `D_K x D_K x M x N`, so each output value is

```text
G[k,l,n] = sum_{i,j,m} K[i,j,m,n] F[k+i-1,l+j-1,m].
```

For each of the `D_F^2` output locations and `N` output channels, I sum over `D_K^2` spatial taps and `M` input channels. The cost is therefore

```text
D_K^2 M N D_F^2.
```

This expression exposes the problem. The spatial kernel, input depth, output depth, and feature-map area all multiply. Looking at the index sum, a standard convolution is doing two things inside one weight tensor: the `i,j` indices slide a spatial neighborhood, and the `m` index contracts input channels into each output channel `n`. Spatial filtering and channel mixing are fused, which is why the spatial factor `D_K^2` and the output-channel factor `N` sit in the same product and multiply each other.

If those two jobs are separable, I should be able to do them in two cheaper passes. Let me try keeping only the spatial part first: one `D_K x D_K` filter applied to each input channel on its own,

```text
Ghat[k,l,m] = sum_{i,j} Khat[i,j,m] F[k+i-1,l+j-1,m].
```

The `n` index is gone from this sum; the output index is `m`, the same as the input. Cost:

```text
D_K^2 M D_F^2.
```

Before I trust that this composes into something equivalent in shape, I want to be sure I understand exactly what this depthwise step can and cannot do, because the whole plan rests on the two passes being complementary rather than redundant. I work a tiny case by hand. Take a `4 x 4` input with `M = 2` channels, one `3 x 3` filter per channel, then later an `M -> N` mix with `N = 3`. The depthwise pass over the valid region gives a `2 x 2 x 2` map — still two channels, not three. Two facts I can read straight off the index expression and confirm on the small array: output channel `m` is built only from input channel `m` (zeroing input channel 0 leaves depthwise channel 1 bit-for-bit identical, while channel 0 changes), and no entry of the depthwise output is a combination of two different input channels. So this pass filters space but cannot create a single new channel mixture. That is exactly the half of the original convolution I have dropped, and it tells me precisely what the second pass has to supply: cross-channel combination with no spatial reach.

The operation that does only that is a `1 x 1` convolution. At each spatial location it applies an `M` by `N` matrix and touches no neighbors:

```text
M N D_F^2.
```

On the same toy case, multiplying the `2 x 2 x 2` depthwise map by a `2 x 3` matrix produces a `2 x 2 x 3` map, and checking one output position by hand, `G[0,0,0]` equals `sum_m Ghat[0,0,m] Kpw[m,0]` — every output channel is a linear combination of all `M` depthwise channels. So depthwise supplies the spatial filtering, pointwise supplies the mixing, and together their output has the right `N`-channel shape. The two passes are complementary, not two attempts at the same thing.

Putting the two steps together gives

```text
D_K^2 M D_F^2 + M N D_F^2.
```

Now the question is whether this is actually cheaper, and by how much. I divide by the standard convolution cost:

```text
(D_K^2 M D_F^2 + M N D_F^2) / (D_K^2 M N D_F^2).
```

The common factor `M D_F^2` cancels from every term, leaving

```text
1/N + 1/D_K^2.
```

Two terms with disjoint meaning: `1/N` is what I pay to keep per-channel spatial filtering, `1/D_K^2` is what I pay to keep channel mixing. With `3 x 3` kernels, `1/D_K^2 = 1/9 = 0.1111`. The first term shrinks as the layer gets wider: at `N = 64` it is `0.0156`, small beside `0.1111`, so the layer ends up at roughly one eighth to one ninth of a full `3 x 3` convolution. I want to see that the cancellation I did symbolically agrees with raw arithmetic on a real layer before I lean on it. Take a representative internal layer, `D_K = 3`, `M = N = 512`, `D_F = 14`. The full convolution is `3*3*512*512*14*14 = 462,422,016` multiply-adds with `3*3*512*512 = 2,359,296` weights. The split is `3*3*512*14*14 + 512*512*14*14 = 903,168 + 51,380,224 = 52,283,392` multiply-adds with `3*3*512 + 512*512 = 266,752` weights. Dividing, `52,283,392 / 462,422,016 = 0.11306`, and `1/512 + 1/9 = 0.001953 + 0.111111 = 0.11306`. The two routes land on the same number, so the cancellation is right and the saving is real: about `8.8x` fewer multiply-adds and about `8.8x` fewer weights here.

The split also tells me where the remaining work lives. In that layer the depthwise pass is `903,168` multiply-adds and the pointwise pass is `51,380,224` — the pointwise is `56x` larger, so essentially all the surviving compute is the `1 x 1` convolution. In general the ratio of the two terms is `D_K^2 : N`, which for `3 x 3` filters and hundreds of channels makes the pointwise term dominate. That is a useful shape for hardware, because a `1 x 1` convolution is just a dense matrix multiply over the channel dimension at each spatial position. It does not need the `im2col` lowering a general spatial convolution often needs. So the saved multiply-adds are not bought by making the computation irregular or sparse; the remaining heavy operation is a dense GEMM-like channel mixer.

This makes me reconsider whether to factorize the spatial part further. Flattening the depthwise filter into one-dimensional pieces is a stronger assumption — it forces the per-channel spatial filter to be rank one. But I just measured that the entire depthwise term is `903,168` of `52,283,392`, under `2%` of the layer. Even if a flattened filter were free, it could remove at most that `2%` while constraining the spatial filters more tightly and risking representational loss. The arithmetic does not justify the extra constraint. The cleaner compromise is full two-dimensional spatial filters per channel and a full dense `1 x 1` channel mixer.

The repeating block is therefore depthwise `3 x 3` convolution, normalization, activation, then pointwise `1 x 1` convolution, normalization, activation. The first layer is a special case. With only three RGB input channels, a depthwise first layer would have only three spatial filters before any channel expansion, and the toy case showed depthwise alone creates no new channels — so it would carry almost no representational work. I keep the stem as a full `3 x 3` convolution from 3 to 32 channels with stride 2, then use the split block everywhere else.

For the whole network, I use the usual pyramid logic: as spatial resolution shrinks, channel count grows. The stem maps `224 x 224` to `112 x 112`. Strided depthwise steps then create the sequence `112 -> 56 -> 28 -> 14 -> 7`. The channel schedule is `32 -> 64 -> 128 -> 128 -> 256 -> 256 -> 512`, then five more `512 -> 512` blocks, then `512 -> 1024`, then `1024 -> 1024`. I want the final feature size to come out at `7 x 7`, so I need to check the strides multiply out correctly: starting from `112` after the stem and halving once per stride-2 block, the schedule above has four stride-2 blocks, `112/2/2/2/2 = 7`. That forces the last `1024` block to be stride 1 — a stride-2 there would push the map to `3 x 3` and contradict the retained `7 x 7` size. A global average pool reduces the spatial map to `1 x 1`, and a linear classifier produces logits. To name the depth, I count the first convolution, every depthwise and pointwise convolution separately, and the final classifier: `1 + 2*13 + 1 = 28` layers.

Now I need budget knobs, not just one fixed model. Channel width is the first knob. In the split block the cost is

```text
D_K^2 M D_F^2 + M N D_F^2.
```

If I scale every channel count by `alpha`, then `M -> alpha M` and `N -> alpha N`:

```text
D_K^2 (alpha M) D_F^2 + (alpha M)(alpha N) D_F^2.
```

The depthwise term carries one factor of `alpha`, the pointwise term carries `alpha^2`. So total cost is not a clean `alpha^2` — but since I already measured the pointwise term to be the overwhelming majority, the total should fall close to `alpha^2`. Let me check against the raw count rather than assume it. At `alpha = 0.75` the representative layer has `M = N = 384`: depthwise `3*3*384*14*14 = 677,376`, pointwise `384*384*14*14 = 28,901,376`, total `29,578,752` multiply-adds and `3*3*384 + 384*384 = 150,912` weights. Compared with the `alpha = 1.0` figures, compute went `52.28M -> 29.58M`, a ratio `0.566`, while pure `alpha^2 = 0.5625`. Close, and slightly above it exactly because the depthwise term only scaled linearly — which is the small correction I predicted. Typical settings `1.0, 0.75, 0.5, 0.25` define separate thinner networks trained from scratch, not pruned copies of a larger model.

Spatial resolution is the second knob. If the input resolution is scaled by `rho`, the internal feature maps scale with it, so every `D_F^2` becomes `(rho D_F)^2`. With both knobs the block cost is

```text
D_K^2 alpha M (rho D_F)^2 + alpha M alpha N (rho D_F)^2.
```
