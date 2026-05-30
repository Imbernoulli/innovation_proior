OK, let me think this through from scratch. I want a convolutional network that runs in real time on a phone — tens of milliseconds, on a CPU with no real GPU, inside a power budget. The accuracy leaders right now get there by going deeper and wider: VGG is a tower of 3×3 convolutions costing something like fifteen billion multiply-adds a frame, Inception is cleverer but still well over a billion. None of that fits. So the goal isn't "a bit smaller"; it's an order of magnitude or two less compute, while losing only a little accuracy. And I'd like a knob — give me a target budget, I hand back a model that fits it.

Before I reach for any architecture, let me just stare at where the cost actually comes from, because if I don't understand the cost structure I'll shrink the wrong thing. Take one standard convolutional layer. The input feature map is D_F × D_F × M — spatial size D_F by D_F, M channels. The output, with stride one and padding so the spatial size is preserved, is D_F × D_F × N. The kernel is a 4-D tensor K of shape D_K × D_K × M × N. Each output pixel and output channel is

    G[k,l,n] = Σ_{i,j,m} K[i,j,m,n] · F[k+i-1, l+j-1, m].

Count the multiply-adds. For every one of the D_F·D_F output locations, for every one of the N output channels, I sum over D_K·D_K spatial taps and M input channels. So the cost is

    D_K · D_K · M · N · D_F · D_F.

Look at the shape of that. It's a *product* of four conceptually separate things: the spatial kernel footprint D_K², the input depth M, the output depth N, and the spatial extent D_F². They all multiply. That's the whole problem in one line — there's no single term to attack, every factor scales the others.

Now, what is the layer actually *doing* for that price? Two jobs, fused into one operation. Job one: spatial filtering — within each channel, the D_K × D_K support detects local spatial structure (edges, textures). Job two: channel combination — it takes the M input channels and mixes them, via the m-sum, into N brand-new feature channels. A standard conv does filtering *and* combining in a single step, and that's exactly why the kernel-size factor D_K² and the output-channel factor N sit in the same product and multiply: the same weights K[i,j,m,n] are paying for both jobs at once.

So here's the question I keep circling back to: can I factor those two jobs apart? If filtering and combining were two separate operations, maybe their costs would *add* instead of *multiply*, and an additive cost is dramatically smaller than a multiplicative one. Let me try to actually build that.

Job one alone — spatial filtering, no channel mixing. That means: apply a single D_K × D_K filter to each input channel, independently, producing one filtered channel per input channel. No sum over m anymore; channel m only ever talks to channel m. Call its kernel K̂ of shape D_K × D_K × M (one filter per channel, not M×N filters):

    Ĝ[k,l,m] = Σ_{i,j} K̂[i,j,m] · F[k+i-1, l+j-1, m].

The m-th filter is applied to the m-th channel to make the m-th output channel. This is a depthwise convolution. Its cost: for each of D_F·D_F locations, each of M channels, sum over D_K·D_K taps — and crucially there's no inner sum over a separate set of channels, no N:

    D_K · D_K · M · D_F · D_F.

Compare that to the standard conv's D_K·D_K·M·N·D_F·D_F — I've dropped a whole factor of N. That's already a big saving. But it can't be the whole layer, because depthwise convolution only filters; it never combines channels. The output has exactly M channels and each is just a filtered version of the corresponding input channel. There's no mechanism to build a new representation that mixes information across channels, which is most of what a conv net's expressive power comes from. So I need job two as a separate step.

Job two alone — channel combination, no spatial extent. I want to take the M filtered channels and form N new channels, each a linear combination across the M, but I do *not* want to do any more spatial filtering (the depthwise step already handled space). A linear combination across channels at each pixel, with no spatial footprint, is precisely a convolution with a 1×1 kernel: a kernel of shape 1 × 1 × M × N. At each spatial location it's a matrix-vector product, M → N. People call this a pointwise convolution. Its cost: D_F·D_F locations, each doing an M×N mixing, kernel footprint 1:

    M · N · D_F · D_F.

Now chain them: depthwise (filter) then pointwise (combine), each followed by normalization and a nonlinearity so it trains. The total cost is the sum:

    D_K · D_K · M · D_F · D_F  +  M · N · D_F · D_F.

The two jobs that used to multiply now *add*. Let me see exactly how much I save by dividing the separable cost by the standard cost:

    (D_K²·M·D_F² + M·N·D_F²) / (D_K²·M·N·D_F²).

Split the fraction across the sum. The common factor M·D_F² cancels in both terms, so:

    = (D_K²·M·D_F²)/(D_K²·M·N·D_F²) + (M·N·D_F²)/(D_K²·M·N·D_F²)
    = 1/N + 1/D_K².

That's clean. The reduction factor is 1/N + 1/D_K². With 3×3 kernels, D_K² = 9, so the second term is 1/9. And N — the number of output channels — is 64, 128, 256, 512, 1024 in any real net, so 1/N is at most ~1/64 and usually much smaller; it's negligible next to 1/9. So the speedup is essentially 1/D_K² ≈ 1/9: between 8 and 9 times less computation for a 3×3 layer. Eight to nine times, for one factorization, and I haven't even shrunk the network yet. The accuracy cost of replacing full convs with this — I'd expect small, because I haven't removed the ability to filter spatially or to mix channels, I've only forbidden doing both *in the same weights*; the net can still learn both, just in two steps.

Let me sanity-check the asymmetry, because it tells me where to spend effort. Depthwise costs D_K²·M·D_F², pointwise costs M·N·D_F². Their ratio is D_K² : N = 9 : N. For N in the hundreds to a thousand, the pointwise (1×1) term is tens of times larger than the depthwise term — almost all the compute now lives in the 1×1 convolutions. That's worth holding onto: it means further factorizing the *spatial* part (à la flattened, rank-1 filters, or n×1 then 1×n tricks) would buy me almost nothing, since the depthwise step is already a sliver of the cost. The thing to be careful about is the 1×1 convs.

And here's a reason that's actually good news rather than a worry. A 1×1 convolution, at each spatial location, is just a matrix multiply of the M-vector by an N×M weight matrix — so the whole layer is one big dense matrix multiplication (a GEMM) over the flattened activations. It maps to GEMM with *no* memory reshuffle. A general k×k convolution, by contrast, has to be lowered to a GEMM by first doing an im2col — copying overlapping patches into a big matrix — which costs memory traffic and is the thing that makes generic convs slow in practice (it's exactly what Caffe does). GEMM itself is the most heavily optimized kernel in all of numerical computing. So by pushing ~95% of the compute into 1×1 convs, I'm pushing it into the one operation that hardware already runs near peak. This matters because minimizing multiply-adds on paper is not the same as being fast: unstructured sparse operations, for instance, don't actually beat dense ones until extreme sparsity. I want my savings to be *dense and structured* so they show up as real latency, and 1×1 convs deliver that. So I won't chase the spatial factorization; I'll keep full 2D depthwise filters and let the 1×1s carry the load.

Why this particular split and not one of the more aggressive factorizations floating around? I could go fully separable — decompose each filter into rank-1 1D pieces, separable in every dimension, the way flattened networks do. But that's a much stronger low-rank assumption on the filters, and it tends to cost real accuracy; I'd be over-constraining what each layer can represent. Depthwise-separable is the gentler factorization: it makes exactly one structural assumption — that spatial filtering and channel mixing can be done in sequence rather than jointly — and keeps each piece full-rank within its own job (full 2D spatial kernels, full dense channel mixing). That feels like the right amount of factorization: enough to collapse the multiplicative cost into an additive one, not so much that I cripple the layer.

So my building block is settled: a 3×3 depthwise conv, then BN, then ReLU; then a 1×1 pointwise conv, then BN, then ReLU. The normalization is the standard batch-norm — normalize each channel's pre-activations by minibatch statistics with a learned scale and shift — which is what makes deep conv stacks train stably; I put it after each of the two convs, with a ReLU after each, so the depthwise step isn't left as a bare linear map before mixing.

Now stack these into a whole network. The very first layer is a problem case: the input has only 3 channels (RGB), and depthwise conv would apply one filter per channel — three filters total — which throws away almost all the early representational power and there's nothing to "separate" yet. So I'll make the first layer an ordinary full 3×3 convolution, 3→32 channels, stride 2 to immediately halve the spatial size. After that, everything is depthwise-separable blocks.

For the body, I want channels to grow as resolution shrinks — the usual pyramid, so the per-layer compute stays roughly balanced as I go deeper. Downsampling: rather than insert pooling layers, I'll fold the stride into the depthwise convolution (and the stem) — a stride-2 depthwise conv both filters and halves resolution for free. Start at 224×224; stem takes it to 112; then strided depthwise steps take 112→56→28→14→7. The channel schedule doubles at each downsampling: stem to 32, then 32→64, then →128 (and one more 128 block), →256 (and one more), →512, then a run of five more blocks that stay at 512, then →1024 with a final 1024 block. Why a run of five layers all at 512×512, 14×14? That's where I want the most representational work — a deep stack at a moderate resolution and width is where features get rich — and at 14×14 each such layer is cheap enough to afford several. After the last block I do a global average pool over the 7×7 map down to 1×1, then a single fully-connected layer 1024→1000 into a softmax. No nonlinearity on that final FC. Counting the depthwise and pointwise pieces as separate layers, that's 28 layers. Because every layer is described by the same simple block, the topology is trivial to explore — there are no branchy modules to hand-tune.

Let me put a number on a single representative layer to make sure the savings are real, using an internal block: D_K=3, M=512, N=512, D_F=14. Full conv would cost 3·3·512·512·14·14 ≈ 462 million multiply-adds, with 3·3·512·512 ≈ 2.36 million parameters. The depthwise-separable version costs 3·3·512·14·14 + 512·512·14·14 ≈ 0.9M + 51.4M ≈ 52.3 million multiply-adds (and the 0.9M depthwise term really is a rounding error next to the 51.4M pointwise term — confirming where the compute lives), with 3·3·512 + 512·512 ≈ 0.27 million parameters. So one block: 462M → 52.3M, about 9×, and parameters 2.36M → 0.27M, just as the 1/N + 1/D_K² analysis predicted. Across the whole net this lands the full model around half a billion multiply-adds and a few million parameters — versus VGG's ~15 billion and ~138 million. That's the order of magnitude I was after.

But I promised a *knob*. The base model is small, yet someone will always need it smaller — half the latency, a quarter of the size. I don't want to redesign the architecture each time; I want to slide one number. Where in the cost does a single dial do the most good? The cost of a separable block is

    D_K²·M·D_F² + M·N·D_F².

The channel counts M and N appear in the dominant (pointwise) term as a *product* M·N. So if I uniformly scale every layer's channel counts by a factor α — replace M with αM and N with αN everywhere — that term becomes α²·M·N·D_F², and the depthwise term becomes α·D_K²·M·D_F². The pointwise term dominates, so the whole cost (and the parameter count, which is also dominated by the M·N pointwise weights) drops by roughly α². One dial, quadratic savings. Call α the width multiplier: α in (0,1], with handy settings 1, 0.75, 0.5, 0.25; α=1 is the base network, smaller α gives thinner networks. Each choice defines a genuinely new, smaller architecture that I train from scratch — this is not pruning a trained model, it's a family of architectures indexed by α.

There's a second axis I haven't touched: the spatial size D_F². It enters every term and I've been treating the input as fixed at 224. But I can just feed the network smaller images. If I scale the input resolution by ρ, then because each layer's spatial map is proportional to the input, every D_F becomes ρ·D_F and the block cost becomes

    D_K²·αM·(ρD_F)² + αM·αN·(ρD_F)² = ρ²·(D_K²·αM·D_F² + αM·αN·D_F²).

So a resolution multiplier ρ scales compute by ρ². I set ρ implicitly by choosing the input resolution — 224, 192, 160, 128 — with ρ=1 at 224. Note ρ multiplies only the spatial factor, so it cuts compute but leaves the parameter count untouched (parameters are the kernel weights, independent of feature-map size). That's a useful difference from α, which cuts both.

Stack the two: a block under both multipliers costs D_K²·αM·(ρD_F)² + αM·αN·(ρD_F)². Let me trace one layer through, D_K=3, M=N=512, D_F=14. Full conv: ~462M mult-adds, 2.36M params. Make it depthwise-separable: ~52.3M, 0.27M. Apply α=0.75: channels 512→384, the pointwise term scales by 0.75²≈0.56, giving ~29.6M mult-adds and ~0.15M params. Now apply ρ=0.714 (i.e. shrink the 14×14 map to 10×10): compute scales by 0.714²≈0.51, ~29.6M → ~15.1M, while params stay at 0.15M because ρ doesn't touch the weights. So from 462M down to 15M on one layer, with two intuitive dials.

A couple of choices I should pin down by reasoning rather than leaving to chance. First: given a fixed compute budget, is it better to make the network *thinner* (small α, keep all the layers) or *shallower* (full width, drop some layers)? My instinct is that depth is doing real work — those five stacked 512-channel layers are where the representation matures — so I'd rather pay for depth and buy the budget back by thinning. If I had to make it shallower I'd remove that run of five 14×14×512 separable layers, but I expect a thinned-but-deep net to beat a shortened one at equal compute, because you lose less by making each layer a bit narrower than by deleting whole stages of nonlinear processing. So width multiplier, not layer removal, is the primary knob.

Second: regularization. These models are tiny — a depthwise filter has only D_K²·M parameters, the pointwise carries M·N — so the model has far less capacity to overfit than a big Inception or VGG. That flips the usual training recipe. Heavy data augmentation, label smoothing, auxiliary classifier heads — those exist to tame *over*-parameterized models, and on a small model they mostly just impede fitting. So I'll dial augmentation *down*, drop the side heads and label smoothing, and shrink the range of crop distortions. And specifically I'll put little or no weight decay on the depthwise filters: they hold so few parameters that they contribute almost nothing to overfitting, and L2-penalizing them just starves the one part of the network that does the spatial filtering. Standard optimizer of the day — RMSProp with asynchronous SGD — does the rest.

Let me write the block and the network, and tie each piece back to the reasoning. The depthwise conv is just a normal conv with the number of groups equal to the number of input channels — that is what "one filter per channel, no cross-channel mixing" means in a conv API: with groups = in_channels each output channel sees exactly one input channel.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel, stride, padding, groups=1):
    # The standard primitive: conv -> batch-norm -> ReLU. Used for the stem,
    # the depthwise step, and the pointwise step alike.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableBlock(nn.Module):
    """Factor a standard conv into (1) spatial filtering, per channel, then
    (2) channel mixing, with no spatial extent. Cost adds instead of multiplies."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        # Stride lives here so a strided block also downsamples. Cost ~ Dk^2 * M * Df^2.
        self.depthwise = conv_bn_relu(in_ch, in_ch, kernel=3, stride=stride,
                                      padding=1, groups=in_ch)
        # Pointwise: 1x1 conv combines the M channels into N. This is a dense
        # GEMM (no im2col), and carries ~all of the layer's compute. Cost ~ M * N * Df^2.
        self.pointwise = conv_bn_relu(in_ch, out_ch, kernel=1, stride=1, padding=0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module):
    # Each tuple is (output_channels, stride) for one separable block.
    # Strides of 2 fold downsampling into the depthwise conv: 112->56->28->14->7.
    # The run of five 512->512 stride-1 blocks is the deep core where features mature.
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
           (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
           (1024, 2), (1024, 1)]

    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        # width multiplier alpha: scale every channel count, cost/params drop ~alpha^2.
        def c(ch):
            return max(8, int(ch * width_mult))

        # First layer is a FULL conv: only 3 input channels, nothing to separate yet.
        in_ch = c(32)
        self.stem = conv_bn_relu(3, in_ch, kernel=3, stride=2, padding=1)

        blocks = []
        for out_ch, stride in self.cfg:
            out_ch = c(out_ch)
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)   # 7x7 -> 1x1 global average
        self.fc = nn.Linear(in_ch, num_classes)  # no nonlinearity before softmax

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

Resolution multiplier ρ doesn't need a line of code — it's just the input image size: feed 224, 192, 160, or 128 and every internal map shrinks with it, scaling compute by ρ² and leaving the weights alone.

The whole causal chain, end to end: a standard conv's cost is multiplicative in D_K²·M·N·D_F² because the same kernel both filters spatially and mixes channels; split those two jobs into a per-channel depthwise filter (cost D_K²·M·D_F², drops N) followed by a 1×1 pointwise mix (cost M·N·D_F², drops D_K²), and the costs add instead of multiply, giving a 1/N + 1/D_K² ≈ 1/9 reduction for 3×3 kernels; almost all the remaining compute is in the 1×1 convs, which are dense GEMMs that run fast on real hardware, so the savings are real latency rather than just paper FLOPs; stack these blocks (with a full-conv stem because the input is only 3 channels) into a simple, branch-free 28-layer pyramid with BN+ReLU throughout; and expose two budget dials — a width multiplier α that scales channels for ~α² cost and parameter savings, and a resolution multiplier ρ set by the input size for ~ρ² compute savings — trained from scratch with light regularization because such small models barely overfit.
