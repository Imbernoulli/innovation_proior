Let me start from the thing that actually bothers me. Convolution runs vision, runs speech, was the
original tool for time series — and yet on the current time-series benchmarks it has been quietly
retired. The story everyone tells is that a 1-D temporal convolution has too small an effective receptive
field: a kernel of width 3 reaches three steps, stacking layers grows the field only linearly (or
geometrically with dilation, but with holes), so to relate a point to something a few hundred steps back
you need an absurd depth, and meanwhile attention relates any two points in one layer. So attention and
MLP-mixers took the field. But I have just watched the exact same drama play out in vision and resolve the
other way: ViT displaced CNNs, and then ConvNeXt showed that a *modernized* pure ConvNet — large
depthwise kernels, pointwise channel mixing, an inverted-bottleneck FFN — matches Swin. The lesson there
was not "attention is necessary"; it was "the old ConvNet was under-built." So my suspicion is that the
time-series verdict is the same mistake: nobody modernized the temporal ConvNet. The receptive-field
complaint is really a complaint about *small* kernels and *traditional* conv blocks, not about
convolution as such. If I take the ConvNeXt recipe seriously and port it to the time axis, I should get a
large ERF and competitive accuracy at convolutional cost. Let me try to derive that port from scratch and
see where the naive transfer breaks, because the breakage is where the real design is.

First, the receptive field, since that is the whole objection. How do I make a 1-D conv see hundreds of
steps without going deep? The cheap answer is staring at me from ConvNeXt: use a *large* kernel, and make
it *depthwise*. A depthwise conv applies one kernel per channel — no cross-channel mixing — so a kernel of
size 51 costs `channels × 51` parameters, not `channels² × 51`. That is what makes large kernels
affordable: the cost that kills a dense large kernel is the channel-by-channel coupling, and depthwise
simply removes it. So a single depthwise conv with a large kernel gives every channel a large ERF in one
layer, directly answering the "ERF too small" objection without depth or dilation holes. Good — that is
the temporal mixer. But it raises the obvious follow-up: a depthwise conv mixes *only* along time and
never across channels, so by itself it is half a model. I need a separate operator to mix channels. That
separation — depthwise for time, pointwise (1×1) for channels — is exactly the modern-conv decomposition,
and it is going to turn out to be more subtle here than in vision, so let me hold it and first make the
large kernel actually trainable.

Because large kernels have a known pathology: trained from scratch they optimize poorly. The gradient has
to discover structure across a wide support all at once, and the fine local detail — a two- or three-step
wiggle — is easy for a wide kernel to under-weight or smear, since a single broad kernel is being asked to
be both a long-range integrator and a sharp local detector. The fix that the re-parameterization line
established (RepLKNet in vision) is to train the large kernel *in parallel with a small kernel*: a large
branch (size `K`) and a small branch (size `k < K`), each its own conv followed by BatchNorm, summed at
the output. The small branch gives the local detail a clean, easy-to-train gradient path; the large branch
learns the long-range support; they do not fight because they are additive. And there is a free lunch at
inference: a conv followed by BatchNorm is itself an affine map that folds into the conv weights (scale the
kernel by `gamma/sqrt(var+eps)`, shift the bias by `beta - mean·gamma/sqrt(var+eps)`), and two parallel
convs of different size sum into one conv if I zero-pad the small kernel up to the large kernel's width and
add. So after training I fuse large+BN and small+BN into a single equivalent large depthwise conv — the
deployed model is one kernel, no overhead, but it was *trained* as two branches. This is the
`ReparamLargeKernelConv`: train two branches, merge to one. I will keep both branches explicit during
training because that is what the forward pass computes; the merge is an identity I can apply later.

Now back to channel mixing, and here is where the naive ConvNeXt port breaks and the real contribution
lives. In vision there is one channel axis — the `D` feature maps — and a 1×1 conv mixes across it. But a
multivariate time series has *two* axes that both deserve to be called "channels": the physical **variable**
axis `M` (the sensors, the MEG electrodes, the spectral wavelengths) and the learned **feature** axis `D`
(the embedding dimension I lift each variable into). A naive port would flatten these into one `M·D` axis
and run a single 1×1 conv over it — but that conflates two genuinely different kinds of mixing. Mixing
across `D` within a variable is the ordinary feed-forward "process this variable's features" operation.
Mixing across `M` is *cross-variable* dependence — how the sensors covary — which is a completely separate
modelling decision, the one channel-independent forecasters like PatchTST deliberately *dropped*. If I
fuse the axes I cannot control these independently, and I cannot even ask whether cross-variable mixing
helps. So I will keep the working tensor as a genuine 4-D object, `[B, M, D, N]` (batch, variable, feature,
time), and use *two distinct* pointwise FFNs.

Let me design the two FFNs by being explicit about which axis each touches. ConvFFN1 mixes features within
each variable: I want variables to stay independent here, so I view the tensor as `[B, M·D, N]` and use a
1×1 conv with `groups = M`. Grouping by `M` means the `M·D` channels are partitioned into `M` groups of
`D`, and the conv mixes only within a group — exactly "mix the `D` features of variable `m`, for each `m`,
with shared weights across variables." That is the per-variable channel-MLP, the inverted-bottleneck FFN
(`D → r·D → D` with a GELU between), the direct analogue of ConvNeXt's pointwise FFN. ConvFFN2 mixes across
variables per feature: now I want features to stay independent and variables to interact. So I permute to
put the variable axis where the conv will mix it — view as `[B, D·M, N]` with the data arranged so that the
`D·M` channels are `D` groups of `M`, and use a 1×1 conv with `groups = D`. Grouping by `D` means the conv
mixes only within each feature's `M` variables — "for feature `d`, mix across all `M` variables." This is
the cross-variable operator, and it is the thing that distinguishes this design from every
channel-independent model: a *learnable, per-feature interaction across the variables, applied at every
time step*. Crucially the grouped-conv trick makes both FFNs cheap and lets me share weights in the right
places (across variables for ConvFFN1, across features for ConvFFN2) without writing any loop.

So a block is now four operators with a clean division of labour: the depthwise large-kernel conv mixes
**time**; a BatchNorm over the feature dimension keeps activations scaled (BatchNorm, not LayerNorm,
because time series carry outliers — a glitch or regime jump — and per-token LayerNorm gets dragged by an
outlier landing in its token, while BatchNorm normalizes each feature across the batch and dilutes it, the
same reasoning ConvNeXt and PatchTST landed on); ConvFFN1 mixes **features**; ConvFFN2 mixes **variables**;
and a residual connection wraps the whole block so each block learns a correction. Time, feature, variable
— three axes, three dedicated mixers, never conflated. That separation is the architecture. Let me sanity
check the reshape bookkeeping for ConvFFN2 because the permutes are where this silently breaks: I have
`[B, M, D, N]`; permute to `[B, D, M, N]`, reshape to `[B, D·M, N]` so each consecutive block of `M`
channels is one feature's variables; grouped conv with `groups = D`; reshape back to `[B, D, M, N]`,
permute back to `[B, M, D, N]`. Symmetric, no axis lost.

Now the front end. I need to lift each scalar variable-series into the `D`-dimensional feature space, and I
want a patch embedding both to cut the temporal length (so the depthwise conv's large kernel spans more
real time per step) and because patches carry local shape, the lesson from PatchTST. The stem is a
`Conv1d(1, D, kernel=patch_size, stride=patch_stride)` applied per variable — I fold the variable axis into
the batch so the *same* stem weights process every variable. That is channel-independence at the embedding
(shared weights, data-efficient, the variable count need not be fixed), but unlike PatchTST I do *not* keep
channels independent through the whole backbone — ConvFFN2 reintroduces cross-variable mixing in every
block. When `patch_stride < patch_size` the patches overlap and the last window would miss the final
steps, so I replicate-pad the tail by `patch_size − patch_stride` before the stem conv — replicate, not
zero, so the boundary carries the real edge value rather than an invented zero. The patch count is
`N = seq_len // patch_stride`.

I also want depth that compounds the receptive field across scales, the way a vision pyramid does. So I
stack the blocks in **stages**, and between stages I downsample: a BatchNorm then a strided
`Conv1d(dims[i], dims[i+1], kernel=r, stride=r)` that folds the temporal length by `r` and grows the
feature width. After downsampling, each block's large kernel covers `r×` more real time, so two or three
stages give an enormous ERF cheaply. When the length is not divisible by `r` I replicate-pad the tail
before downsampling, same edge-faithful reasoning. I keep the stack shallow — a couple of blocks per stage,
two stages — because the benchmark datasets (UEA especially) are small and a deep stack overfits; the ERF
comes from the large kernel and the downsampling, not from raw depth.

The head depends on the task; for classification I have, after the stages, `[B, M, D, N_final]`, and I need
`[B, num_class]`. The faithful canonical head is the simplest thing: a GELU, dropout, flatten the entire
`M · D · N_final` representation to one vector, and a single `Linear(M · D · N_final, num_class)`. The
flatten-and-project draws the class boundary, but — and this is the payoff of the design — it is no longer
carrying the burden of cross-channel modelling, because ConvFFN2 has already mixed the variables inside
every block. The padding tail is handled by the replicate-padding (no spurious zeros) plus the learned
head, so no explicit mask multiply is needed in this head. For the other four tasks the backbone is
identical and only the head changes (a flatten-linear to the horizon for forecasting, a projection back to
the channels for imputation/anomaly), which is the "one backbone, swap the head" generality I wanted; the
contribution is the block, not the head.

Let me step back and check I have actually answered the objections rather than rearranged furniture. ERF
too small? Solved by the large depthwise kernel plus stage downsampling — a genuinely large temporal
receptive field in few layers, at depthwise cost. No cross-variable modelling? Solved by ConvFFN2, the
dedicated per-feature cross-variable mixer that channel-independent models lacked — and kept separable from
feature mixing so I can dial it independently. Large kernels untrainable? Solved by the small-kernel reparam
branch that fuses away at inference. Task-general and efficient? One convolutional backbone, head-swapped,
no quadratic attention. Overfitting on small data? Shallow stacks, shared stem weights, grouped convs that
share parameters across variables/features. So the bet is concrete: a modern pure-convolution backbone
should match or beat TimesNet and the Transformer/MLP models across all five tasks — and on multivariate
classification specifically, the explicit cross-variable ConvFFN should help exactly where
channel-independent and head-only-mixing models are weakest (datasets whose signal is cross-sensor
covariance, like the MEG FaceDetection task).

Let me write it as the code I would ship — the reparam large-kernel depthwise conv, the block with its two
distinct grouped FFNs, the staged backbone with the patch stem and downsampling, and the classification
head — kept in the train-time two-branch form (the merge is an inference-time identity):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_ch, out_ch, kernel_size, stride, padding, groups, dilation=1, bias=False):
    seq = nn.Sequential()
    seq.add_module('conv', nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias))
    seq.add_module('bn', nn.BatchNorm1d(out_ch))
    return seq


def fuse_bn(conv, bn):
    # inference-time identity: fold a conv+BN into an equivalent conv (kernel, bias)
    kernel = conv.weight
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1)
    return kernel * t, bn.bias - bn.running_mean * bn.weight / std


class ReparamLargeKernelConv(nn.Module):
    """Depthwise large-kernel temporal conv trained as large+small branches (each conv+BN), summed.
    The two branches fuse into one equivalent large kernel at inference via merge_kernel()."""

    def __init__(self, channels, large_size, small_size, groups):
        super().__init__()
        self.large_size = large_size
        self.small_size = small_size
        self.large = conv_bn(channels, channels, large_size, 1, large_size // 2, groups, bias=False)
        self.small = conv_bn(channels, channels, small_size, 1, small_size // 2, groups, bias=False)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):                # merged single kernel (inference)
            return self.lkb_reparam(x)
        return self.large(x) + self.small(x)            # two branches (training)

    def merge_kernel(self):                              # inference-time fusion (not used in training)
        eq_k, eq_b = fuse_bn(self.large.conv, self.large.bn)
        sk, sb = fuse_bn(self.small.conv, self.small.bn)
        pad = (self.large_size - self.small_size) // 2
        eq_k = eq_k + F.pad(sk, [pad, pad])
        eq_b = eq_b + sb
        merged = nn.Conv1d(self.large.conv.in_channels, self.large.conv.out_channels,
                           self.large_size, padding=self.large_size // 2,
                           groups=self.large.conv.groups, bias=True)
        merged.weight.data, merged.bias.data = eq_k, eq_b
        self.lkb_reparam = merged
        self.__delattr__('large'); self.__delattr__('small')


class Block(nn.Module):
    """DWConv (time) -> BN -> ConvFFN1 (feature mix, groups=nvars) -> ConvFFN2 (variable mix,
    groups=dmodel) -> residual. nvars=M variables, dmodel=D features/variable."""

    def __init__(self, large_size, small_size, dmodel, dff, nvars, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(nvars * dmodel, large_size, small_size, groups=nvars * dmodel)
        self.norm = nn.BatchNorm1d(dmodel)
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn1drop1, self.ffn1drop2 = nn.Dropout(drop), nn.Dropout(drop)
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.ffn2drop1, self.ffn2drop2 = nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x):                                # x: [B, M, D, N]
        inp = x
        B, M, D, N = x.shape
        x = self.dw(x.reshape(B, M * D, N))              # depthwise temporal conv
        x = self.norm(x.reshape(B, M, D, N).reshape(B * M, D, N))
        x = x.reshape(B, M, D, N).reshape(B, M * D, N)
        x = self.ffn1drop2(self.ffn1pw2(self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))))  # feature mix
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop2(self.ffn2pw2(self.ffn2act(self.ffn2drop1(self.ffn2pw1(x)))))  # variable mix
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)
        return inp + x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars, drop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(large_size, small_size, dmodel, dmodel * ffn_ratio,
                                           nvars, drop) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Model(nn.Module):
    """ModernTCN classifier: channel-independent patch stem, staged large-kernel blocks with
    cross-feature + cross-variable ConvFFNs, flatten-and-project head."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.nvars = configs.enc_in
        self.num_class = configs.num_class
        self.patch_size = getattr(configs, 'patch_size', 8)
        self.patch_stride = getattr(configs, 'patch_stride', 4)
        self.downsample_ratio = getattr(configs, 'downsample_ratio', 2)
        ffn_ratio = getattr(configs, 'ffn_ratio', 1)
        num_blocks = getattr(configs, 'num_blocks', [1, 1])
        large_size = getattr(configs, 'large_size', [31, 29])
        small_size = getattr(configs, 'small_size', [5, 5])
        dims = getattr(configs, 'dims', [64, 64])
        drop = getattr(configs, 'dropout', 0.1)
        self.num_stage = len(num_blocks)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv1d(1, dims[0], self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(dims[0])))
        for i in range(self.num_stage - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], self.downsample_ratio, stride=self.downsample_ratio)))

        self.stages = nn.ModuleList([
            Stage(ffn_ratio, num_blocks[s], large_size[s], small_size[s], dims[s], self.nvars, drop)
            for s in range(self.num_stage)])

        patch_num = self.seq_len // self.patch_stride
        d_model = dims[self.num_stage - 1]
        fold = self.downsample_ratio ** (self.num_stage - 1)
        self.head_nf = d_model * (patch_num // fold if patch_num % fold == 0 else patch_num // fold + 1)
        self.class_dropout = nn.Dropout(getattr(configs, 'class_dropout', 0.0))
        self.head_class = nn.Linear(self.nvars * self.head_nf, self.num_class)

    def forward_feature(self, x):                        # x: [B, M, L]
        B, M, L = x.shape
        x = x.unsqueeze(-2)                              # [B, M, 1, L]
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0 and self.patch_size != self.patch_stride:
                pad = self.patch_size - self.patch_stride
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad)], dim=-1)   # replicate-pad tail
            elif i > 0 and N % self.downsample_ratio != 0:
                pad = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = self.stages[i](x.reshape(B, M, D_, N_))
        return x

    def classification(self, x_enc, x_mark_enc):
        x = x_enc.permute(0, 2, 1)                       # [B, M, L]
        x = self.forward_feature(x)                     # [B, M, D, N_final]
        x = self.class_dropout(F.gelu(x))
        x = x.reshape(x.shape[0], -1)
        return self.head_class(x)                        # [B, num_class]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```

The causal chain, start to finish. Convolution was retired from time series on a single charge — too
small an effective receptive field — but the same charge was levelled at CNNs in vision and overturned by
ConvNeXt's modernization, so the verdict was likely "the conv block was under-built," not "convolution is
wrong." Porting the modern-conv recipe gives a *large depthwise kernel* for a big temporal ERF at
per-channel cost, made trainable by a parallel small-kernel re-param branch that fuses to one kernel at
inference. The naive port breaks on the channel axis, because a multivariate series has two channel-like
axes — the learned feature dimension and the physical variable dimension — that demand separate mixers; so
the block keeps a 4-D tensor and uses two distinct grouped 1×1 FFNs, ConvFFN1 (groups=nvars) for
cross-feature mixing and ConvFFN2 (groups=dmodel) for the cross-variable mixing that channel-independent
models drop, with the depthwise conv handling time and BatchNorm chosen over LayerNorm against outliers. A
channel-independent patch stem and staged strided downsampling compound the ERF cheaply, shallow depth
guards against overfitting on small benchmarks, and a flatten-and-project head reads off the class — the
real cross-channel work already done inside every block. The result is one pure-convolution backbone, head
swapped per task, that targets the cross-variable weakness of the channel-independent and head-only-mixing
models exactly where it matters most.
