A time series hands us, at each step, an $M$-channel vector observed over a window of length $L$, and the same input underlies five tasks — long- and short-term forecasting, imputation, anomaly detection, and classification — yet the field has splintered into task-specific architectures and, more strikingly, has quietly retired convolution altogether. The charge against it is precise: a 1-D temporal convolution has too small an effective receptive field. A width-3 kernel reaches three steps, stacking layers grows that field only linearly (geometrically with dilation, but with holes), so relating a point to something a few hundred steps back demands an absurd depth, while attention relates any two positions in a single layer. On that argument the Transformer forecasters (Informer, Autoformer, FEDformer, PatchTST, the inverted iTransformer) and the MLP models (DLinear, TSMixer, TimeMixer) took the benchmarks, and the strongest task-general baseline, TimesNet, even reaches *for* convolution but only after reshaping the 1-D series into a 2-D period grid — convolving in a borrowed space rather than along native time. The trouble is that I have just watched the identical drama resolve the other way in vision: ViT displaced CNNs, and then ConvNeXt showed a *modernized* pure ConvNet — large depthwise kernels, pointwise channel mixing, an inverted-bottleneck FFN — matches Swin. The lesson there was not "attention is necessary" but "the old ConvNet was under-built." So the time-series verdict is most likely the same mistake: nobody modernized the temporal ConvNet, and the receptive-field complaint is really a complaint about *small* kernels and *traditional* blocks, not about convolution as such. What we need is a single convolutional backbone that achieves a genuinely large temporal ERF, models the multivariate axis explicitly (the cross-variable dependence channel-independent forecasters deliberately drop), serves all five tasks by swapping only a head, stays at convolutional cost, and trains from scratch on modest benchmark sizes without overfitting.

I propose ModernTCN, a pure-convolution backbone that ports the modern-ConvNet recipe to time and then repairs the one place where the naive port breaks. The receptive-field objection falls first, and cheaply, to a *large depthwise* 1-D convolution. Depthwise means one kernel per channel with no cross-channel coupling, so a size-31 or size-51 kernel costs only $\text{channels}\times K$ parameters instead of $\text{channels}^2\times K$ — the channel-by-channel coupling is exactly what makes a dense large kernel unaffordable, and depthwise removes it, giving every channel a large temporal ERF in one layer with no depth and no dilation holes. But a large kernel trained from scratch optimizes poorly: a single wide kernel is asked to be both a long-range integrator and a sharp local detector, and the fine two-or-three-step detail gets smeared. The fix is structural re-parameterization. I train the depthwise conv as two parallel branches — a large branch of size $K$ and a small branch of size $k<K$, each its own conv followed by BatchNorm, summed at the output:
$$y = \mathrm{BN}_\text{large}\!\big(\mathrm{conv}_K(x)\big) + \mathrm{BN}_\text{small}\!\big(\mathrm{conv}_k(x)\big).$$
The small branch hands the local detail a clean, easy-to-train gradient path while the large branch learns the long-range support, and because they are additive they do not fight. The payoff is an inference-time identity that erases the overhead entirely. A conv followed by BatchNorm is itself an affine map that folds into the conv weights: scale the kernel by $\gamma/\sqrt{\sigma^2+\epsilon}$ and shift the bias to $\beta - \mu\,\gamma/\sqrt{\sigma^2+\epsilon}$. Two parallel convs of different size also sum into one conv if the small kernel is zero-padded up to the large kernel's width and added. So after training I fuse large+BN and small+BN into a single equivalent large depthwise kernel — the deployed model is one convolution, numerically identical to the trained two-branch form. This is the `ReparamLargeKernelConv`: trained as two branches, merged to one.

The depthwise conv mixes only along time, so it is half a model; something else must mix channels, and here is where the naive ConvNeXt port breaks and the real design lives. In vision there is one channel axis and a single 1×1 conv mixes it. But a multivariate series has *two* axes that both deserve the name "channel": the physical **variable** axis $M$ (the sensors, the electrodes, the spectral bands) and the learned **feature** axis $D$ into which each variable is embedded. Flattening them into one $M\cdot D$ axis and running a single 1×1 conv conflates two genuinely different operations — mixing across $D$ within a variable is the ordinary per-variable feed-forward, while mixing across $M$ is cross-variable dependence, a separate modelling decision and precisely the one channel-independent models discard. Conflating them would forfeit any independent control and any way to even ask whether cross-variable mixing helps. So I keep the working tensor as a true 4-D object $[B, M, D, N]$ (batch, variable, feature, time) and use two distinct grouped pointwise FFNs. ConvFFN1 mixes features within each variable: view the tensor as $[B, M\cdot D, N]$ and apply a 1×1 conv with $\text{groups}=M$, which partitions the $M\cdot D$ channels into $M$ groups of $D$ and mixes only within a group — the per-variable inverted-bottleneck FFN ($D\to r\!\cdot\!D\to D$ with a GELU between, weights shared across variables), the direct analogue of ConvNeXt's FFN. ConvFFN2 mixes across variables per feature: permute to $[B, D, M, N]$, reshape to $[B, D\cdot M, N]$ so each consecutive block of $M$ channels holds one feature's variables, and apply a 1×1 conv with $\text{groups}=D$, which mixes only within each feature's $M$ variables — a learnable, per-feature interaction across the variables applied at every time step. This is the cross-variable operator that distinguishes the design from every channel-independent model, and the grouped-conv trick makes both FFNs cheap and shares weights in exactly the right places (across variables for ConvFFN1, across features for ConvFFN2) with no loop.

A block is therefore four operators with a clean division of labour, wrapped in a residual so each block learns a correction: the depthwise large-kernel conv mixes **time**; a BatchNorm over the feature dimension keeps activations scaled; ConvFFN1 mixes **features**; ConvFFN2 mixes **variables**. The norm is BatchNorm rather than LayerNorm on purpose — time series carry outliers (a glitch, a regime jump), and per-token LayerNorm gets dragged when an outlier lands in its token, whereas BatchNorm normalizes each feature across the batch and dilutes the outlier, the same conclusion ConvNeXt and PatchTST reached. Time, feature, variable: three axes, three dedicated mixers, never conflated. The permute bookkeeping for ConvFFN2 is symmetric and lossless — $[B,M,D,N]\to[B,D,M,N]\to[B,D\cdot M,N]\to$ grouped conv $\to[B,D,M,N]\to[B,M,D,N]$ — which is where it would silently break if done carelessly. The front end lifts each scalar variable-series into the $D$-dimensional feature space with a patch-embedding stem, `Conv1d(1, D, kernel=patch_size, stride=patch_stride)` applied per variable by folding the variable axis into the batch so the *same* stem weights process every variable. That gives channel-independence and data efficiency at the embedding (the variable count need not be fixed) and, following PatchTST, lets each patch carry local shape while cutting temporal length so the large kernel spans more real time per step; unlike PatchTST, channels are *not* kept independent through the backbone, since ConvFFN2 reintroduces cross-variable mixing in every block. When $\text{patch\_stride}<\text{patch\_size}$ the tail is replicate-padded by $\text{patch\_size}-\text{patch\_stride}$ so the boundary carries the real edge value rather than an invented zero, and the patch count is $N=\text{seq\_len}//\text{patch\_stride}$. Depth then compounds the receptive field across scales like a vision pyramid: blocks are stacked in stages, and between stages a BatchNorm and a strided `Conv1d(dims_i, dims_{i+1}, kernel=r, stride=r)` fold the length by $r$ and grow the width, so each subsequent block's large kernel covers $r\times$ more real time and two or three stages yield an enormous ERF cheaply (again replicate-padding the tail when length is not divisible by $r$). The stack is kept deliberately shallow — a couple of blocks, two stages — because the benchmark datasets, UEA especially, are small and the ERF comes from the large kernel and the downsampling, not raw depth. For classification the head is the simplest faithful thing: after the stages I have $[B, M, D, N_\text{final}]$, apply GELU and dropout, flatten the entire $M\cdot D\cdot N_\text{final}$ representation, and project with a single `Linear` to `num_class`. The flatten-and-project draws the class boundary but no longer carries the burden of cross-channel modelling, because ConvFFN2 already mixed the variables inside every block; the replicate-padding means no spurious zeros and no explicit mask is needed. For the other four tasks the backbone is identical and only the head changes — the contribution is the block, not the head. Every objection is met: the ERF is large (depthwise kernel plus stage downsampling), the wide kernel is trainable (small-kernel re-param that fuses away), cross-variable dependence is modelled (ConvFFN2) and kept independently dialable, the model is task-general and avoids quadratic attention, and overfitting is guarded by shallow stacks, shared stem weights, and grouped convs — so the explicit cross-variable ConvFFN should help most exactly where channel-independent and head-only-mixing models stall, on multivariate signals whose information is cross-sensor covariance.

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
    kernel = conv.weight
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1)
    return kernel * t, bn.bias - bn.running_mean * bn.weight / std


class ReparamLargeKernelConv(nn.Module):
    """Depthwise large-kernel temporal conv: large+small branches (conv+BN), summed; fuse at inference."""

    def __init__(self, channels, large_size, small_size, groups):
        super().__init__()
        self.large_size, self.small_size = large_size, small_size
        self.large = conv_bn(channels, channels, large_size, 1, large_size // 2, groups, bias=False)
        self.small = conv_bn(channels, channels, small_size, 1, small_size // 2, groups, bias=False)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):                # merged single kernel (inference)
            return self.lkb_reparam(x)
        return self.large(x) + self.small(x)            # two branches (training)

    def merge_kernel(self):
        eq_k, eq_b = fuse_bn(self.large.conv, self.large.bn)
        sk, sb = fuse_bn(self.small.conv, self.small.bn)
        pad = (self.large_size - self.small_size) // 2
        eq_k, eq_b = eq_k + F.pad(sk, [pad, pad]), eq_b + sb
        merged = nn.Conv1d(self.large.conv.in_channels, self.large.conv.out_channels, self.large_size,
                           padding=self.large_size // 2, groups=self.large.conv.groups, bias=True)
        merged.weight.data, merged.bias.data = eq_k, eq_b
        self.lkb_reparam = merged
        self.__delattr__('large'); self.__delattr__('small')


class Block(nn.Module):
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
        x = self.dw(x.reshape(B, M * D, N))
        x = self.norm(x.reshape(B, M, D, N).reshape(B * M, D, N))
        x = x.reshape(B, M, D, N).reshape(B, M * D, N)
        x = self.ffn1drop2(self.ffn1pw2(self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))))
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop2(self.ffn2pw2(self.ffn2act(self.ffn2drop1(self.ffn2pw1(x)))))
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
        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0 and self.patch_size != self.patch_stride:
                pad = self.patch_size - self.patch_stride
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad)], dim=-1)
            elif i > 0 and N % self.downsample_ratio != 0:
                pad = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = self.stages[i](x.reshape(B, M, D_, N_))
        return x

    def classification(self, x_enc, x_mark_enc):
        x = x_enc.permute(0, 2, 1)
        x = self.forward_feature(x)
        x = self.class_dropout(F.gelu(x))
        x = x.reshape(x.shape[0], -1)
        return self.head_class(x)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
