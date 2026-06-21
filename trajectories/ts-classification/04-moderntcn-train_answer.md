TimesNet landed almost exactly where I aimed it, and the one place it did not improve now defines the next move. Handwriting jumped to 0.3353 (from PatchTST's 0.2541 and the floor's 0.2306) — the ladder's biggest gain, vindicating cross-period 2-D structure plus mask-aware pooling on the phase-recurrent gestures. EthanolConcentration reached 0.3194, clearing both PatchTST (0.2852) and the floor (0.2890), because it never per-window-normalizes away the slow spectral trend. But FaceDetection sat at 0.6745 — *below* PatchTST (0.6853) and the floor (0.6822), a small but real loss, exactly as I predicted: TimesNet mixes channels only in the value embedding and the flatten head, never inside the encoder, so on the one dataset whose entire signal is cross-sensor covariance across ~144 MEG channels it trades a little channel-friendliness for cross-period structure it needs less. And every rung has now mixed channels the same lazy way — a static linear combination in the head — and FaceDetection has refused to move past the high-0.68 ceiling for any of them. None of the three has a *dedicated, learnable stage whose job is to mix information across the variables at each temporal position*. That is the gap to attack, without giving back what TimesNet won on the other two.

I propose **ModernTCN**: a pure-convolution encoder that makes time mixing, feature mixing, and variable mixing three *separate* operators inside every block. Start with the temporal side, because I also want to push past TimesNet's strength, not just patch its weakness. TimesNet went to 2-D precisely because a sane-width 1-D kernel only reaches its immediate neighbours — same-phase points one period apart never land in one kernel unless the kernel is as wide as the period. But that quietly concedes wide kernels are impossible, and I question it: a *large depthwise* kernel is cheap, one kernel per channel, so a width-31 or width-51 kernel costs only $\text{channels}\times\text{kernel}$ parameters rather than $\text{channels}^2\times\text{kernel}$. A depthwise kernel spanning tens of timesteps gives each channel a genuinely large effective receptive field in a single layer — the same long-range reach TimesNet manufactured with 2-D — but kept in the natural 1-D layout, where the padding mask and variable length are trivial. A large kernel does have a training pathology: it is hard to optimize from scratch, and fine local detail (a two- or three-step wiggle that distinguishes Handwriting gestures) is easy for a wide kernel to smear. The structural-reparameterization fix is to train the large kernel *in parallel with a small kernel* — a large branch (size 31) and a small branch (size 5), each its own depthwise conv with its own BatchNorm, summed at the output. The small branch guarantees fine local detail a clean, easy-to-train path while the large branch learns the long-range support; at inference the two branches and their BatchNorms fuse into one equivalent kernel (small kernel zero-padded up to the large width and added), so deployment is a single depthwise conv with no overhead. I keep both branches explicit in training because that is what the forward pass actually computes — the fusion is an inference-time identity, not a change to the function.

The channel axis is where the cross-variable idea is made real, and it is the subtlest part. Lift each variable to a feature vector of width $D$, so the working tensor is $[\text{B},M,D,N]$ — batch, $M$ variables, $D$ features per variable, $N$ temporal positions. The depthwise temporal conv runs over $M\cdot D$ grouped channels (genuinely depthwise, $\text{groups}=M\cdot D$), mixing only along time, never across variables or features — keeping the temporal operator clean and per-stream. Then two *different* pointwise mixers, and the distinction between them is the heart of the design. **ConvFFN1** mixes across the **feature** dimension $D$ within each variable: a grouped $1\times1$ conv with $\text{groups}=M$, so each variable's features mix among themselves but variables stay separate — the per-variable channel MLP, the standard feature-mixing FFN. **ConvFFN2** mixes across the **variable** dimension $M$: permute so the variable axis becomes the conv's channel axis and group by $D$ ($\text{groups}=D$), so each feature index is mixed across all $M$ variables while features stay separate. ConvFFN2 is the explicit cross-variable operator every earlier rung lacked — a learnable, per-feature interaction across the MEG sensors, applied at every temporal position, which is exactly the structure FaceDetection's decision lives in. Both FFNs are the usual two-layer $1\times1$ conv with a GELU and dropout between, expanding $D\to \text{ffn\_ratio}\cdot D\to D$. A block is: depthwise large-kernel temporal conv, BatchNorm, ConvFFN1 (feature mix), ConvFFN2 (variable mix), wrapped in a residual connection. That decomposition — separable temporal conv, then feature mixing, then variable mixing — cleanly separates the three kinds of structure (time, feature, variable) the earlier rungs entangled.

A few mechanics decide whether this behaves on the heterogeneous UEA shapes. The stem embeds each variable independently — a `Conv1d(1, D, kernel=patch_size, stride=patch_stride)` applied with the variable axis folded into the batch, so the same weights process every variable (the channel-independent, shared-weight stem; PatchTST's data-efficiency argument still holds, and matters because FaceDetection has 144 variables and little data per class). With $\text{patch\_stride}<\text{patch\_size}$ I replicate-pad the tail by $\text{patch\_size}-\text{patch\_stride}$ so the last timestep is never dropped, the same edge-faithful padding the decomposition and PatchTST used; the patch count is $N=\text{seq\_len}//\text{patch\_stride}$. Between stages, a BatchNorm then a strided `Conv1d(dims[i], dims[i+1], kernel=downsample_ratio, stride=downsample_ratio)` $r$-folds the temporal length and grows the feature width, the standard convolutional pyramid that lets the large-kernel reach compound across scales (with a replicate-pad of the tail when the length is not divisible). The stack is shallow — a couple of blocks per stage — because the UEA datasets are small and a deep stack would overfit, the same restraint that sized the earlier rungs. Normalization is BatchNorm, not LayerNorm, and deliberately so: time series carry outliers, LayerNorm normalizes within a token and gets dragged by an outlier landing there, while BatchNorm normalizes each feature across the batch and dilutes it — the same reason PatchTST used BatchNorm, doubly right here because the whole architecture is convolutional. The depthwise branches each carry their own BatchNorm (which is what makes the train-time-two-branch / inference-time-one-kernel fusion exact), with a BatchNorm over the feature dimension after the depthwise conv.

The head keeps TimesNet's mask-awareness in spirit while adapting to the convolutional backbone. After the stages I have $[\text{B},M,D,N_\text{final}]$; the head applies a GELU and dropout, flattens the whole $M\cdot D\cdot N_\text{final}$ representation, and projects with a single `Linear(M · head_nf, num_class)`. That flatten-and-project draws the final decision, but the cross-variable ConvFFN2 inside every block has already done the real channel mixing, so the head no longer carries the entire burden the linear floor's head did. The one wrinkle is the mask: the stem's strided conv changes the temporal length from `seq_len` to $N_\text{final}$, so I cannot multiply by `x_mark_enc` at full resolution the way TimesNet did. The reference head relies instead on the *replicate*-padding (not zero-padding) of the tail before every conv, so the padded region carries the boundary value rather than a spurious zero, and the learned flatten head can suppress the constant tail; I keep the head faithful to that reference — GELU, dropout, flatten, linear — since that is exactly what produced ModernTCN's measured UEA results, and an ad-hoc mask multiply at a downsampled resolution would diverge from the canonical implementation without a principled gain. So the sharpest test is FaceDetection: ConvFFN2 is the first dedicated cross-variable operator on the ladder, and if explicit cross-variable mixing finally breaks the high-0.68 ceiling none of the three rungs could crack — while the large kernel holds EthanolConcentration's long-range spectral reach and the small-kernel branch protects Handwriting's fine strokes against the strided downsampling — then the recurring weakness really was the absence of a dedicated cross-channel stage.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_ch, out_ch, kernel_size, stride, padding, groups, dilation=1, bias=False):
    # a Conv1d followed by BatchNorm1d (the train-time form of a reparam branch)
    seq = nn.Sequential()
    seq.add_module('conv', nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias))
    seq.add_module('bn', nn.BatchNorm1d(out_ch))
    return seq


class ReparamLargeKernelConv(nn.Module):
    """Depthwise large-kernel temporal conv with a parallel small-kernel branch (train-time form).
    forward = large(conv+bn) + small(conv+bn); the two fuse to one kernel at inference."""

    def __init__(self, channels, large_size, small_size, groups):
        super(ReparamLargeKernelConv, self).__init__()
        self.large = conv_bn(channels, channels, large_size, stride=1,
                             padding=large_size // 2, groups=groups, bias=False)
        self.small = conv_bn(channels, channels, small_size, stride=1,
                             padding=small_size // 2, groups=groups, bias=False)

    def forward(self, x):
        return self.large(x) + self.small(x)


class Block(nn.Module):
    """ModernTCN block: depthwise large-kernel conv -> BN -> ConvFFN1 (feature mix) ->
    ConvFFN2 (variable mix), residual. nvars = M variables, dmodel = D features/variable."""

    def __init__(self, large_size, small_size, dmodel, dff, nvars, drop=0.1):
        super(Block, self).__init__()
        # depthwise over all M*D channels: mixes only along time
        self.dw = ReparamLargeKernelConv(nvars * dmodel, large_size, small_size, groups=nvars * dmodel)
        self.norm = nn.BatchNorm1d(dmodel)
        # ConvFFN1: cross-feature within each variable (groups = nvars keeps variables separate)
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        # ConvFFN2: cross-variable per feature (groups = dmodel keeps features separate)
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, M, D, N]
        inp = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)                                   # depthwise temporal conv
        x = x.reshape(B, M, D, N).reshape(B * M, D, N)
        x = self.norm(x)                                 # BN over feature dim
        x = x.reshape(B, M, D, N).reshape(B, M * D, N)
        # ConvFFN1: mix features within each variable
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)
        # ConvFFN2: mix across variables per feature
        x = x.permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)
        return inp + x                                   # residual


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars, drop=0.1):
        super(Stage, self).__init__()
        dff = dmodel * ffn_ratio
        self.blocks = nn.ModuleList(
            [Block(large_size, small_size, dmodel, dff, nvars, drop) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Model(nn.Module):
    """ModernTCN classification fill: channel-independent stem, multi-stage depthwise large-kernel
    blocks with cross-feature + cross-variable ConvFFNs, flatten-and-project head."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.nvars = configs.enc_in
        self.num_class = configs.num_class

        # config knobs (ModernTCN classification defaults)
        self.patch_size = getattr(configs, 'patch_size', 8)
        self.patch_stride = getattr(configs, 'patch_stride', 4)
        self.downsample_ratio = getattr(configs, 'downsample_ratio', 2)
        ffn_ratio = getattr(configs, 'ffn_ratio', 1)
        num_blocks = getattr(configs, 'num_blocks', [1, 1])
        large_size = getattr(configs, 'large_size', [31, 29])
        small_size = getattr(configs, 'small_size', [5, 5])
        dims = getattr(configs, 'dims', [64, 64])
        drop = getattr(configs, 'dropout', 0.1)
        self.class_drop = getattr(configs, 'class_dropout', 0.0)

        self.num_stage = len(num_blocks)

        # stem + downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stage - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=self.downsample_ratio,
                          stride=self.downsample_ratio),
            ))

        # backbone stages
        self.stages = nn.ModuleList([
            Stage(ffn_ratio, num_blocks[s], large_size[s], small_size[s],
                  dmodel=dims[s], nvars=self.nvars, drop=drop)
            for s in range(self.num_stage)])

        # head dimensions: patch_num after stem, then r-folded by each downsample
        patch_num = self.seq_len // self.patch_stride
        d_model = dims[self.num_stage - 1]
        if patch_num % (self.downsample_ratio ** (self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // (self.downsample_ratio ** (self.num_stage - 1))
        else:
            self.head_nf = d_model * (patch_num // (self.downsample_ratio ** (self.num_stage - 1)) + 1)

        self.act_class = F.gelu
        self.class_dropout = nn.Dropout(self.class_drop)
        self.head_class = nn.Linear(self.nvars * self.head_nf, self.num_class)

    def forward_feature(self, x):
        # x: [B, M, L]
        B, M, L = x.shape
        x = x.unsqueeze(-2)                              # [B, M, 1, L]
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:  # replicate-pad the tail before the stem conv
                    pad_len = self.patch_size - self.patch_stride
                    x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def classification(self, x_enc, x_mark_enc):
        # x_enc: [B, seq_len, enc_in] -> channel-first [B, M, L]
        x = x_enc.permute(0, 2, 1)
        x = self.forward_feature(x)                      # [B, M, D, N_final]
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = x.reshape(x.shape[0], -1)                    # flatten M x D x N
        x = self.head_class(x)                           # [B, num_class]
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
