**Problem (from step 3).** TimesNet won EthanolConcentration (0.3194) and Handwriting (0.3353) but was the
*worst* rung on FaceDetection (0.6745) — below PatchTST (0.6853) and the linear floor (0.6822). Across all
three rungs the channel axis was never first-class: the floor mixed channels only in one linear head, PatchTST
forbade in-encoder channel mixing (channel-independent), and TimesNet mixed only in the embedding/head. No rung
had a dedicated cross-variable operator, and FaceDetection — whose signal is cross-sensor covariance over ~144
MEG channels — stayed pinned at the high-0.68 ceiling.

**Key idea.** ModernTCN, a pure-convolution encoder that makes time, feature, and variable mixing three
separate operators inside every block. (1) A channel-independent stem patch-embeds each variable to width `D`.
(2) A **depthwise large-kernel temporal conv** (a large branch, e.g. size 31, parallel with a small branch,
size 5, each conv+BN, summed — the structural-reparam form; they fuse to one kernel at inference) gives each
channel a long receptive field in the natural 1-D layout, replacing TimesNet's 2-D period reshape and
PatchTST's patch attention. (3) **ConvFFN1** (1×1 conv, `groups=nvars`) mixes features within each variable;
(4) **ConvFFN2** (1×1 conv, `groups=dmodel`) mixes across variables per feature — the explicit cross-variable
stage every earlier rung lacked. Blocks are residual, stacked in stages with strided-conv downsampling; the head
is GELU + dropout + flatten + `Linear`.

**Why it works (and where it might not).** ConvFFN2 is a learnable per-feature interaction across all variables
at every temporal position — directly aimed at the FaceDetection ceiling that head-only channel mixing could
never crack. The large depthwise kernel keeps TimesNet's long-range temporal reach while the small-kernel branch
protects the fine local stroke detail Handwriting needs; BatchNorm (not LayerNorm) dilutes outliers; replicate-
padding before every conv avoids the spurious-zero artifact, so the flatten head needs no explicit mask multiply.
Risk: strided downsampling can coarsen Handwriting's fine gestures (the small-kernel branch must compensate), and
if FaceDetection does not move, the ceiling is set by something other than channel mixing.

**Scaffold edit / hyperparameters (ModernTCN classification defaults).** `patch_size=8`, `patch_stride=4`,
`downsample_ratio=2`, `ffn_ratio=1`, `num_blocks=[1,1]` (2 stages), `large_size=[31,29]`, `small_size=[5,5]`,
`dims=[64,64]`, `dropout=0.1`, depthwise groups `=nvars·dmodel`, ConvFFN1 `groups=nvars`, ConvFFN2
`groups=dmodel`; head `Linear(nvars · head_nf, num_class)`. Frozen protocol: RAdam, `lr 1e-3`, CrossEntropy,
patience 10. Faithful to the canonical ModernTCN classification reference (large+small reparam branches kept in
their train-time summed form).

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
