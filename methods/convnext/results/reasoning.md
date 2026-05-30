OK, let me think this through from scratch. The situation I'm staring at is that hierarchical vision Transformers — Swin in particular — have just taken over as the backbone of choice for classification, detection, and segmentation, and the story everyone tells is that self-attention is simply a better primitive than convolution. That story bugs me, because when I actually line up a Swin against a plain ResNet, attention is only one of the things that differ. The training is different (AdamW, 300 epochs, heavy augmentation, stochastic depth), the macro layout is different (how blocks are distributed across stages, how the image is tokenized, how resolution drops between stages), and a pile of micro choices are different (LayerNorm vs BatchNorm, GELU vs ReLU, how many norms and activations per block). When Swin wins, I can't actually point at attention and say "that's the cause." It's a confound.

So let me not argue about it abstractly. Let me take a standard ResNet-50, and one transplant at a time, move it toward Swin-T — same FLOP budget, ~4.5G, so capacity isn't doing the work — and just watch the ImageNet-1K number after each change. If by the end I've matched Swin without ever adding an attention module, then the gap was never attention; it was the bundle. If I get stuck and only attention closes the last bit, then fine, attention really is special. Either way I learn something the system-level comparison can't tell me. Let me run this as a literal playthrough.

Start with the baseline. The textbook ResNet-50 trained the old way — 90 epochs, light augmentation — gets about 76.1% top-1. But that's not a fair starting line, because the thing I'm comparing against was trained with the modern Transformer recipe. So before touching the architecture at all, I should retrain ResNet-50 with that recipe: AdamW instead of SGD, 300 epochs instead of 90, a 20-epoch linear warmup then cosine decay, batch size 4096, weight decay 0.05, and the whole augmentation/regularization stack — Mixup, Cutmix, RandAugment, Random Erasing, label smoothing, and stochastic depth on the residual branches. I expect this to help a lot, and it does: 76.1 → 78.8. That's +2.7 for free, with the architecture untouched. Already a big chunk of the supposed "attention advantage" is just training. I'm going to freeze this recipe now and keep it identical for every architectural change that follows, so that from here on, any movement in the number is the architecture and nothing else.

Now the architecture. I'll go macro first, then micro, because the macro decisions change the FLOP distribution and I want to nail those before fiddling with layers.

First macro knob: how is compute distributed across the four stages? ResNet-50 uses block counts (3, 4, 6, 3). That allocation was always somewhat empirical — the heavy third stage existed partly so a detector head could sit on a 14×14 plane. Swin-T uses a stage ratio of 1:1:3:1 (and the bigger Swins go 1:1:9:1), so it pours even more of its blocks into stage three. Let me just match that shape: change (3, 4, 6, 3) to (3, 3, 9, 3). That also happens to align FLOPs with Swin-T. Result: 78.8 → 79.4. Small but real; the compute distribution mattered a little.

Second macro knob: the stem — how the raw image gets reduced at the very front. ResNet's stem is a 7×7 stride-2 conv followed by a 3×3 max pool, an overlapping operation that downsamples 4×. ViT and Swin instead "patchify": chop the image into non-overlapping patches with a single conv whose kernel equals its stride. Swin uses patch size 4, which keeps the same 4× initial reduction the four-stage hierarchy wants. So replace the ResNet stem with a 4×4, stride-4 convolution — one clean non-overlapping projection. Result: 79.4 → 79.5. Essentially flat, which is itself informative: the elaborate overlapping stem buys nothing here; a plain patchify stem is just as good and simpler. Keep it.

Now I want to bring in the block-level idea that made ResNeXt better than ResNet. ResNeXt's principle is "use more groups, expand width": replace the dense 3×3 in the bottleneck with a grouped convolution, which slashes that layer's FLOPs, then spend the savings on width. The extreme of grouping is depthwise convolution — number of groups equals number of channels, so each filter touches exactly one channel. Why push it all the way to depthwise? Because of what depthwise convolution *is*: it mixes information purely across space, one channel at a time, and never across channels. And the 1×1 convolutions around it do the opposite — they mix purely across channels at a single spatial location. So a depthwise-plus-1×1 design cleanly separates spatial mixing from channel mixing. That is exactly the structure of a Transformer block: self-attention is a spatial weighted-sum applied per channel (spatial mixing), and the MLP mixes channels (channel mixing), and neither does both at once. If I'm trying to understand what makes the Transformer block tick inside a ConvNet, this separation is the natural thing to import.

So swap the 3×3 for a 7×... no, wait, keep the kernel at 3×3 for now and just make it depthwise; I'll come back to kernel size. Depthwise conv is cheap, so on its own it tanks both FLOPs and accuracy — down to about 78.3 at only 2.4G. That's the capacity loss ResNeXt warned about. Now spend the freed budget on width, the way ResNeXt says: widen the network's base channel count from 64 to 96, which is exactly Swin-T's width. Result: 80.5 at 5.3G. Good — crossing 80 for the first time, and the depthwise/1×1 spatial-channel separation is now in place.

Next: the inverted bottleneck. Look at the Transformer MLP — its hidden dimension is four times its input dimension. It expands wide in the middle and contracts back. A ResNet bottleneck does the opposite: it contracts to a narrow middle (1×1 reduce), does the spatial conv there, then expands. The "expand-in-the-middle, narrow-at-the-ends" shape is precisely MobileNetV2's inverted residual (Sandler et al. 2018), where the block goes narrow → wide (expansion factor, 6 there) → narrow with the skip connecting the thin ends. The Transformer MLP is the same shape with an expansion of 4. Since I'm chasing the Transformer block, I'll use expansion 4 — not MobileNetV2's 6 — because matching the MLP ratio of 4× is the whole point of the analogy. So the block becomes: depthwise conv at width 96, then 1×1 up to 384, then 1×1 back down to 96. There's a nice side effect: inverting the dimensions shrinks the 1×1 in the downsampling residual shortcuts, so total FLOPs actually drop, from 5.3G to 4.6G. And accuracy nudges up: 80.5 → 80.6. (In the bigger ResNet-200 regime the same change helps much more, about +0.8, so this is more than noise.)

Now I want large kernels, because the one thing a 3×3 conv badly lacks compared to attention is reach — even Swin's local windows are at least 7×7, far bigger than 3×3. But there's a prerequisite. Right now in my inverted-bottleneck block the depthwise conv sits where the channels are narrow at the entry, fine, but the natural Transformer ordering is to do the token-mixing op (MSA) *first*, before the wide MLP. And there's an efficiency argument that clicks here: in an inverted bottleneck the middle is wide and the ends are narrow, so I want the expensive, awkward operation — the large-kernel depthwise conv — to act where channels are *few*, and let the cheap, dense 1×1 layers do the heavy lifting in the wide part. So move the depthwise conv up to the top of the block. On its own this reordering costs a bit: FLOPs down to 4.1G, accuracy temporarily down to 79.9. That's a dip I'm accepting because it's the setup for the kernel-size payoff.

With the depthwise conv up front and cheap, I can afford to grow its kernel. Sweep it: 3, 5, 7, 9, 11. The number climbs 79.9 (3×3) → 80.6 (7×7) and then flattens — 9 and 11 give nothing more, and 11 even dips slightly. So the receptive field benefit saturates at 7×7, which, not coincidentally, is right where Swin's window size sits. Larger than that is just wasted FLOPs. Lock in 7×7 depthwise. I've now recovered the dip and I'm back at 80.6, but with a genuinely large kernel and a block whose macro shape mirrors a Transformer block. I verified the saturation in the big regime too — there it saturates even earlier, around 5.

That's the macro and block-shape work done. Everything left is micro — the small per-layer choices.

Activation function. ConvNets default to ReLU; Transformers use GELU, the smooth x·Φ(x) variant. Swap ReLU for GELU. Result: 80.6, unchanged. So the activation *identity* isn't where the gap lives — interesting on its own.

But the *number* of activations is different. Walk a Transformer block: the attention sublayer has its linear projections with no activation between them, and the MLP sublayer has exactly one activation, between its two linear layers. A ResNet/ConvNet block does the opposite — it sticks an activation after essentially every conv, including every 1×1. That's a lot of nonlinearities. Let me match the Transformer's sparsity: delete every GELU in the block except the single one sitting between the two 1×1 layers. Result: 80.6 → 81.3. That's a real jump, +0.7, and it lands me exactly on Swin-T's 81.3. So the block was over-saturated with activations; thinning them out helps.

Same logic for normalization. A Transformer block has fewer norms too — one before attention, one before the MLP. My block currently carries several BatchNorms. Cut them down to a single norm, placed before the first 1×1. Result: 81.3 → 81.4. And I check whether adding one back at the very start of the block helps — it doesn't, so one norm it is. Now I actually have *fewer* norms per block than a Transformer does, and the network is happier for it.

The kind of norm. BatchNorm is the ConvNet default and it does real work — speeds convergence, regularizes — but it drags along all its known headaches: it couples examples through batch statistics, it behaves differently at train vs test time, and it interacts badly with things like weight averaging. LayerNorm, which Transformers use everywhere, is per-sample and has none of that coupling. I know from experience that if you naively swap LayerNorm into a *vanilla* ResNet it gets worse — that's a real failure mode. But my block is no longer a vanilla ResNet; it's been restructured all the way to a Transformer-like block with sparse norms. Try the swap now: BatchNorm → LayerNorm. Result: 81.4 → 81.5. It not only trains fine, it's slightly better. So the old "LN hurts ConvNets" result was about the old block, not an intrinsic law.

Last piece: downsampling between stages. ResNet folds spatial downsampling into the first block of each stage — a stride-2 3×3 conv with a stride-2 1×1 on the shortcut. Swin instead uses a *separate* downsampling layer between stages (its patch merging). Let me do the same: a standalone 2×2 stride-2 conv between stages. And — this diverges. Training blows up. So the separation alone isn't free; ripping downsampling out of the residual block and making it a bare strided conv destabilizes the activation statistics right where resolution and channel count jump. The fix follows from what I already learned about norms doing stabilization work: add a normalization layer wherever the spatial resolution changes. Concretely, a LayerNorm before each downsampling conv, one after the stem, and one after the final global average pooling — the same places Swin puts them. With those norms in, training is stable and the number goes to 82.0, clearly past Swin-T's 81.3.

So I'm done with the playthrough, and I never added an attention module. A pure ConvNet, modernized along exactly the axes that distinguished Swin — recipe, stage ratio, patchify stem, depthwise + width, inverted bottleneck, large kernel moved up front, GELU, sparse activations, sparse norms, LayerNorm, separate downsampling with boundary norms — matches and then beats it at equal FLOPs. That's the answer to the question I started with: the gap was the bundle, not attention. I'll call this network ConvNeXt.

Let me also pin down a couple of details that make the block actually trainable at depth and that I'll want in the code. One is stochastic depth: I'm already using it as a regularizer, and the clean way to set it is to linearly ramp the drop probability from 0 at the first block to some maximum at the last, across all blocks — early layers, which everything downstream depends on, get dropped least. The other is residual-branch scaling. When you stack many residual blocks, it helps to start each block close to the identity so the optimizer doesn't have to fight a large initial perturbation; the per-channel learnable diagonal from CaiT (Touvron et al. 2021) — LayerScale — does exactly this. It's a learnable vector γ, one entry per channel, multiplying the block's residual branch, initialized to a tiny value (1e-6) so each block starts as essentially an identity map and then learns how much of its branch to let through, per channel. I'll add that γ scaling just before the drop-path and the residual add.

One implementation nicety. My block does a depthwise conv (which wants NCHW layout), then a LayerNorm over channels, then two 1×1 convs. But a 1×1 conv is just a linear map over the channel dimension — so if after the depthwise conv I permute to (N, H, W, C), I can do the LayerNorm and the two 1×1s as plain `nn.Linear` layers over the last axis, which is slightly faster in PyTorch, then permute back. I'll need a LayerNorm that can operate in either layout: in channels-last it's just `F.layer_norm`; in channels-first (which I use after the stem and before each downsampling conv, where the tensor is still NCHW) I compute mean and variance over the channel dim by hand and apply the per-channel weight and bias.

Now the block, end to end: input → depthwise 7×7 conv (per-channel spatial mixing, large receptive field) → permute to channels-last → LayerNorm (the single norm) → Linear up to 4×dim (channel expansion, the inverted bottleneck) → GELU (the single activation) → Linear back to dim → multiply by γ (LayerScale) → permute back → drop-path → add the input. And the network: a patchify stem (4×4 stride-4 conv + a channels-first LayerNorm), four stages of these blocks with counts (3, 3, 9, 3) and widths (96, 192, 384, 768) doubling per stage, separate 2×2 stride-2 downsampling convs each preceded by a channels-first LayerNorm, then global average pool → final LayerNorm → linear head. Weights init with a truncated normal (std 0.02); scale the head's weight and bias by a configurable factor.

Here it is.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    # LayerNorm in two layouts. channels_last -> use F.layer_norm directly
    # (after the dw conv, where I permute to N,H,W,C). channels_first -> normalize
    # over the channel dim by hand (after the stem / before each downsample, NCHW).
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # spatial mixing, large kernel, per-channel
        self.norm = LayerNorm(dim, eps=1e-6)                                     # the single norm
        self.pwconv1 = nn.Linear(dim, 4 * dim)                                   # inverted bottleneck: expand 4x (1x1 conv as Linear)
        self.act = nn.GELU()                                                     # the single activation
        self.pwconv2 = nn.Linear(4 * dim, dim)                                   # project back
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None  # LayerScale, start ~identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)        # N,C,H,W -> N,H,W,C so the 1x1s are Linears
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x           # per-channel residual scaling
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)    # residual add, with stochastic depth
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(                                  # patchify stem: 4x4 stride-4, no overlap
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(                  # separate downsampling: norm THEN 2x2 stride-2 conv
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # linear stochastic-depth ramp
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)           # final norm after global pool
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))                     # global average pool then norm

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
```

The causal chain, start to finish: I refused to accept "attention is just better" as an explanation because the comparison was confounded, so I held FLOPs fixed and transplanted Swin's non-attention decisions into a ResNet one at a time — recipe first (+2.7), then stage ratio, patchify stem, depthwise conv with compensating width (the spatial/channel-mixing separation that mirrors attention+MLP), inverted bottleneck at 4× to match the MLP, the depthwise conv moved to the front so a now-affordable large 7×7 kernel gives reach, then the micro thinning (GELU, one activation, one norm, LayerNorm) and finally separate downsampling stabilized by norms at every resolution change — and each step is either neutral or positive, ending above Swin-T at equal compute. The conclusion writes itself: the performance the field credited to self-attention was carried mostly by training and architectural choices that have nothing to do with attention, and a pure ConvNet that adopts them — ConvNeXt — gets there without a single attention layer.
