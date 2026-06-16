# SqueezeNet

## The problem

Almost all CNN research optimizes for one number — ImageNet accuracy — and lets parameter count be whatever falls out. But for any target accuracy there are many architectures that reach it, and they differ wildly in size. Fewer parameters at equal accuracy buys three concrete things: (1) faster distributed data-parallel training, since gradient communication per step scales with the number of parameters; (2) cheaper over-the-air model updates to deployed devices (a 240MB AlexNet is a heavy transfer to a car); (3) feasible deployment on FPGAs and ASICs, whose on-chip memory is often under 10MB with no off-chip storage, so a small enough model can sit entirely on-chip and stream video through it with no external-memory traffic.

The goal: an architecture that matches AlexNet-level ImageNet accuracy (57.2% top-1 / 80.3% top-5) with ~50x fewer parameters, designed from scratch — and small enough that, composed with compression, it drops under 0.5MB.

## Key idea: attack all three factors of the conv parameter count

A convolution layer of 3x3 filters has parameter count `(input channels) × (number of filters) × (3 × 3)`. Three multiplicative factors, attacked by three design strategies:

- **Strategy 1 — replace 3x3 filters with 1x1 filters.** A 1x1 filter has 9x fewer parameters than a 3x3 filter. Given a filter budget, make the majority 1x1. The only thing a 1x1 filter cannot do is capture spatial structure, so keep just enough 3x3 filters to see space.
- **Strategy 2 — decrease the number of input channels to the 3x3 filters.** Even with few 3x3 filters, each still costs `9 × (input channels)`. Put a cheap 1x1 *squeeze* layer in front that thins the channel count before the 3x3 filters see it, slashing the dominant `input channels` factor.
- **Strategy 3 — downsample late so conv layers have large activation maps.** The spatial size of a layer's output is set by where the stride>1 steps sit. Concentrate downsampling toward the end so most layers keep large maps; large activation maps tend to raise classification accuracy at equal parameters. This buys accuracy back for free — it touches zero parameters. (Strategies 1 and 2 cut parameters while preserving accuracy; Strategy 3 maximizes accuracy on a fixed budget.)

## The Fire module

Strategies 1 and 2 fold into one reusable building block, the **Fire module**: a *squeeze* convolution layer (only 1x1 filters) feeding an *expand* layer that mixes 1x1 and 3x3 filters. Three tunable dimensions:

- `s_1x1` — number of filters in the squeeze layer (all 1x1),
- `e_1x1` — number of 1x1 filters in the expand layer,
- `e_3x3` — number of 3x3 filters in the expand layer.

The squeeze first reduces the incoming channels to `s_1x1`; the expand then runs both a 1x1 bank (`e_1x1` filters) and a 3x3 bank (`e_3x3` filters) on those `s_1x1` channels and **concatenates** their outputs along the channel axis, giving `e_1x1 + e_3x3` output channels. The constraint that makes this save parameters: set `s_1x1 < (e_1x1 + e_3x3)`, so the squeeze genuinely limits the number of input channels seen by the (expensive) 3x3 filters — Strategy 2. The liberal use of 1x1 filters everywhere is Strategy 1.

Two implementation details. A conv framework does not natively give one layer two kernel sizes, so the expand is implemented as two separate conv layers (one 1x1, one 3x3) whose outputs are concatenated on the channel dimension — numerically identical to a single mixed-kernel layer. The 3x3 expand bank gets 1 pixel of zero-padding so its output height and width match the 1x1 bank's. ReLU is applied to the activations of the squeeze and of each expand bank.

Worked example (fire2): input 96 channels, `s_1x1 = 16`, `e_1x1 = e_3x3 = 64`. Squeeze 96→16 at 1x1 = 1,536 params; expand-1x1 16→64 = 1,024; expand-3x3 16→64 at 3x3 = 9,216; total ≈ 11.8k, output 128 channels. Without the squeeze, feeding 96 channels straight into 64 3x3 filters costs `96 × 64 × 9 = 55,296` for the 3x3 bank alone — ~5x more, and the gap widens in later modules where the input is 256 or 512 channels.

## The macroarchitecture

SqueezeNet begins with a standalone convolution layer (conv1), followed by 8 Fire modules (fire2-fire9), ending with a final convolution layer (conv10) that maps to the 1000 class channels. The number of filters per Fire module increases gradually from the beginning to the end of the network. Stride-2 max-pooling in the feature extractor is placed after conv1, fire4, and fire8; the final conv10 activations are then reduced by global average pooling. These late reductions are Strategy 3, so most Fire modules operate on large activation maps. There are **no fully-connected layers**: following the NiN lesson, the head is a 1x1 conv to `num_classes` followed by global average pooling. Dropout with ratio 50% is applied after fire9. Trained with SGD in Caffe, starting at learning rate 0.04 with linear decay.

The 24 per-module dimensions are generated from five metaparameters: `base_e` (expand width of the first module), `incr_e` (expand-width increment), `freq` (how often the width steps up), `pct_3x3` (fraction of expand filters that are 3x3), and `SR` (squeeze ratio). For Fire module `i`: `e_i = base_e + incr_e × floor(i / freq)`; `e_{i,3x3} = e_i × pct_3x3`, `e_{i,1x1} = e_i × (1 − pct_3x3)`; and `s_{i,1x1} = SR × e_i`. SqueezeNet uses `base_e = 128`, `incr_e = 128`, `freq = 2`, `pct_3x3 = 0.5`, `SR = 0.125` (the name comes from this low squeeze ratio: squeeze layers have 0.125x the filters of the accompanying expand layers, i.e. 8x fewer channels). This yields the per-module dimensions: fire2/fire3 = (16, 64, 64), fire4/fire5 = (32, 128, 128), fire6/fire7 = (48, 192, 192), fire8/fire9 = (64, 256, 256).

Full channel schedule:

| block | dimensions |
| --- | --- |
| conv1 | 3 -> 96, 7x7, stride 2 |
| fire2, fire3 | in 96/128, `s_1x1=16`, `e_1x1=64`, `e_3x3=64`, out 128 |
| fire4, fire5 | in 128/256, `s_1x1=32`, `e_1x1=128`, `e_3x3=128`, out 256 |
| fire6, fire7 | in 256/384, `s_1x1=48`, `e_1x1=192`, `e_3x3=192`, out 384 |
| fire8, fire9 | in 384/512, `s_1x1=64`, `e_1x1=256`, `e_3x3=256`, out 512 |
| conv10 | 512 -> 1000, 1x1 |

The whole network lands at ~1.2M parameters, ~50x fewer than AlexNet's ~60M, at AlexNet-level top-5 accuracy, in 4.8MB at 32-bit.

## Composing with compression

The architecture is dense and small, but still compressible — the two approaches stack. Applying Deep Compression (pruning + codebook quantization + Huffman coding) to SqueezeNet: 8-bit quantization with 33% sparsity → 0.66MB (363x smaller than 32-bit AlexNet) at equivalent accuracy; 6-bit quantization with 33% sparsity → 0.47MB (510x smaller than AlexNet), still at equivalent accuracy. So a small architecture designed from scratch is itself amenable to compression, reaching under 0.5MB.

## Optional macroarchitecture variant: bypass connections

Because the squeeze layers are a deliberate 8x bottleneck, a bypass that routes information *around* a Fire module can help. A **simple bypass** is just a wire — an identity shortcut that elementwise-adds the input to the output, requiring the module to learn a residual; it adds zero parameters but needs input and output channel counts to match, which holds for only half the modules (those where the expand width did not just step up — bypasses go around fire3, fire5, fire7, fire9). A **complex bypass** puts a 1x1 conv on the shortcut to fix mismatched channel counts, adding parameters. The vanilla, bypass-free network stays the default for minimal size; simple bypass is a near-free accuracy upgrade.

## Runnable PyTorch

```python
import torch
import torch.nn as nn


class FeatureModule(nn.Module):
    """Squeeze then expand. The squeeze 1x1 thins the channels so the 3x3
    expand bank is cheap (Strategy 2); the expand is a mix of 1x1 and 3x3
    filters (Strategy 1). Output = channel-wise concat of the two banks."""
    def __init__(self, in_channels, squeeze, expand1x1, expand3x3):
        super().__init__()
        # Squeeze: 1x1 conv reducing channels to `squeeze` (< expand1x1 + expand3x3).
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        # Expand bank A: 1x1 filters (9x cheaper than 3x3).
        self.expand1x1 = nn.Conv2d(squeeze, expand1x1, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        # Expand bank B: 3x3 filters, padded 1px so output H,W match the 1x1 bank.
        self.expand3x3 = nn.Conv2d(squeeze, expand3x3, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        a = self.expand1x1_act(self.expand1x1(x))
        b = self.expand3x3_act(self.expand3x3(x))
        return torch.cat([a, b], dim=1)   # concatenate on the channel axis


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: standalone conv (3-channel input, nothing to squeeze yet),
        # 7x7 stride 2. Pooling is sparse and pushed late (Strategy 3): after
        # conv1, fire4, fire8 -> most Fire modules see large activation maps.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.features = nn.Sequential(
            FeatureModule(96, 16, 64, 64),      # fire2  -> 128 out
            FeatureModule(128, 16, 64, 64),     # fire3  -> 128
            FeatureModule(128, 32, 128, 128),   # fire4  -> 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(256, 32, 128, 128),   # fire5  -> 256
            FeatureModule(256, 48, 192, 192),   # fire6  -> 384
            FeatureModule(384, 48, 192, 192),   # fire7  -> 384
            FeatureModule(384, 64, 256, 256),   # fire8  -> 512
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(512, 64, 256, 256),   # fire9  -> 512
        )
        # No fully-connected layers (the NiN lesson): a 1x1 conv classifier
        # (conv10) plus global average pooling replaces AlexNet's huge FC head.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                              # dropout after fire9
            nn.Conv2d(512, num_classes, kernel_size=1),     # conv10: 1x1 -> classes
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                   # global average pool
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)                          # logits: (N, num_classes)


if __name__ == "__main__":
    net = Net(num_classes=1000)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")          # ~1.2M, ~50x fewer than AlexNet
    y = net(torch.randn(1, 3, 224, 224))
    print(y.shape)                              # torch.Size([1, 1000])
```

## Verification

- **Conv parameter formula and the three strategies** — `(input channels) × (filters) × (3×3)`; Strategy 1 (replace 3x3 with 1x1, 9x cheaper), Strategy 2 (squeeze layers cut input channels to 3x3 filters), Strategy 3 (downsample late for large activation maps, He & Sun's constrained-time-cost result): grounded in src `SqueezeNet_ICLR.tex` §"Architectural Design Strategies" (Strategies 1–3) and §Introduction motivations.
- **Fire module** — squeeze layer is 1x1-only of size `s_1x1`; expand mixes `e_1x1` 1x1 and `e_3x3` 3x3 filters; constraint `s_1x1 < e_1x1 + e_3x3`; expand implemented as two convs concatenated on the channel dim; 1px zero-pad on the 3x3 bank; ReLU after squeeze and expand: §"The Fire Module" and §"Other SqueezeNet details".
- **Macroarchitecture dims** — conv1 → fire2-fire9 → conv10; source text names stride-2 max-pooling after conv1/fire4/fire8/conv10, while the runnable scaffold represents the terminal post-conv10 reduction as the NiN-style global average pool; 50% dropout after fire9; no FC layers; metaparameters `base_e=128, incr_e=128, freq=2, pct_3x3=0.5, SR=0.125`; the per-module `(s, e1, e3)` triples (16,64,64)/(32,128,128)/(48,192,192)/(64,256,256) follow from `e_i = base_e + incr_e·floor(i/freq)`, `s = SR·e_i`: §"The SqueezeNet architecture" and §"CNN Microarchitecture metaparameters". Channel-in dims for each Fire in the code equal the previous module's `e_1x1 + e_3x3` output (96→128→128→256→256→384→384→512→512), consistent with the squeeze-feeds-both-banks design.
- **Accuracy / size / compression** — AlexNet 57.2% top-1 / 80.3% top-5 at 240MB; SqueezeNet 4.8MB at 32-bit (~50x), 0.66MB at 8-bit + 33% sparsity (363x), 0.47MB at 6-bit + 33% sparsity (510x) via Deep Compression, all at AlexNet-level accuracy: §Evaluation, Table T:model-compression.
- **Bypass variant** — simple bypass (identity, zero params) around fire3/fire5/fire7/fire9 needs matched channel counts; complex bypass adds a 1x1 conv for mismatched counts; motivation is relieving the 8x squeeze bottleneck (SR=0.125): §"CNN Macroarchitecture Design Space Exploration".
- **Code** — `FeatureModule` fills the context scaffold with the Fire module structure (squeeze→ReLU→{expand1x1,expand3x3}→ReLU→concat), and `Net` fills the scaffolded `stem`, Fire-module stack, and 1x1-conv + adaptive-avg-pool head. The `__main__` block runs a forward pass: output shape `(1, 1000)`.
