Everyone is optimizing for one number — ImageNet accuracy — and treating model size as whatever falls out. But for any given accuracy there are lots of architectures that hit it, and they're wildly different in parameter count. That's the slack I want to exploit. Why do I care about parameters specifically, not FLOPs? Three reasons that all point the same way. Distributed training is communication-bound, and the gradient communication per step scales with the number of parameters, so fewer parameters means faster training. Shipping a model to a deployed device — over the air, to a car — means moving the parameters across a network; a 240MB AlexNet is a painful update. And FPGAs often have under 10MB of on-chip SRAM and no off-chip memory at all; if the whole model fits on-chip, you can stream video through it with zero external-memory traffic, which is the difference between feasible and not. So the target is sharp: AlexNet-level accuracy on ImageNet with an order of magnitude or two fewer parameters, and small enough that compression can push it under a megabyte.

There's an obvious competing approach I should name and then set aside: take a big trained model and compress it — SVD on the weight matrices, prune the small weights to a sparse matrix and retrain, quantize to a few bits with a codebook. Those work, 9× to 35× on AlexNet. But they all *start* from the heavy architecture. I want to attack the problem from the other end: design a small architecture from scratch. And these aren't rivals — a small dense model can still be compressed afterward, so the two stack.

So where do parameters actually live? Write down the parameter count of one convolution layer of 3×3 filters: it's `(input channels) × (number of filters) × 3 × 3`. Three multiplicative factors. The `3×3 = 9` is the spatial footprint of each filter. The number of filters is the output width. The input channels is how deep the incoming feature map is. If I want fewer parameters and I'm staring at a product, the natural move is to push on each factor separately.

Factor one, the `9`. A 1×1 filter has exactly 9× fewer parameters than a 3×3 filter, and it still does something useful — it mixes channels at each pixel, the Network-in-Network idea. So one lever is: given a budget of filters, make as many of them 1×1 as I can get away with. The only thing a 1×1 filter *can't* do is capture spatial structure — it has no receptive field beyond a single pixel. So I can't make *everything* 1×1, or the network goes blind to spatial patterns. I need *some* 3×3 filters to see space. That turns "how many 3×3 filters?" into a knob to sweep, not a thing to assume — I want the fewest 3×3 filters that still preserve accuracy.

Factor three, the input channels feeding the 3×3 filters. The parameters of a 3×3 layer are `(input channels) × (3×3 filters) × 9`. Even after I've minimized the *number* of 3×3 filters, each one still costs `9 × (input channels)`. So if a 3×3 filter is fed a fat input — say 512 channels — it's expensive no matter what. The lever here is to *thin the input* right before the 3×3 filters see it. How do I cheaply reduce channel count? A 1×1 convolution: take whatever channels are coming in and squeeze them down to a smaller number with a 1×1 layer, then let the 3×3 filters operate on that thinner feature map. The 1×1 squeeze is itself cheap (1×1 filters), and it shrinks the dominant `input channels` factor of the expensive 3×3 layer.

Those two levers are both about *cutting* parameters, and cutting parameters will cost accuracy somewhere. I want a third move that *buys accuracy back* without spending parameters. Here's where the activation-map size comes in. The spatial size of each layer's output is set by where I put the stride>1 downsampling steps. If I downsample early, almost every layer works on a small map; if I push the downsampling late, most layers work on large maps. Large activation maps mean each filter is applied at more spatial positions, which — at equal parameters — empirically gives higher accuracy. So the third move is to *delay downsampling*: keep stride 1 through most of the network and concentrate the stride-2 steps toward the end. This raises accuracy and touches zero parameters.

Now I need to fold the first two levers into a single reusable module so the whole network is describable by a few numbers, the way Inception modules let you describe GoogLeNet. Let me design that module directly from the two levers and see what falls out.

The channel-thinning lever says: put a 1×1 layer that *squeezes* the channel count down first. So the module starts with a 1×1 convolution layer — a squeeze layer — that takes the incoming channels and reduces them to some small number `s`. Then the 1×1-preference lever says: do the real filtering with mostly 1×1 filters and only some 3×3. So after the squeeze, I want an *expand* layer that produces the module's output by combining two banks of filters operating on those `s` squeezed channels: a bank of `e1` filters that are 1×1, and a bank of `e3` filters that are 3×3. The expand output is the concatenation of these two banks along the channel dimension, giving `e1 + e3` output channels. The 1×1 squeeze feeding both banks is what keeps the 3×3 bank cheap — the `e3` 3×3 filters each cost `9 × s`, and `s` is small. Three hyperparameters per module: `s`, `e1`, `e3`. It squeezes then expands and fires off both filter banks, so I'll call it a Fire module.

There's one constraint that decides whether this actually saves anything. The squeeze has to *squeeze*: I need `s` to be less than `e1 + e3`, so that the 3×3 filters in the expand layer are fed fewer channels than the module outputs. If `s` were as large as the output, the squeeze wouldn't be limiting the 3×3 input at all and the channel-thinning lever would be doing nothing. Let me parameterize that with a squeeze ratio `SR`: set `s = SR × (e1 + e3)` with `SR < 1`. How small can `SR` go? That's exactly the kind of thing to sweep rather than guess — I'd train a family of networks varying `SR` from, say, 0.125 up to 1.0, each from scratch, and watch where accuracy plateaus. My expectation: very small `SR` (aggressive squeeze) costs accuracy, and there's a knee somewhere. The aggressive end is `SR = 0.125`; if it preserves the accuracy target, it is the most parameter-frugal point, and if it doesn't, the sweep tells me to relax the bottleneck.

Similarly, the 1×1-vs-3×3 split inside the expand layer: how important is spatial resolution in the filters? Define `pct_3x3` as the fraction of expand filters that are 3×3, and sweep it. I'd expect accuracy to rise as I add 3×3 capacity and then plateau — there's some point past which more 3×3 filters just add parameters without accuracy. As a balanced default to be confirmed by the sweep I'll set `pct_3x3 = 0.5`, half 1×1 and half 3×3.

To make a whole network describable by a few meta-numbers, let the per-module dimensions grow on a schedule rather than be set 24 numbers by hand. Let `base_e` be the expand width of the first module, and bump the expand width by `incr_e` every `freq` modules: module `i` gets `e_i = base_e + incr_e × floor(i / freq)`. Then `e1_i = e_i × (1 − pct_3x3)`, `e3_i = e_i × pct_3x3`, and `s_i = SR × e_i`. So the entire architecture collapses to: `base_e`, `incr_e`, `freq`, `pct_3x3`, `SR`. I'll use `base_e = 128`, `incr_e = 128`, `freq = 2` — width steps up every two modules — `pct_3x3 = 0.5`, `SR = 0.125`.

Now lay out the macro-architecture, applying the late-downsampling move. Start with one ordinary convolution layer: the input has only 3 channels — nothing to squeeze yet — so a standalone conv stem makes sense, with a largish 7×7 receptive field and stride 2. Then a stack of eight Fire modules, fire2 through fire9, with the expand widths stepping up per the schedule (128, 128, 256, 256, 384, 384, 512, 512 worth of expand filters across the eight). End with a final convolution layer, conv10, that maps to the 1000 class channels. And no fully-connected layers anywhere — the NiN lesson is that the FC classifier is where AlexNet/VGG hide most of their parameters, so I replace it with a 1×1 conv to `num_classes` followed by global average pooling. That single choice removes the biggest parameter sink in the field.

Downsampling placement: keep stride 1 through the Fire modules and put max-pools (stride 2) only after conv1, after fire4, after fire8, and a global pool after conv10. A 50% dropout after fire9, right before conv10, for regularization. I should check that this placement actually does what I want — keep most modules on large maps — rather than just assert it. Walk the spatial size from a 224×224 input: conv1 is 7×7 stride 2 with no padding, `(224−7)/2 + 1 = 109`; pool1 is 3×3 stride 2 with ceil rounding, `ceil((109−3)/2)+1 = 54`; fire2, fire3, fire4 all run at 54×54 (stride 1); pool4 drops it to `ceil((54−3)/2)+1 = 27`; fire5, fire6, fire7, fire8 run at 27×27; pool8 drops to `ceil((27−3)/2)+1 = 13`; fire9 runs at 13×13. So three Fire modules see 54×54 and four see 27×27 — seven of the eight operate on maps of 27×27 or larger, and only fire9 is at 13×13. That is the late-downsampling design working as intended; had I instead pooled after every couple of layers from the start, most modules would be stuck on tiny maps. Good.

A small implementation wrinkle: the expand layer wants both 1×1 and 3×3 filters producing output at the same spatial size, but a conv framework doesn't natively give one layer two kernel sizes. So I implement the expand as two separate conv layers — one with `e1` 1×1 filters, one with `e3` 3×3 filters (padded by 1 pixel so its output is the same height and width as the 1×1 bank's) — and concatenate their outputs along the channel axis. Is that really equivalent to a single mixed-kernel layer? Each output channel of a conv depends only on its own filter and the shared input; the 1×1 channels never read the 3×3 filters' weights and vice versa, so stacking the two banks' outputs into one `e1 + e3`-channel tensor is exactly the set of feature maps a single layer with `e1` 1×1 and `e3` 3×3 filters would produce. The padding is what makes the two banks share spatial dimensions so the concatenation is even well-defined. So yes, it's the same thing, just expressed as two ops. ReLU after the squeeze, and ReLU after each expand bank.

Before I trust the design I want to see the parameter arithmetic on a concrete module, fire2: input 96 channels (from conv1), `e = 128` so `e1 = e3 = 64`, and `s = SR × e = 0.125 × 128 = 16`. Squeeze: a 1×1 layer 96→16, costing `96 × 16 = 1536` parameters. Expand-1×1: 16→64 at 1×1, `16 × 64 = 1024`. Expand-3×3: 16→64 at 3×3, `16 × 64 × 9 = 9216`. Module total ≈ 11.8k parameters (weights only), output 128 channels. Now imagine *not* squeezing — feeding 96 channels straight into 64 3×3 filters: `96 × 64 × 9 = 55,296` just for the 3×3 bank, ~5× the whole squeezed module. The mechanism is clear in the ratio: the 3×3 bank's cost is proportional to its input channel count, and the squeeze cuts that from 96 to 16, a factor of 6. Does that factor grow in later modules? Take fire8: input 384, `s = 64`, so the 3×3 bank costs `64 × 256 × 9 = 147,456` squeezed versus `384 × 256 × 9 = 884,736` unsqueezed — again a factor of `384/64 = 6`. The *ratio* stays ~6× because `s` scales up with the schedule, but the *absolute* saving balloons: ~46k parameters saved in fire2 against ~737k saved in fire8. So the squeeze matters most exactly where the network is widest, which is where it would otherwise hurt.

Now the whole-network count, which I should compute rather than eyeball. Weights only: the eight Fire modules, summing `(in·s) + (s·e1) + (s·e3·9)` per module over the schedule (16,64,64)/(32,128,128)/(48,192,192)/(64,256,256), come to 11,776 + 12,288 + 45,056 + 49,152 + 104,448 + 110,592 + 188,416 + 196,608 = 718,336. conv1 is `3 × 96 × 49 = 14,112`. conv10 is `512 × 1000 = 512,000`. That's 1,244,448 weight parameters; adding the biases on every conv brings the total to about 1.25M. Against AlexNet's ~60M that is the ~50× reduction I was after. So the design hits the size target on paper; whether it preserves AlexNet-level accuracy is the one thing this arithmetic can't tell me, and that's what the ImageNet run has to settle. (The aggressive `SR = 0.125` squeeze is severe enough that I'd name the architecture after it: SqueezeNet.)

One more axis, at the macro level: bypass connections. ResNet showed that adding an earlier activation to a later one helps deep nets train and adds essentially no parameters. There's a reason it might matter *specifically here*: my squeeze layers are a deliberate bottleneck — with `SR = 0.125`, each squeeze has 8× fewer channels than its expand, so only a thin slice of information passes through. A bypass around a Fire module gives information a path *around* that bottleneck. The catch is shape-matching: a plain additive bypass needs the input and output channel counts equal, and that's only true for some of my modules. Which ones? On the doubling schedule, a module's input equals its output exactly when the expand width didn't just step up — so fire3, fire5, fire7, fire9 (input 128→out 128, 256→256, 384→384, 512→512), the even modules in each width pair. The other four change width and can't take a plain identity add. So I can wire simple identity bypasses around the four shape-matching modules, requiring those modules to learn a residual. Where the channel counts differ, a "complex" bypass would need a 1×1 conv on the shortcut to fix the channel count, which adds parameters. I'd test both. My expectation is that the simple identity bypasses help by relieving the squeeze bottleneck without adding parameters, and that the complex bypass helps less per parameter; so the default architecture stays bypass-free for minimal size, with simple bypass as a candidate accuracy upgrade. I'll keep these as variants rather than baking them into the core, since the core's job is to be as small as possible.

Let me write it, mirroring how it goes together in a conv framework.

```python
import torch
import torch.nn as nn


class FeatureModule(nn.Module):
    """Squeeze then expand. The squeeze 1x1 thins the channels so the 3x3
    expand bank is cheap (strategy 2); the expand is mostly 1x1 filters
    with some 3x3 (strategy 1). Output = concat of the two expand banks."""
    def __init__(self, in_channels, squeeze, expand1x1, expand3x3):
        super().__init__()
        # Squeeze: 1x1 conv reducing channels to `squeeze` (< expand1x1 + expand3x3).
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        # Expand bank A: 1x1 filters (9x cheaper than 3x3).
        self.expand1x1 = nn.Conv2d(squeeze, expand1x1, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        # Expand bank B: 3x3 filters, padded so output H,W match the 1x1 bank.
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
        # Stem: a standalone conv (3-channel input, nothing to squeeze yet),
        # 7x7 stride 2. Pooling is sparse and pushed late (strategy 3): after
        # conv1, fire4, fire8 only -> most Fire modules see large maps.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.features = nn.Sequential(
            FeatureModule(96, 16, 64, 64),     # fire2  -> 128 out
            FeatureModule(128, 16, 64, 64),    # fire3  -> 128
            FeatureModule(128, 32, 128, 128),  # fire4  -> 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(256, 32, 128, 128),  # fire5  -> 256
            FeatureModule(256, 48, 192, 192),  # fire6  -> 384
            FeatureModule(384, 48, 192, 192),  # fire7  -> 384
            FeatureModule(384, 64, 256, 256),  # fire8  -> 512
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(512, 64, 256, 256),  # fire9  -> 512
        )
        # No fully-connected layers (the NiN lesson): a 1x1 conv classifier
        # plus global average pooling replaces AlexNet's huge FC head.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
```

The causal chain, end to end: parameters in a conv layer are `input_channels × num_filters × kernel_area`, so I attack all three factors — swap 3×3 filters for 1×1 (9× cheaper, keep just enough 3×3 to see spatial structure), and put a cheap 1×1 *squeeze* layer in front of the 3×3 filters to thin their input channels — and bundle both into a reusable Fire module (squeeze 1×1, then a concatenated expand bank of 1×1 and 3×3 filters), with the squeeze ratio kept aggressively low (0.125). I buy accuracy back for free by delaying downsampling so most modules keep large activation maps, and I delete the fully-connected classifier entirely (the field's biggest parameter sink) in favor of a 1×1-conv + global-average-pool head. The result is a stack of eight Fire modules between two standalone convs, which the parameter arithmetic puts at ~1.25M weights — ~50× fewer than AlexNet — and which I expect to preserve AlexNet-level accuracy, with optional simple identity bypasses around the four shape-matching modules to relieve the squeeze bottleneck at no parameter cost. The accuracy claim is the one piece the design alone can't verify; the ImageNet run decides it.
