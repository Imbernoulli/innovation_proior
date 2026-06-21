The encoder–decoder with per-level skips reaches **69.10 mIoU** on Cityscapes val: the prediction is dense and the boundaries are no longer mush. But look at what that encoder *is* — a stack of plain two-conv blocks trained from scratch on a few thousand Cityscapes images. It is a reasonable feature extractor, but it has never seen ImageNet, and the rest of vision has converged on one fact: a deep residual network pretrained on a million labeled images learns features so good that almost every downstream task does better starting from them than from a bespoke encoder. My U-shaped net leaves that on the table; its semantics come entirely from a small, narrow training set, while a pretrained ResNet-50 brings a far richer notion of *what objects look like* before it sees a single street scene.

I propose **FCN on a dilated, ImageNet-pretrained backbone**: keep the part of the last rung that worked — a fully-convolutional net that produces a dense per-pixel prediction, with no fully-connected bottleneck collapsing the spatial grid — but mount it on a *pretrained* deep backbone instead of a from-scratch encoder. There are two distinct levers, and they should be kept separate: *getting the resolution up inside the backbone*, before the head ever runs, and *what the head does* with the resulting feature map. For the first, I do not have to accept the backbone's native output stride of $32$. The downsampling in the last stages comes from stride-$2$ convolutions; if I replace those strides with *dilation* — spreading the kernel taps apart by the same factor instead of striding — each unit still sees the same wide receptive field, the pretrained weights still apply (the kernel is the same size, only the spacing of where it samples has changed), but the feature map stops shrinking. Dilating the last two stages takes the backbone to output stride $8$: a $4\times$ finer grid for free, in the sense that I keep the pretrained weights and add no parameters, paying only in compute because the convs now run on a larger spatial map. This is the R-50-D8 backbone — ResNet-50 with `strides=(1,2,1,1)` and `dilations=(1,1,2,4)`, output stride $8$.

The head then *learns* the dense prediction rather than reading it off a fixed projection. At output stride $8$ the feature map is rich — $2048$ channels of pretrained semantics — and reasonably fine. I run a few $3\times3$ conv+norm+ReLU layers over it, because a couple of $3\times3$ convs can do local smoothing and refinement — cleaning up the per-pixel decision using the neighborhood — that a pointwise $1\times1$ projection cannot, then the $1\times1$ classifier, then a bilinear upsample by $8$ to full resolution. The whole thing is convolutional end to end: every layer is a convolution, the spatial grid is never collapsed, so the net accepts any input size, emits a correspondingly-sized dense map, and the "upsampling" is an in-network resize the loss is computed through — so the net learns dense prediction directly against the dense labels. One refinement carries over the U-Net lesson in miniature: reading off only the very deepest map throws away a less-processed, more-faithful copy of the features, so I add a single "concat the input" skip — run the $3\times3$ convs, then concatenate their output with the head's *input* feature before the final classifier, letting the classifier see refined and raw deep features side by side. I do not need the full U-Net decoder machinery here, because the dilated backbone has already kept the resolution I need; one skip suffices. The bet against $69.10$ is that two things move at once — the backbone is now a pretrained ResNet-50, so the *what* is far stronger, and dilation keeps it at output stride $8$, so the *where* is decent without any decoder — and that strong pretrained semantics at a moderately-fine resolution win on the class-averaged metric. The acknowledged cost is the flip side of dropping the U-Net's full decoder: at output stride $8$ the prediction is still $8\times$ coarser than the labels, so the finest boundaries are softer than a full-resolution decoder produced — I trade some of the U-Net's localization for much stronger semantics and bet the semantics win.

```python
class FCNHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True,
                 dilation=1, **kwargs):
        super().__init__(**kwargs)
        conv_padding = (kernel_size // 2) * dilation
        convs = [ConvModule(self.in_channels, self.channels, kernel_size,
                            padding=conv_padding, dilation=dilation,
                            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for _ in range(num_convs - 1):
            convs.append(ConvModule(self.channels, self.channels, kernel_size,
                                    padding=conv_padding, dilation=dilation,
                                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)
        if concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels, kernel_size,
                                       padding=kernel_size // 2,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)             # deep OS=8 feature, 2048 ch
        feats = self.convs(x)                          # learned local refinement
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))  # skip
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)                    # 1x1 -> classes; upsample x8
```

Backbone: R-50-D8, ImageNet-pretrained; head channels $512$, two $3\times3$ convs, `concat_input=True`. The final scores are bilinearly upsampled by $8$ and trained with per-pixel cross-entropy.
