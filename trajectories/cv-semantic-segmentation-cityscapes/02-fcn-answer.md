**Problem (from step 1).** The encoder–decoder reaches 69.10, but its encoder is a from-scratch
Cityscapes-only feature extractor — it never saw ImageNet. The strongest features in vision come from a
deep residual backbone pretrained on a million images; the bespoke U-shaped encoder leaves that on the
table.

**Key idea — FCN on a dilated pretrained backbone.** Keep the prediction *fully convolutional* (no
fully-connected bottleneck; the spatial grid is never collapsed) but mount it on an ImageNet-pretrained
**ResNet-50 dilated to output stride 8** (R-50-D8: `strides=(1,2,1,1)`, `dilations=(1,1,2,4)`).
Dilation enlarges the receptive field without striding, so the backbone keeps the pretrained weights
yet emits a 1/8-resolution feature map instead of 1/32. The head then *learns* the dense prediction: a
couple of 3×3 conv+norm+ReLU layers refine the deep feature locally, an optional skip concatenates the
head's input with its refined output, a 1×1 conv maps to 19 classes, and a bilinear upsample by 8
returns full resolution — with the loss computed through the upsample.

**Why it works.** Two levers move at once: pretrained ResNet-50 features give far stronger semantics
than a from-scratch encoder (the *what*), and dilating the last stages to output stride 8 keeps a
moderately fine grid without adding parameters (the *where*). The learned 3×3 convs do local
refinement a fixed 1×1 projection cannot; the input skip lets the classifier see refined and raw deep
features together. Strong pretrained features at output stride 8 beat the lightweight encoder–decoder.

**Change / code.** The fully-convolutional refine-and-classify head over the dilated backbone.

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

Backbone: R-50-D8, ImageNet-pretrained; head channels 512, two 3×3 convs, `concat_input=True`. The
final scores are bilinearly upsampled by 8 and trained with per-pixel cross-entropy.
