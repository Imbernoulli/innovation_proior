# SENet synthesis

## Pain point
A conv layer fuses spatial+channel info within a LOCAL receptive field. Output u_c = sum_s v_c^s * x^s sums over input channels with fixed learned weights — channel interdependencies are entangled with local spatial filtering, implicit and local. Each spatial unit only sees its receptive field; no per-instance global view of which channels matter. Spatial side has been heavily explored (multi-scale Inception, spatial attention STN); channel side underexplored. Goal: cheaply let the network use GLOBAL information to recalibrate (emphasize/suppress) channel responses per input.

## Ancestors / lineage
- Standard conv (eqn 1): u_c = v_c * X = sum_s v_c^s * x^s. Channel mixing fixed, local, instance-agnostic.
- VGG/Inception: depth + multi-scale → stronger representations; Inception models cross-scale spatial.
- BN (Ioffe & Szegedy 2015): stabilizes deep training, smoother loss surface.
- ResNet (He 2016): identity skip connections enable very deep nets; preact variant.
- Highway nets (Srivastava 2015): GATING on shortcut — learned data-dependent gates to regulate info flow. (Gating precedent.)
- Grouped/multibranch conv (ResNeXt, Inception): cardinality; 1x1 conv (NiN) remaps channels but still local/instance-agnostic.
- Attention (Itti, Mnih, Vaswani): bias compute toward informative signal components. Residual Attention Net (Wang 2017): heavy trunk-and-mask hourglass attention between stages — powerful but expensive. CBAM (concurrent). 
- Feature engineering (Fisher vectors, SPM): global pooled statistics summarize whole image.

## The three operations (derive)
SQUEEZE F_sq: global average pooling over HxW. z_c = (1/(H*W)) sum_{i=1..H} sum_{j=1..W} u_c(i,j). z in R^C. Gives each channel a scalar with GLOBAL receptive field (whole image), forming a channel descriptor. Cheap, parameter-free. (Ablation: GAP slightly beats global max pool; NoSqueeze variant — replace pool+FC with 1x1 convs keeping spatial dims so no global embedding — hurts, confirming global info matters.)

EXCITATION F_ex: must (1) be flexible/nonlinear, (2) non-mutually-exclusive (multiple channels can be on — NOT one-hot/softmax). Self-gating MLP: s = sigma(W2 delta(W1 z)), delta=ReLU, sigma=sigmoid. Bottleneck: W1 in R^{(C/r)xC} reduces C->C/r, W2 in R^{Cx(C/r)} expands back. Reduction ratio r (default 16). 
- Why bottleneck: limit params/complexity & aid generalization vs a full CxC matrix. Params per block (two FC, no bias) = C*(C/r) + (C/r)*C = 2C^2/r. Total over net = (2/r) sum_s N_s C_s^2.
- Why sigmoid not softmax: softmax forces competition (sum to 1, ~one-hot), excludes co-activation; sigmoid gives independent gates in (0,1). Ablation: ReLU excitation much worse (drops below baseline), tanh slightly worse than sigmoid.
- r tradeoff: smaller r -> more params, not monotonically better; r=16 good balance.

SCALE F_scale: x_tilde_c = s_c * u_c, channelwise multiply scalar gate by whole feature map. Output drops into next layer.

## Integration
- VGG: insert after nonlinearity following each conv.
- Inception: F_tr = whole Inception module.
- ResNet: F_tr = the non-identity (residual) branch; squeeze+excitation act on the residual branch output BEFORE summation with identity. (SE-PRE / SE-Identity also fine; SE-POST after the add is worse -> apply before branch aggregation.)
- SE_3x3 variant: put SE after the 3x3 (fewer channels) -> fewer params, comparable acc.

## Cost
SE-ResNet-50: +~2.5M params over ~25M (~10%); FLOPs 3.86->3.87 GFLOPs (+0.26%) at r=16. Most params in final stage (most channels); removing last-stage SE -> ~4% param increase, <0.1% top5 cost.

## Role (motivating empirical / diagnostic, about the SE-equipped net itself — keep out of context except where it's about existing systems)
Earlier layers: excitation class-agnostic (shared low-level). Later: class-specific. SE_5_2/5_3 saturate near 1 (≈identity) -> last-stage SE redundant. (This is proposed-method analysis — exclude from context.md; can appear as forward-looking "what I'd want to check" intent only, NOT as results, in reasoning.)

## Code (canonical: moskomule/senet.pytorch)
SELayer: AdaptiveAvgPool2d(1) -> Linear(C, C//r, bias=False) -> ReLU -> Linear(C//r, C, bias=False) -> Sigmoid; forward: y=pool(x).view(b,c); y=fc(y).view(b,c,1,1); return x*y.expand_as(x).
SEBottleneck: conv1(1x1)->bn->relu, conv2(3x3)->bn->relu, conv3(1x1)->bn, se(planes*4), then out+=residual, relu. SE on planes*4 channels before add.
