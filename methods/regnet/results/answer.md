# RegNet

## Problem

Manual architecture design yields transferable principles but is hard to scale across many interacting choices; NAS finds a strong single network instance but no principle and no rule for adapting to a new compute budget. RegNet is the outcome of a third paradigm — *designing the design space*: progressively simplifying an unconstrained space of network structures into a low-dimensional space of simple, regular networks that is densely populated with good models, interpretable, and generalizes across compute regimes.

## Key idea

Measure a *design space* (a population of architectures) by sampling n models, training them in a cheap regime (400 MFLOPs, 10 epochs), and comparing the error empirical distribution functions, F(e) = (1/n) Σ_i 1[e_i < e]. A space with a lower EDF is better. Starting from AnyNetX (4 stages × {depth d_i, width w_i, bottleneck ratio b_i, group width g_i} = 16 dof, ~10¹⁸ configs), apply refinement steps that simplify the space while keeping or improving the EDF:

1. **Share b** across stages (AnyNetX_B) — EDF unchanged.
2. **Share g** across stages (AnyNetX_C) — EDF unchanged.
3. **w_{i+1} ≥ w_i** (AnyNetX_D) — EDF improves; good nets have increasing widths.
4. **d_{i+1} ≥ d_i** (AnyNetX_E) — EDF improves; depths also increase.

Inspecting the per-block width staircases of the best models, they hug a line. Capture this with a **quantized linear parametrization**: a continuous per-block width u_j = w_0 + w_a·j (0 ≤ j < d), re-expressed multiplicatively as u_j = w_0·w_m^{s_j} ⇒ s_j = log(u_j/w_0)/log(w_m), then quantized by rounding: w_j = w_0·w_m^{⌊s_j⌉}. Blocks sharing ⌊s_j⌉ form a stage of width w_i = w_0·w_m^i and depth d_i = #{j : ⌊s_j⌉ = i}. A network is then fully specified by **6 scalars**: d, w_0, w_a, w_m, b, g. The space of these regular networks is **RegNet** (~3×10⁸ configs, 10 orders of magnitude smaller than AnyNetX_A), with a better EDF, far higher random-search efficiency (~32 samples suffice), and no overfitting across higher FLOPs/epochs, 5 stages, or other block types.

**Design principles read off RegNetX** (analysis at 100 models / 25 epochs): best depth is stable at ~20 blocks across regimes; best bottleneck ratio b = 1.0 (no bottleneck); width multiplier w_m ≈ 2.5; g, w_a, w_0 grow with FLOPs. Activations ∝ √FLOPs and matter for runtime on memory-bound accelerators, so cap activations/params alongside FLOPs. Inverted bottleneck (b<1) and depthwise (g=1) hurt; fixed 224×224 resolution is best. Adding Squeeze-and-Excitation gives the **Y block / RegNetY**, which improves the EDF (SE reduction ratio 0.25).

## Code

```python
import numpy as np
import torch.nn as nn

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Six scalars -> per-stage widths and depths via the quantized linear function."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0                          # u_j = w_0 + w_a * j
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))          # s_j = log(u_j/w_0)/log(w_m)
    ws_all = w_0 * np.power(w_m, ks)                            # w_j = w_0 * w_m^round(s_j)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q     # snap to multiple of q
    ws, ds = np.unique(ws_all, return_counts=True)              # -> per-stage w_i and d_i
    return ws.tolist(), ds.tolist(), len(ws)

def adjust_block_compatibility(ws, bs, gs):
    """Ensure bottleneck width w*b is divisible by group width g."""
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    return ws, bs, gs

def conv2d(w_in, w_out, k, stride=1, groups=1):
    return nn.Conv2d(w_in, w_out, k, stride=stride, padding=(k - 1) // 2, groups=groups, bias=False)
def norm2d(w): return nn.BatchNorm2d(w)
def activation(): return nn.ReLU(inplace=True)
def gap2d(): return nn.AdaptiveAvgPool2d((1, 1))

class SE(nn.Module):
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = gap2d()
        self.f_ex = nn.Sequential(conv2d(w_in, w_se, 1), activation(),
                                  conv2d(w_se, w_in, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """1x1 -> 3x3 group conv [+SE] -> 1x1."""
    def __init__(self, w_in, w_out, stride, bot_mul, group_w, se_r):
        super().__init__()
        w_b = int(round(w_out * bot_mul))
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1); self.a_bn = norm2d(w_b); self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups); self.b_bn = norm2d(w_b); self.b_af = activation()
        self.se = SE(w_b, int(round(w_in * se_r))) if se_r else None
        self.c = conv2d(w_b, w_out, 1); self.c_bn = norm2d(w_out)
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBottleneckBlock(nn.Module):
    def __init__(self, w_in, w_out, stride, bot_mul, group_w, se_r):
        super().__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride); self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, bot_mul, group_w, se_r)
        self.af = activation()
    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))

class AnyStage(nn.Module):
    def __init__(self, w_in, w_out, stride, d, bot_mul, group_w, se_r):
        super().__init__()
        for i in range(d):
            self.add_module(f"b{i+1}", ResBottleneckBlock(w_in, w_out, stride, bot_mul, group_w, se_r))
            stride, w_in = 1, w_out
    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class RegNet(nn.Module):
    def __init__(self, w_a, w_0, w_m, depth, bot_mul=1.0, group_w=8, se_r=0.0,
                 stem_w=32, num_classes=1000):
        super().__init__()
        ws, ds, _ = generate_regnet(w_a, w_0, w_m, depth)
        bs, gs = [bot_mul] * len(ws), [group_w] * len(ws)
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
        self.stem = nn.Sequential(conv2d(3, stem_w, 3, stride=2), norm2d(stem_w), activation())
        prev_w = stem_w
        for i, (d, w, b, g) in enumerate(zip(ds, ws, bs, gs)):
            self.add_module(f"s{i+1}", AnyStage(prev_w, w, 2, d, b, g, se_r))
            prev_w = w
        self.head = nn.Sequential(gap2d(), nn.Flatten(), nn.Linear(prev_w, num_classes))
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

# RegNetX-200MF: w_a=36.44, w_0=24, w_m=2.49, depth=13, group_w=8, bot_mul=1.0
# RegNetY = same with se_r=0.25
```

RegNetX/Y models are obtained by sampling the 6 RegNet parameters, picking the best of ~25 random settings per FLOP regime, and retraining (e.g. 100 epochs, SGD with momentum, cosine schedule). Under comparable training and FLOPs, RegNet models match or beat EfficientNet while being substantially faster on GPUs.
