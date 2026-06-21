Every convolutional block already mixes channels: the $c$-th output map is $u_c = v_c * X = \sum_{s} v_c^s * x^s$, a sum over the input channels weighted by the learned filter. But that mixing has three properties I want to change. It is buried inside the spatial filters, so the same coefficients have to specify both the spatial pattern and the cross-channel composition at once; each response is computed from a finite receptive field, so in lower and middle layers no local decision can consult the whole image before deciding how to combine channels; and once training fixes the weights, the channel composition is identical for every input — the operator has no content-dependent channel policy. The tools already on the table touch this without isolating it. A $1\times 1$ convolution recombines channels cheaply but with fixed weights applied independently at each location. Grouped and multi-branch designs (Inception, ResNeXt) enlarge the set of local transformations but the selected compositions stay frozen after training. Batch Normalization's per-channel affine scale is structurally close to channel modulation, yet its scale is a learned constant applied uniformly at inference rather than a function of the current example. Attention and gating precedents — recurrent visual attention, spatial transformers, Highway gates, residual-attention trunk-and-mask modules — either bias *where* a model samples in space or rely on comparatively heavy auxiliary subnetworks, so none is a minimal, drop-in way to make a residual block treat channels differently for different inputs. What is missing is exactly that: an input-conditioned, globally informed quantity, kept separate from the convolution weights, that modulates the channels the block just produced.

I propose the Squeeze-and-Excitation block. The idea is to attach one input-dependent scalar to each output channel and let it rescale that channel: if a channel is useful for the current input, more of it passes; if not, it is attenuated. I make the modulation *multiplicative* rather than additive on purpose — an additive offset would inject a new signal into the feature map, whereas multiplying the feature already computed by the residual branch directly changes the strength of an existing feature, which is the semantics I want. So the recalibrated channel is $\tilde{x}_c = s_c\, u_c$ with a positive gate $s_c$. The whole block is built from three operations: squeeze, excite, scale. The squeeze must summarize a whole $H\times W$ map into one number per channel, because the gate has to depend on more than the single local window that produced one entry of $u_c$. I use the spatial average

$$z_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} u_c(i,j),$$

and the constant $1/(HW)$ is load-bearing: it is an *average*, not a raw sum, so the descriptor's scale does not grow simply because a feature map has more spatial positions. A max would only record the single strongest location; the average records the channel's overall response across the full map, so it is the simpler and more faithful first aggregation. This produces a descriptor $z \in \mathbb{R}^{C}$, one statistic per channel.

The excitation turns $z$ into the gate vector $s$. Mapping each $z_c$ independently to $s_c$ would throw away relationships among channels, but the entire point is to model channel *dependencies*, so the gate for one channel must be allowed to depend on the whole descriptor — a learned map from $\mathbb{R}^C$ to $\mathbb{R}^C$. A full $C\times C$ matrix would cost $C^2$ weights in every block, which is far too much for a lightweight side computation, especially in late stages where $C$ is large. I therefore factor the map through a smaller hidden width: reduce $C$ to $C/r$, apply a ReLU $\delta$, then expand back to $C$, giving gate logits $W_2\,\delta(W_1 z)$ with $W_1 \in \mathbb{R}^{(C/r)\times C}$ and $W_2 \in \mathbb{R}^{C\times(C/r)}$. The weight count per block is $C(C/r) + (C/r)C = 2C^2/r$, before any optional biases. The final nonlinearity has to let several channels be high at once, which rules out a softmax: softmax forces the gates to sum to one, so raising one channel would suppress others even when several features are simultaneously useful. A componentwise sigmoid avoids that competition, giving each channel its own independent gate in $(0,1)$; ReLU would be unbounded at the top, and tanh would admit negative gates unless shifted, so the sigmoid carries the cleanest meaning for channel attenuation. The excitation is therefore

$$s = \sigma\!\left(W_2\,\delta(W_1 z)\right), \qquad \tilde{x}_c = s_c\, u_c.$$

There is no missing sign and no additive residual hidden inside the recalibration — the scale is positive because the sigmoid is positive, and the only spatial constant anywhere is the $1/(HW)$ in the squeeze. The reduction ratio $r$ enters only through the hidden width and the parameter count; the standard setting is $r=16$.

The last design decision is placement inside a residual block. The shortcut is the path that keeps signal and gradient flow simple, so I must not casually rescale it. If I multiplied the full sum $F(x) + \text{shortcut}(x)$ I would also multiply the shortcut, which is not the default I want from a drop-in module. Instead I take the non-identity branch $U = F(x)$ to be the thing being recalibrated, apply both squeeze and excitation to it, and add the shortcut afterward, so the block returns $\mathrm{ReLU}(\tilde{U} + \text{shortcut}(x))$. The implementation order is then dictated by the tensor algebra: global-average-pool to $(B,C,1,1)$, flatten to $(B,C)$, push through the two channel-mixing linear maps with ReLU and sigmoid, reshape the gate to $(B,C,1,1)$, multiply the residual-branch tensor, and only then add the shortcut. (A $1\times1$ convolution on a $1\times1$ pooled map and a linear layer on the flattened descriptor are the same channel map written in two framework dialects.) The cost stays small: across stages $s$ with $N_s$ blocks and $C_s$ channels the added weights are $(2/r)\sum_s N_s C_s^2$, with optional FC biases adding only $\sum_s N_s(C_s/r + C_s)$ beyond that, and the extra FLOPs are one pooling, two matrix-vector products, a sigmoid, and a channelwise multiply per example. For the CIFAR scaffold, $\text{expansion}=1$, so the second convolution emits $\text{planes}$ channels and the gate has length $\text{planes}$; with $r=16$ the canonical widths $16, 32, 64$ give integer hidden widths $1, 2, 4$ exactly as $\text{channels}//\text{reduction}$ produces. What makes the block work is that the convolution remains the main feature extractor while the SE unit gives every block an input-conditioned, globally informed handle on the channels it just computed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s


class CustomBlock(nn.Module):
    """Basic residual block with channel recalibration."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.se = SELayer(planes, reduction=16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)
```
