What I want is a generative model of natural images whose likelihood I can actually evaluate — given an image, I want $p(x)$ as a real number, exact, with no bound and no estimator standing in the way. That single requirement rules out most of the families I would otherwise reach for. A variational autoencoder defines $p(x) = \int p(x \mid z)\,p(z)\,dz$, an integral I cannot compute, so I would be training an evidence lower bound and reporting a bound rather than the thing itself; worse, the latent bottleneck tends to make the pixels conditionally independent given $z$, throwing away precisely the fine structure I want to capture. An RBM or DBM carries a partition function I cannot evaluate, so again no exact likelihood. The autoregressive estimators that *do* give an exact number — NADE, EoNADE, MADE — were built for small binary vectors with a single tied hidden layer or a fully-connected masked autoencoder, neither of which exploits the 2-D translation structure of images or scales to them. And the recurrent image models that reached natural images, RNADE and the spatial-LSTM RIDE model, emit a *continuous* density per pixel, which commits to a parametric shape on values that are genuinely discrete and leaks probability mass outside $[0, 255]$. So the gap is concrete: I need an exact, comparable likelihood, an expressive conditional that honors the discreteness of pixels, and an architecture that shares weights spatially and scales.

I propose PixelRNN and its fully-convolutional sibling PixelCNN. The starting point is the only factorization of a joint that needs no approximation, the chain rule. Fixing the raster-scan ordering — top row left to right, then the next row — flattens an $n\times n$ image into $x_1,\dots,x_{n^2}$ and gives
$$p(x) = \prod_{i=1}^{n^2} p(x_i \mid x_1,\dots,x_{i-1}),$$
so $\log p(x)$ is a sum of per-position log-probabilities. A pixel is not one value but three, so I apply the chain rule again *inside* the pixel with the order R $\to$ G $\to$ B, refusing to assert that the channels are independent given the context:
$$p(x_i \mid x_{<i}) = p(x_{i,R}\mid x_{<i})\cdot p(x_{i,G}\mid x_{<i},x_{i,R})\cdot p(x_{i,B}\mid x_{<i},x_{i,R},x_{i,G}).$$
This buys the central asymmetry I am happy to accept: at training time every "previous pixel" is known input, so all conditionals are computed in one parallel forward pass and their log-probs summed; only *generation* is sequential, drawing a value, feeding it back, and repeating $n^2$ times.

For the conditional's functional form I deliberately drop the continuity assumption and make each conditional a 256-way **softmax** over the intensities $\{0,\dots,255\}$ (a single sigmoid for binary images). This is the choice that beats the obvious continuous-density alternative: a Gaussian is unimodal, a mixture is only as multimodal as its components, and either way it imposes a smooth shape and puts mass below $0$ and above $255$ on impossible values. A free softmax is arbitrarily multimodal — it can spike on $0$ and on $255$ at once, as sky-and-shadow pixels demand — with no shape prior and no leaked mass, and its loss is exactly the categorical cross-entropy, which *is* the negative log-likelihood I want. The price is that the softmax does not know $51$ neighbors $52$; with enough data it simply learns smooth, peaked, or long-tailed shapes wherever they fit, a small cost I prefer to a parametric cage.

The hard part is the architecture: a net that at every position emits $p(x_i\mid x_{<i})$ and never reads $x_i$ or any later pixel. A convolution is the problem incarnate, because a $k\times k$ kernel centered at $i$ mixes a neighborhood that includes $x_i$ itself and pixels below and to the right; if it reads the answer, it can drive the loss to zero by copying. The fix is that a convolution's connectivity *is* its kernel, so I forbid a connection by **zeroing that weight** — a masked convolution. The allowed region is exactly the rows strictly above $i$ plus the part of $i$'s own row strictly to its left, so in a centered $k\times k$ kernel every weight in rows below center is zeroed, and the center row is zeroed from the center column onward. Two mask types are needed, and getting them right is load-bearing. In the first layer the center weight reads the raw value $x_i$, so it must be zeroed — call that **mask A**, which excludes the center. In every later layer the "center" carries a *feature* that, by induction, already depends only on $x_{<i}$, so reading it is not only safe but desirable, since it lets legitimate context flow straight up the stack — call that **mask B**, which keeps the center. Using mask A everywhere would shave one pixel of reachable context off at every layer and needlessly cripple the field, so it is mask A on the first layer only and mask B on all the rest. The mask also extends into the channel dimension to enforce R $\to$ G $\to$ B: with the feature maps split into three color groups, mask A lets group $g$ read only earlier groups (G $\leftarrow$ R, B $\leftarrow$ R,G, no self) and mask B lets $g$ read groups up to and including itself, which is exactly $p(R)\,p(G\mid R)\,p(B\mid R,G)$ written as connectivity.

Stacking masked convolutions gives a fast, parallel, exact-likelihood model, but it has a *bounded* receptive field — $L$ layers reach only about $L\cdot k$ pixels — so the conditional can never depend on the full earlier history. To reach unbounded context I turn to recurrence, whose hidden state summarizes everything seen so far. A naive 2-D LSTM is $O(n^2)$ sequential steps and too slow, so I claw back parallelism two ways. The **Row LSTM** processes a whole row per step: a masked length-$k$ row convolution computes the four gate pre-activations for the entire row at once, and a matching row-wise state-to-state convolution carries the recurrence from the row above, with the standard update
$$[o_i, f_i, i_i, g_i] = \text{activation}\!\left(K^{ss}\circledast h_{i-1} + K^{is}\circledast x_i\right),\quad c_i = f_i\odot c_{i-1} + i_i\odot g_i,\quad h_i = o_i\odot\tanh(c_i),$$
using $\sigma$ for the output, forget, and input gates and $\tanh$ for the content gate $g$. This is $n$ sequential steps instead of $n^2$, but tracing the field shows each row pulls a $k$-wide window from the row above, so the dependency region is a *triangle* — unbounded vertically, but blind to far-side pixels on earlier rows. To recover the *entire* valid context I **skew** the image, offsetting row $r$ rightward by $r$ so that a down-left diagonal of the original lands in one column; then a column convolution sweeps a whole diagonal family in parallel. The **Diagonal BiLSTM** uses a $1\times1$ input-to-state conv and a two-position $2\times1$ state-to-state conv, runs two mirrored scans, and — crucially for causality — shifts the right-direction map *down one row* before adding it to the left, so every contributed pixel is strictly previous in raster order. The two scans between them sweep the complete up-and-left region with no triangular gap, for any image size; the two-position kernel is the minimal nonlinear step and a larger one would only add parameters to an already-global field. To make deep stacks of these layers trainable I wrap each recurrent block in a residual connection — input $2h$ features reduced to $h$ per gate, recurrence, a $1\times1$ upsample back to $2h$, then add the input — giving signal and gradient a direct path without changing the probability model.

Staring at the pure masked-conv stack, I found a subtler failure than its bounded range, and it is worth fixing properly. A single masked kernel leans left: within a row it connects only leftward, and across rows it fans within that same left-leaning cone. Compose the masks over any depth and a wedge of pixels just to the *upper-right* of $i$ — legitimately in $x_{<i}$, since they sit on earlier rows — is *never reached*, because no chain of allowed connections routes up-and-right and back down to $i$. That is not a causality bug but a capacity bug: real context is silently amputated. Patching one cleverly-shaped mask fights the geometry, so the clean fix is to split the finite convolutional field into two stacks that are each easy to reason about. A **vertical stack** reads only rows above the current one, so its upper-right pixels are no longer amputated; a **horizontal stack** reads the current row strictly to the left. The vertical stack feeds the horizontal stack through a $1\times1$ link, but never the reverse, which preserves causality while letting the horizontal stack use above-row information from both sides. Implementing the link demands care that it exclude the current row: a $1\times k$ horizontal mix in the vertical stream, then a $(k/2+1)\times1$ convolution with extra top padding cropped back to height $H$, so the linked tensor carries only rows above; the horizontal stack gets its strictly-left field from a $1\times(k/2+1)$ convolution padded on the left and cropped back to width $W$. While redesigning the stack I also replace the flat ReLU with a **gated activation** borrowed from the LSTM, splitting the conv output and computing $\tanh(a)\odot\sigma(b)$ — a content path squashed by $\tanh$ and modulated multiplicatively by a sigmoid gate — which is a richer nonlinearity than ReLU. A residual connection eases optimization on the later horizontal layers, but it must be omitted on the very first causal input layer, where adding the input back would leak $x_i$. Training minimizes the summed per-pixel cross-entropy in parallel; sampling proceeds one pixel at a time in raster order.

```python
import torch
import torch.nn.functional as F
from torch import nn


class OrderedConv2d(nn.Conv2d):
    """Mask the kernel so position i never reads positions >= i.
    type 'A' excludes the center weight; type 'B' keeps it."""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class OrderedImageLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, first_layer=False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.first_layer = first_layer
        p = (kernel_size - 1) // 2
        self.act = GatedActivation()

        self.v_1xN = nn.Conv2d(in_ch, out_ch, (1, kernel_size), padding=(0, p))
        self.v_Nx1 = nn.Conv2d(out_ch, 2 * out_ch, (kernel_size // 2 + 1, 1),
                               padding=(p + 1, 0))
        self.v_1x1 = nn.Conv2d(in_ch, 2 * out_ch, 1)
        self.link = nn.Conv2d(2 * out_ch, 2 * out_ch, 1)

        self.h_1xN = nn.Conv2d(in_ch, 2 * out_ch, (1, kernel_size // 2 + 1),
                               padding=(0, p + int(first_layer)))
        self.h_res = nn.Conv2d(out_ch, out_ch, 1)
        self.h_skip = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, above_state, row_state):
        v_in, h_in = above_state, row_state
        _, _, H, W = v_in.shape
        v = self.v_Nx1(self.v_1xN(v_in))[:, :, :H, :]
        link = self.link(v)
        v = self.act(v + self.v_1x1(v_in))

        h = link + self.h_1xN(h_in)[:, :, :, :W]
        h = self.act(h)
        skip = self.h_skip(h)
        h = self.h_res(h)
        if not self.first_layer:
            h = h + h_in
        return v, h, skip


class ImageDensityModel(nn.Module):
    def __init__(self, in_ch=1, n_values=256, n_layers=10, channels=64,
                 split_state=True, head_channels=64):
        super().__init__()
        self.split_state = split_state
        if split_state:
            self.input_layer = OrderedImageLayer(in_ch, channels, kernel_size=7, first_layer=True)
            self.layers = nn.ModuleList(
                OrderedImageLayer(channels, channels, kernel_size=3, first_layer=False)
                for _ in range(n_layers)
            )
            head_in = channels
        else:
            self.input_layer = OrderedConv2d("A", in_ch, channels, 7, padding=3, bias=False)
            self.layers = nn.ModuleList(
                OrderedConv2d("B", channels, channels, 3, padding=1, bias=False)
                for _ in range(n_layers)
            )
            head_in = channels
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv2d(head_in, head_channels, 1),
            nn.ReLU(), nn.Conv2d(head_channels, n_values, 1)
        )

    def forward(self, x):
        if self.split_state:
            v, h, skip = self.input_layer(x, x)
            for layer in self.layers:
                v, h, s = layer(v, h)
                skip = skip + s
            return self.head(skip)

        h = F.relu(self.input_layer(x))
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.head(h)


def nll_loss(logits, target_pixels):
    per_pixel = F.cross_entropy(logits, target_pixels, reduction="none")
    return per_pixel.flatten(1).sum(1).mean()


def train_step(model, x, opt):
    target = (x[:, 0] * 255).long()
    loss = nll_loss(model(x), target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def sample(model, n, H, W, n_values=256):
    device = next(model.parameters()).device
    img = torch.zeros(n, 1, H, W, device=device)
    for i in range(H):
        for j in range(W):
            logits = model(img)[:, :, i, j]
            probs = F.softmax(logits, dim=1)
            img[:, 0, i, j] = torch.multinomial(probs, 1).squeeze(1).float() / (n_values - 1)
    return img
```
