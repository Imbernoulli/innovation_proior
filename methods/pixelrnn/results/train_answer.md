I propose to call this method PixelRNN, with PixelCNN as its fast, fully convolutional sibling. The goal is to build a probability model p(x) over natural images that is at once exact and tractable — I want to compute the exact likelihood of any image and to sample new images — while still being expressive enough to capture the nonlinear, long-range, multimodal dependencies between pixels, including the dependencies among the R, G, and B channels inside a single pixel.

The starting point is the autoregressive factorization. I order the pixels in raster scan, row by row and left to right, and apply the chain rule:

p(x) = Π_i p(x_i | x_1, …, x_{i-1}).

Because there are no latent variables, there is no approximate inference and no lower bound: the likelihood is exactly this product. Sampling is ancestral — draw x_1, then x_2 given x_1, and so on — so a model that can evaluate p(x_i | x_{<i}) can also generate. The entire difficulty is therefore transferred to the conditional model: how well can I represent p(x_i | x_{<i}) for every position i?

A pixel, however, is not a single scalar. In a color image each position carries three channels, and those channels are correlated. I therefore apply the chain rule once more inside each pixel, ordering the channels R → G → B:

p(x_i | x_{<i}) = p(x_{i,R} | x_{<i}) · p(x_{i,G} | x_{<i}, x_{i,R}) · p(x_{i,B} | x_{<i}, x_{i,R}, x_{i,G}).

Green sees the red value of the same pixel, and blue sees both red and green. This means the causal mask must enforce not only a spatial ordering but also a channel ordering at the current pixel.

Next I decide the form of each conditional. Earlier autoregressive image models often used continuous densities such as Gaussian or mixture densities over intensities. That forces a parametric shape on the conditional and makes multimodality awkward, yet real pixel conditionals are frequently multimodal — an edge pixel might be very dark or very light but rarely in between. Pixel intensities are actually drawn from a fixed set of 256 values, so I model each conditional as a discrete 256-way softmax. The softmax can represent arbitrary, spiky, or bimodal distributions with no shape assumption, and it trains cleanly as a classification problem.

Training and generation have opposite parallelization properties. At training time the true image is known, so I would like to compute every conditional p(x_i | x_{<i}) in parallel. At generation time the process is unavoidably sequential, because each sampled pixel must be fed back before the next can be drawn. The architecture must therefore allow parallel training while strictly respecting causality: position i must never depend on itself or on any future pixel. If even a trace of x_i leaked into the prediction of x_i, the likelihood would be invalid and generation would be inconsistent.

To enforce causality I use masked convolutions. A standard convolution centered at a pixel looks at neighbors in all directions, including the pixel itself and pixels below and to the right. I zero out the forbidden weights in the kernel before convolving. For a kernel centered on the pixel being predicted, I keep the weights for pixels above and to the left, zero the center and everything below and to the right. Because the input layer is special — the input is the raw image itself — I use a strict mask, called mask A, which additionally blocks a channel from connecting to itself at the current pixel. After the first layer the activations are features, not raw pixels, and the raw-value leak is structurally impossible, so I can relax to mask B, which keeps the center connection and only zeros the future. This A/B distinction is the key to preserving both causality and model capacity.

Stacking masked convolutions while preserving spatial resolution gives the PixelCNN. It is the fastest member of the family because every position is computed in one forward pass during training, but its receptive field is bounded by the depth and size of the stack. For long-range structure that is a limitation, so I also consider recurrent layers.

The Row LSTM processes the image row by row. For each row it precomputes the input-to-state contributions for the four LSTM gates with a masked k×1 convolution over the whole map, then advances the LSTM state down the rows using a state-to-state convolution along each row. The dependency region above a pixel is triangular, which captures more context than a small bounded convolution but still misses pixels far to the sides in nearby rows.

To reach the entire available context I use the Diagonal BiLSTM. The trick is to skew the image so that each row is offset by one position relative to the row above. In this skewed space the original diagonals become straight columns, so a column-wise convolution sweeps along diagonals. The input-to-state part is a 1×1 convolution, and the state-to-state part is a tiny 2×1 convolution; the small kernel is sufficient because the diagonal sweep already provides a global receptive field. One diagonal direction covers everything above and to the left, and a second mirror direction covers above and to the right. Before combining the two directions I shift the second output down by one row so that it cannot see the current or future pixels. The result is that every pixel sees the full legal context above and to both sides, with no access to the future.

Deep stacks of recurrent layers are hard to train, so I add residual connections around each LSTM layer. The input map has 2h features, the recurrent layer operates at h features per gate, and a 1×1 convolution brings the output back to 2h features so that the input can be added. Residual connections let me train networks with up to twelve LSTM layers. For very large images I also use a multi-scale variant: an unconditional PixelRNN first generates a small low-resolution image, and a second PixelRNN conditions on that small image through biased gates to generate the full-resolution image. The low-resolution image injects global layout cheaply, while the high-resolution network fills in the details.

The concrete architecture is as follows. The first layer is a 7×7 masked convolution with mask A. The body is then either a stack of 3×3 mask-B convolutions for PixelCNN, a stack of Row LSTM layers with input-to-state 3×1 mask-B and state-to-state 3×1 unmasked convolutions, or a stack of Diagonal BiLSTM layers with input-to-state 1×1 mask-B and state-to-state 1×2 unmasked convolutions. After the body come two ReLU plus 1×1 mask-B convolution layers, using 1024 feature maps for CIFAR-10 and ImageNet and 32 for MNIST, followed by a 256-way softmax per color channel, or a single sigmoid for binary MNIST. Typical configurations include a 7-layer Diagonal BiLSTM with h=16 for MNIST, a 12-layer Row or Diagonal BiLSTM with h=128 for CIFAR-10, and a 12-layer Row LSTM with h=384 for 32×32 ImageNet.

Training maximizes the exact log-likelihood. Because all true pixels are known, every conditional is computed in parallel in one forward pass, and the loss is simply the 256-way cross-entropy between the predicted distribution and the true discrete pixel value. Generation is sequential: starting from a blank canvas, I draw each pixel in raster order and feed the sampled value back into the model before moving to the next position.

Below is the complete implementation: the masked convolution that enforces causality, a PixelCNN built by stacking it, the Row LSTM module that reuses the same masked convolution for its input-to-state gates, the training loop that turns the exact likelihood into a 256-way cross-entropy computed in parallel over every pixel, and the sequential sampling routine that feeds each generated pixel back into the network.

```python
import torch
from torch import nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0   # center row: self (A) / future
        self.mask[:, :, kH // 2 + 1:] = 0                            # rows below center: future
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# PixelCNN: stack of masked convs, resolution preserved, 256-way head
fm = 128
pixelcnn = nn.Sequential(
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    *[m for _ in range(7) for m in (
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))],
    nn.Conv2d(fm, 256, 1))


class RowLSTM(nn.Module):
    def __init__(self, h, k=3):
        super().__init__()
        self.input_to_state = MaskedConv2d('B', h, 4 * h, (k, 1), 1, (k // 2, 0))
        self.state_to_state = nn.Conv1d(h, 4 * h, k, padding=k // 2)
        self.h = h
    def forward(self, x):                      # x: [B, h, H, W]
        gates_is = self.input_to_state(x)
        B, _, H, W = x.shape
        h_prev = x.new_zeros(B, self.h, W); c_prev = x.new_zeros(B, self.h, W)
        outs = []
        for i in range(H):
            g = gates_is[:, :, i] + self.state_to_state(h_prev)
            o, f, ii, gg = g.chunk(4, dim=1)
            c_prev = torch.sigmoid(f) * c_prev + torch.sigmoid(ii) * torch.tanh(gg)
            h_prev = torch.sigmoid(o) * torch.tanh(c_prev)
            outs.append(h_prev)
        return torch.stack(outs, dim=2)        # [B, h, H, W]


# training: exact likelihood as 256-way cross-entropy, conditionals in parallel
optimizer = torch.optim.Adam(pixelcnn.parameters())
for img in loader:                             # img in [0,1], shape [B,1,H,W]
    target = (img[:, 0] * 255).long()          # discrete target in {0..255}
    logits = pixelcnn(img)                     # [B, 256, H, W]
    loss = F.cross_entropy(logits, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()


# generation: sequential, feed each sampled pixel back in
@torch.no_grad()
def sample(net, n, H, W):
    x = torch.zeros(n, 1, H, W)
    for r in range(H):
        for c in range(W):
            probs = F.softmax(net(x)[:, :, r, c], dim=1)
            x[:, 0, r, c] = torch.multinomial(probs, 1).float() / 255.
    return x
```

The `MaskedConv2d` is exactly the masked-convolution idea argued above, encoded as a buffer multiplied into the weights before every forward pass, so mask type A blocks the center pixel (used only on the raw input) and mask type B keeps it (safe everywhere after that first layer). Stacking eight of these with resolution preserved throughout, no pooling, and a final `1×1` convolution to 256 logits per pixel gives the PixelCNN. The `RowLSTM` module makes the input-to-state contribution for all four gates in one masked convolution over the whole map, then walks down the rows applying the state-to-state convolution and the ordinary LSTM update, exactly the recurrence written out earlier. Training is a single forward pass per image: the true pixel values become the classification target and the loss is the 256-way cross-entropy summed over every position at once, since every conditional is legally computable in parallel when the ground truth is known. Sampling cannot enjoy that parallelism — it walks the image in raster order, row by row and column by column, drawing each pixel from its 256-way softmax and writing the drawn value back into the input tensor before the network is asked to predict the next position. This loop is the same causal wiring in the other direction: the network never gets to see a pixel before it has been generated, so the samples it produces are draws from the exact model it was trained to maximize the likelihood of.
