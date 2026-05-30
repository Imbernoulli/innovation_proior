# VQ-VAE-2

## Problem

Generate high-resolution, high-fidelity, *diverse* images while keeping the measurability and full mode coverage of likelihood-based training. Pixel-space likelihood models waste capacity on perceptually negligible local detail and are far too slow to sample at high resolution; GANs are sharp but drop modes and lack a generalization measure.

## Key idea

Two-stage generation in a compressed discrete latent space.

1. **Compress, hierarchically.** Train a vector-quantized autoencoder that maps an image to a *hierarchy* of discrete code grids: a small **top** grid for global structure and a larger **bottom** grid for local detail. The bottom level is conditioned on the top level *and* on the image, so the two levels carry complementary information rather than the bottom merely refining the top. A fast feed-forward decoder with MSE reconstruction maps the codes back to pixels (no autoregressive pixel decoder, so no hierarchy collapse).

2. **Model the codes.** After the autoencoder is frozen, fit a powerful autoregressive prior over each level's codes. The top prior is a gated PixelCNN interleaved with causal multi-head self-attention (affordable on the small grid, and needed for the long-range correlations of global structure). The bottom prior is a conditional gated PixelCNN with no attention (prohibitive and unnecessary at the larger resolution, since it is conditioned on the top code), fed by a deep residual conditioning stack from the top.

Sampling is ancestral — top, then bottom given top — followed by a single feed-forward decode, an order of magnitude faster than sampling in pixel space. An optional post-hoc filter re-scores class-conditional samples with a pretrained ImageNet classifier and keeps the top-scoring fraction, a diversity-vs-quality dial.

## Objective and quantizer

Nearest-prototype quantization: `Quantize(E(x)) = e_k`, `k = argmin_j ||E(x) − e_j||`. Per level, the loss is
`L = ||x − D(e)||₂² + ||sg[E(x)] − e||₂² + β·||sg[e] − E(x)||₂²`,
with `sg[·]` stop-gradient and gradients reaching the encoder through the quantizer via the straight-through estimator `z_q = z_e + sg[e_k − z_e]`. The codebook (second) term is implemented as an exponential-moving-average update of each prototype toward the mean of the encoder outputs assigned to it (online k-means), decay `γ = 0.99`. Commitment weight `β = 0.25`, codebook size `K = 512`, code dimension `D = 64`. The deterministic one-hot posterior with a fixed uniform prior makes the KL the constant `log K`, eliminating posterior collapse.

## Code

```python
import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim, self.n_embed, self.decay, self.eps = dim, n_embed, decay, eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True))
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(channel, in_channel, 1))
    def forward(self, x):
        return self.conv(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel // 2, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks += [nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                       nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)]
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)


class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        return self.decode(quant_t, quant_b), diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2); diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2); diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        return self.dec(quant)

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t).permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b).permute(0, 3, 1, 2)
        return self.decode(quant_t, quant_b)


# Stage 1 training
criterion = nn.MSELoss()
latent_loss_weight = 0.25
for img in loader:
    out, latent_loss = model(img)
    recon_loss = criterion(out, img)
    loss = recon_loss + latent_loss_weight * latent_loss.mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Stage 2 fits a class-conditional gated-PixelCNN-with-self-attention prior over the 32×32 top codes and a conditional gated PixelCNN (no attention, deep residual conditioning stack from the top code) over the 64×64 bottom codes; generation samples top, then bottom-given-top, decodes once, and optionally keeps only samples a pretrained ImageNet classifier scores highly for their intended class.
